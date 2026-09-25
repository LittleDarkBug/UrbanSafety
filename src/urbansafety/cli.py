"""Point d'entrée : ``python -m urbansafety <commande>`` ou ``urbansafety <commande>``.

Commandes, dans l'ordre du pipeline : download, captions, finetune, index, train, evaluate, demo.
Chaque étape lit les artefacts de la précédente dans le répertoire de travail (``--work``).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import config


def _clip(paths, dev, finetuned=True):
    from .models.clip_lora import apply_lora, freeze, load_clip, load_finetuned

    model, preprocess, tokenizer = load_clip(device=dev)
    model = apply_lora(model).to(dev)
    if finetuned and not load_finetuned(model, paths.clip_weights, dev):
        print(f"poids LoRA absents ({paths.clip_weights}) : poids pré-entraînés {config.CLIP_PRETRAINED}")
    return freeze(model), preprocess, tokenizer


def _loaders(paths, splits):
    from torch.utils.data import DataLoader

    from .data.annotations import clip_labels
    from .data.features import FeatureSequenceDataset, collate_padded

    everything = {**splits["train"], **splits["val"], **splits["test"]}

    def ds(split, labels=None):
        d = FeatureSequenceDataset(paths.video_embeddings, everything, split, config.LSTM_DATASETS, labels)
        print(f"{split} : {len(d)} vidéos {d.counts()}")
        return d

    train = DataLoader(ds("train", clip_labels(splits["train"])), batch_size=16, shuffle=True,
                       collate_fn=collate_padded, drop_last=True)
    val = DataLoader(ds("val"), batch_size=16, collate_fn=collate_padded)
    test = DataLoader(ds("test"), batch_size=16, collate_fn=collate_padded)
    return train, val, test


def cmd_download(paths, args):
    from .data.download import download_all

    print(download_all(paths))


def cmd_captions(paths, args):
    from .captioning import generate_captions, load_slm, release

    model, tokenizer = load_slm(args.slm)
    for split, out in (("train", paths.captions_train), ("val", paths.captions_val)):
        f = paths.bdd100k / split / "annotations" / f"bdd100k_labels_images_{split}.json"
        if f.exists():
            captions = generate_captions(model, tokenizer, json.loads(f.read_text()), out, batch_size=args.batch_size)
            print(f"{split} : {len(captions)} captions")
    release(model, tokenizer)


def cmd_finetune(paths, args):
    import torch
    from torch.utils.data import DataLoader, Subset

    from .data.bdd100k import CaptionDataset
    from .models.clip_lora import apply_lora, finetune, load_clip

    dev = config.device()
    model, preprocess, tokenizer = load_clip(device=dev)
    model = apply_lora(model).to(dev)
    data = CaptionDataset(paths.bdd100k / "train" / "images", paths.captions_train, preprocess)
    subset = Subset(data, torch.randperm(len(data))[: args.subset].tolist())
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    finetune(model, tokenizer, loader, epochs=args.epochs, device=dev)
    torch.save(model.state_dict(), paths.clip_weights)
    print(f"poids LoRA : {paths.clip_weights}")


def cmd_index(paths, args):
    from . import indexing
    from .data.annotations import load_all
    from .embeddings import extract_all

    dev = config.device()
    model, preprocess, tokenizer = _clip(paths, dev)
    dim = model.visual.output_dim
    splits = load_all(paths)
    everything = {**splits["train"], **splits["val"], **splits["test"]}
    print(extract_all(paths, everything, model, preprocess, dev, dim))
    videos = sorted(Path(paths.ucf).rglob("*.mp4"))
    index, kept = indexing.build_video_index(paths.video_embeddings, videos, dim)
    indexing.save(index, kept, paths.video_index, paths.video_paths)
    index, db = indexing.build_text_index(everything, model, tokenizer, dev, dim)
    indexing.save(index, db, paths.text_index, paths.text_database)
    print(f"index vidéo : {len(kept)} vidéos, index texte : {len(db)} phrases")


def cmd_train(paths, args):
    import torch

    from .data.annotations import load_all
    from .evaluation import evaluate
    from .training import TrainConfig, train_detector

    dev = config.device()
    train, val, _ = _loaders(paths, load_all(paths))
    dim = next(iter(train))[0].shape[-1]
    runs = {"base": train_detector(train, val, dim, TrainConfig(epochs=20), dev),
            "pgd": train_detector(train, val, dim, TrainConfig(epochs=15, adversarial=True), dev)}
    scores = {name: evaluate(r.model, val, dev) for name, r in runs.items()}
    best = max(scores, key=lambda n: scores[n]["auc"])
    for name, s in scores.items():
        print(f"{name} : AUC {s['auc']:.4f}, F1 {s['f1']:.4f}")
    torch.save(runs[best].model.state_dict(), paths.detector_weights)
    (paths.work / "histories.json").write_text(json.dumps({n: r.history for n, r in runs.items()}))
    print(f"modèle retenu : {best} ({paths.detector_weights})")


def cmd_evaluate(paths, args):
    import torch

    from .data.annotations import load_all
    from .evaluation import evaluate, plot_report
    from .models.detector import AnomalyDetector

    dev = config.device()
    _, _, test = _loaders(paths, load_all(paths))
    dim = next(iter(test))[0].shape[-1]
    detector = AnomalyDetector(input_dim=dim).to(dev)
    detector.load_state_dict(torch.load(paths.detector_weights, map_location=dev))
    res = evaluate(detector, test, dev)
    print(json.dumps({k: v for k, v in res.items() if k not in ("scores", "labels")}, indent=2))
    hist_file = paths.work / "histories.json"
    histories = json.loads(hist_file.read_text()) if hist_file.exists() else {}
    plot_report(histories, res["scores"], res["labels"], paths.work / "evaluation.png")


def cmd_demo(paths, args):
    import torch

    from . import indexing
    from .app import build_demo
    from .models.detector import AnomalyDetector
    from .retrieval import SurveillanceSystem

    dev = config.device()
    model, preprocess, tokenizer = _clip(paths, dev)
    detector = AnomalyDetector(input_dim=model.visual.output_dim).to(dev)
    detector.load_state_dict(torch.load(paths.detector_weights, map_location=dev))
    vi, vp = indexing.load(paths.video_index, paths.video_paths)
    ti, td = indexing.load(paths.text_index, paths.text_database)
    system = SurveillanceSystem(model, preprocess, tokenizer, detector, vi, vp, ti, td, dev)
    build_demo(system).launch(share=args.share)


COMMANDS = {"download": cmd_download, "captions": cmd_captions, "finetune": cmd_finetune, "index": cmd_index,
            "train": cmd_train, "evaluate": cmd_evaluate, "demo": cmd_demo}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="urbansafety", description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--mode", default="local", choices=("local", "kaggle"))
    parser.add_argument("--work", default=None, help="répertoire des artefacts (défaut : selon le mode)")
    parser.add_argument("--slm", default=config.SLM_MODEL_ID, help="modèle de captions (phase 1)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--subset", type=int, default=1000, help="nombre d'images BDD100K pour LoRA")
    parser.add_argument("--epochs", type=int, default=12, help="epochs de fine-tuning LoRA")
    parser.add_argument("--share", action="store_true", help="lien Gradio public")
    args = parser.parse_args(argv)
    config.seed_everything()
    paths = config.Paths.for_mode(args.mode, args.work)
    paths.work.mkdir(parents=True, exist_ok=True)
    COMMANDS[args.command](paths, args)


if __name__ == "__main__":
    main()
