"""Phase 3b : index FAISS en produit scalaire sur embeddings normalisés, soit la similarité cosinus.

Index vidéo : un vecteur par vidéo (moyenne de ses segments).
Index texte : un vecteur par phrase d'annotation, avec la vidéo d'origine dans une base à côté.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _as_row(vec) -> np.ndarray:
    import faiss

    row = np.ascontiguousarray(vec, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(row)
    return row


def build_video_index(features_dir: Path, video_files: list[Path], embed_dim: int):
    import faiss

    index, kept = faiss.IndexFlatIP(embed_dim), []
    for path in video_files:
        f = Path(features_dir) / f"{Path(path).stem}.pt"
        if not f.exists():
            continue
        mean = torch.load(f, map_location="cpu", weights_only=True).mean(dim=0).numpy()
        if mean.shape[0] == embed_dim:
            index.add(_as_row(mean))
            kept.append(str(path))
    return index, kept


def default_sentences(video_id: str, ann: dict) -> list[str]:
    if ann.get("sentences"):
        return [s for s in ann["sentences"] if isinstance(s, str) and s]
    if ann.get("dataset") == "cadp":
        return ["Traffic accident scene" if ann.get("is_anomaly") else "Normal traffic scene"]
    if ann.get("dataset") == "dota":
        return [f"Traffic anomaly: {ann.get('anomaly_class', 'UK')}"]
    return [f"Video: {video_id}"]


@torch.no_grad()
def build_text_index(annotations: dict[str, dict], model, tokenizer, device: str, embed_dim: int,
                     batch_size: int = 256):
    import faiss

    database = [{"video": vid, "sentence_idx": i, "text": s, "dataset": ann.get("dataset", "ucf_crime")}
                for vid, ann in annotations.items() for i, s in enumerate(default_sentences(vid, ann))]
    index = faiss.IndexFlatIP(embed_dim)
    for start in range(0, len(database), batch_size):
        texts = [d["text"] for d in database[start:start + batch_size]]
        emb = F.normalize(model.encode_text(tokenizer(texts).to(device)), dim=-1)
        index.add(emb.float().cpu().numpy().astype("float32"))
    return index, database


def save(index, items, index_path: Path, items_path: Path) -> None:
    import faiss

    faiss.write_index(index, str(index_path))
    with open(items_path, "wb") as f:
        pickle.dump(items, f)


def load(index_path: Path, items_path: Path):
    import faiss

    index = faiss.read_index(str(index_path))
    with open(items_path, "rb") as f:
        items = pickle.load(f)
    return index, items[: index.ntotal]
