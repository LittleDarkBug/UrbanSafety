"""Phase 3a : embeddings OpenCLIP par segment de 5 secondes.

Vidéos (UCF-Crime) : 5 images réparties dans chaque segment, encodées puis moyennées.
Séquences d'images CADP : groupes de 5 images consécutives.
Séquences DoTA : une image sur 30, soit environ une par seconde.
Les embeddings sont normalisés L2 et enregistrés en ``.pt`` (un fichier par vidéo).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import CLIP_DURATION_S, DEFAULT_FPS, FRAMES_PER_CLIP


@torch.no_grad()
def _encode(model, images: list, device: str):
    return model.encode_image(torch.stack(images).to(device))


def video_features(video_path: Path, model, preprocess, device: str, embed_dim: int):
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    per_clip = int(fps * CLIP_DURATION_S)
    clips = []
    for start in range(0, max(total, 0), per_clip):
        end = min(start + per_clip, total)
        if end - start < per_clip * 0.5:  # dernier segment trop court : ignoré
            break
        frames = []
        for idx in np.linspace(start, end - 1, FRAMES_PER_CLIP).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok:
                frames.append(preprocess(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
        if len(frames) == FRAMES_PER_CLIP:
            clips.append(_encode(model, frames, device).mean(dim=0))
    cap.release()
    if not clips:
        return torch.zeros(1, embed_dim, device=device)
    return F.normalize(torch.stack(clips), dim=-1)


def image_sequence_features(frame_paths: list[str], model, preprocess, device: str,
                            group: int = FRAMES_PER_CLIP, stride: int = 1):
    """Encode une séquence d'images par groupes de ``group`` images (après sous-échantillonnage ``stride``)."""
    from PIL import Image

    paths = frame_paths[::stride]
    clips = []
    for i in range(0, len(paths), group):
        imgs = []
        for p in paths[i:i + group]:
            try:
                imgs.append(preprocess(Image.open(p).convert("RGB")))
            except OSError:
                continue
        if imgs:
            clips.append(_encode(model, imgs, device).mean(dim=0))
    return F.normalize(torch.stack(clips), dim=-1) if clips else None


def extract_all(paths, annotations: dict[str, dict], model, preprocess, device: str, embed_dim: int) -> dict:
    """Extrait les embeddings manquants de toutes les vidéos annotées et renvoie le décompte par jeu."""
    from tqdm import tqdm

    out_dir = paths.video_embeddings
    out_dir.mkdir(parents=True, exist_ok=True)
    ucf_videos = {p.stem: p for p in Path(paths.ucf).rglob("*.mp4")}
    done: dict[str, int] = {}
    for video_id, ann in tqdm(annotations.items(), desc="embeddings"):
        target = out_dir / f"{Path(video_id).stem}.pt"
        ds = ann.get("dataset", "ucf_crime")
        if not target.exists():
            if ds == "ucf_crime" and Path(video_id).stem in ucf_videos:
                feats = video_features(ucf_videos[Path(video_id).stem], model, preprocess, device, embed_dim)
            elif ds == "cadp" and ann.get("frames"):
                feats = image_sequence_features(ann["frames"], model, preprocess, device)
            elif ds == "dota" and ann.get("frames"):
                feats = image_sequence_features(ann["frames"], model, preprocess, device, group=1, stride=DEFAULT_FPS)
            else:
                continue
            if feats is None:
                continue
            torch.save(feats.cpu(), target)
        done[ds] = done.get(ds, 0) + 1
    return done
