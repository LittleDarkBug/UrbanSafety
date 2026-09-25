"""Chargement et harmonisation des annotations UCF-Crime (UCA), DoTA et CADP, puis pseudo-labels par segment.

Chaque vidéo devient un dictionnaire commun : ``duration`` (s), ``timestamps`` ([[début, fin]] en s),
``sentences``, ``dataset``, ``split`` et, pour les jeux d'images, ``frames``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..config import CLIP_DURATION_S, DEFAULT_FPS

DOTA_CLASSES = {
    "ST": "vehicle suddenly stopping",
    "AH": "vehicle ahead collision",
    "LA": "lateral collision",
    "OO": "oncoming obstacle collision",
    "VO": "vehicle-obstacle collision",
    "TC": "lane change accident",
    "OC": "oncoming vehicle collision",
    "UK": "unknown traffic anomaly",
}
# CADP fournit des images nommées « préfixe (n).jpg » : on regroupe par préfixe pour reconstituer des séquences.
_CADP_FRAME = re.compile(r"^(.+?)\s*\((\d+)\)$")


def _frame_files(directory: Path) -> list[Path]:
    d = directory / "images" if (directory / "images").exists() else directory
    return sorted(d.glob("*.jpg")) + sorted(d.glob("*.png"))


def is_anomaly(video_name: str, ann: dict) -> bool:
    dataset = ann.get("dataset", "ucf_crime")
    if dataset == "dota":
        return True  # DoTA ne contient que des anomalies
    if dataset == "cadp":
        return bool(ann.get("is_anomaly", False))
    return "normal" not in video_name.lower()


def load_ucf_crime(ucf_path: Path) -> dict[str, dict]:
    out = {}
    for split in ("Train", "Val", "Test"):
        f = Path(ucf_path) / f"UCFCrime_{split}.json"
        if not f.exists():
            continue
        for video_id, ann in json.loads(f.read_text()).items():
            out[video_id] = {**ann, "dataset": "ucf_crime", "split": split.lower(), "video_id": video_id}
    return out


def load_dota(dota_path: Path, fps: int = DEFAULT_FPS) -> dict[str, dict]:
    dota_path = Path(dota_path)
    out = {}
    for split in ("train", "val"):
        f = dota_path / f"metadata_{split}.json"
        if not f.exists():
            continue
        for video_id, meta in json.loads(f.read_text()).items():
            n = meta.get("num_frames", 300)
            cls = meta.get("anomaly_class", "UK")
            frames_dir = dota_path / "frames" / video_id
            out[video_id] = {
                "duration": n / fps,
                "timestamps": [[meta.get("anomaly_start", 0) / fps, meta.get("anomaly_end", n) / fps]],
                "sentences": [f"Traffic surveillance showing {DOTA_CLASSES.get(cls, 'traffic anomaly')}"],
                "dataset": "dota",
                "anomaly_class": cls,
                "split": split,
                "frames": [str(p) for p in _frame_files(frames_dir)] if frames_dir.exists() else [],
                "num_frames": n,
                "video_id": video_id,
            }
    return out


def load_cadp(cadp_path: Path, fps: int = DEFAULT_FPS) -> dict[str, dict]:
    out = {}
    for split in ("train", "val", "test"):
        for category in ("Accident", "Non Accident"):
            d = Path(cadp_path) / split / category
            if not d.exists():
                continue
            groups: dict[str, list[tuple[int, Path]]] = {}
            for img in list(d.glob("*.jpg")) + list(d.glob("*.png")):
                m = _CADP_FRAME.match(img.stem)
                prefix, num = (m.group(1).strip(), int(m.group(2))) if m else (img.stem, 0)
                groups.setdefault(prefix, []).append((num, img))
            anomalous = category == "Accident"
            for prefix, frames in groups.items():
                frames.sort(key=lambda t: t[0])
                video_id = f"cadp_{split}_{category.replace(' ', '_')}_{prefix}"
                duration = len(frames) / fps
                out[video_id] = {
                    "duration": duration,
                    "timestamps": [[0, duration]] if anomalous else [],
                    "sentences": ["Traffic accident scene" if anomalous else "Normal traffic scene"],
                    "dataset": "cadp",
                    "split": split,
                    "frames": [str(p) for _, p in frames],
                    "is_anomaly": anomalous,
                    "video_id": video_id,
                }
    return out


def load_all(paths) -> dict[str, dict[str, dict]]:
    """Regroupe les jeux actifs par split. DoTA n'a pas de test : son split ``val`` va en validation."""
    splits = {"train": {}, "val": {}, "test": {}}
    sources = []
    if paths.enabled.get("ucf_crime"):
        sources.append(load_ucf_crime(paths.ucf))
    if paths.enabled.get("dota"):
        sources.append(load_dota(paths.dota))
    if paths.enabled.get("cadp"):
        sources.append(load_cadp(paths.cadp))
    for anns in sources:
        for video_id, ann in anns.items():
            split = ann.get("split", "train")
            splits[split if split in splits else "val"][video_id] = ann
    return splits


def summarize(annotations: dict[str, dict]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for video_id, ann in annotations.items():
        c = counts.setdefault(ann.get("dataset", "ucf_crime"), {"normal": 0, "anomaly": 0})
        c["anomaly" if is_anomaly(video_id, ann) else "normal"] += 1
    return counts


def clip_labels(annotations: dict[str, dict], clip_duration: float = CLIP_DURATION_S):
    """Pseudo-labels binaires par segment : 1 si le segment recoupe un intervalle anormal.

    Une vidéo anormale sans horodatage est entièrement étiquetée anormale.
    """
    import torch

    labels = {}
    for video_id, ann in annotations.items():
        n = max(1, int(ann.get("duration", 300) // clip_duration))
        y = torch.zeros(n, dtype=torch.float32)
        if is_anomaly(video_id, ann):
            stamps = ann.get("timestamps", [])
            if not stamps:
                y[:] = 1.0
            for start, end in stamps:
                y[int(start // clip_duration):min(int(end // clip_duration) + 1, n)] = 1.0
        labels[video_id] = y
    return labels
