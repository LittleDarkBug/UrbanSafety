"""Jeu de séquences d'embeddings vidéo (un fichier ``.pt`` par vidéo, forme [segments, dim]) pour le LSTM."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .annotations import is_anomaly


class FeatureSequenceDataset(Dataset):
    """Renvoie ``(features, label)``.

    Avec ``clip_labels``, le label est un vecteur par segment (ajusté à la longueur de la séquence) ;
    sinon c'est le label de la vidéo entière (sac d'instances en MIL).
    """

    def __init__(self, features_dir: Path, annotations: dict[str, dict], split: str | None = None,
                 datasets: tuple[str, ...] | None = None, clip_labels: dict | None = None):
        self.annotations = annotations
        self.clip_labels = clip_labels
        keep = {
            vid for vid, ann in annotations.items()
            if (split is None or ann.get("split", "").lower() == split)
            and (datasets is None or ann.get("dataset", "ucf_crime") in datasets)
        }
        self.files = sorted(f for f in Path(features_dir).glob("*.pt") if f.stem in keep)

    def __len__(self):
        return len(self.files)

    def counts(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for f in self.files:
            ann = self.annotations.get(f.stem, {})
            c = out.setdefault(ann.get("dataset", "ucf_crime"), {"normal": 0, "anomaly": 0})
            c["anomaly" if is_anomaly(f.stem, ann) else "normal"] += 1
        return out

    def __getitem__(self, idx):
        f = self.files[idx]
        feats = torch.load(f, map_location="cpu", weights_only=True)
        bag = float(is_anomaly(f.stem, self.annotations.get(f.stem, {})))
        if self.clip_labels is None:
            return feats, torch.tensor(bag)
        y = self.clip_labels.get(f.stem)
        if y is None:
            return feats, torch.full((feats.shape[0],), bag)
        t = feats.shape[0]
        y = y[:t] if len(y) >= t else F.pad(y, (0, t - len(y)), value=0.0)
        return feats, y


def collate_padded(batch):
    """Complète les séquences par des zéros ; les labels par segment sont complétés de la même façon."""
    feats, labels = zip(*batch)
    x = pad_sequence(feats, batch_first=True)
    y = pad_sequence(labels, batch_first=True) if labels[0].dim() else torch.stack(labels)
    return x, y
