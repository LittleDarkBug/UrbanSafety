"""Chemins, constantes et reproductibilité.

Deux dispositions de fichiers sont prévues : ``local`` (dossiers relatifs au répertoire de travail) et
``kaggle`` (jeux montés sous ``/kaggle/input``).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SEED = 42

# Segmentation temporelle : un segment = 5 secondes, encodé par 5 images moyennées.
CLIP_DURATION_S = 5
FRAMES_PER_CLIP = 5
DEFAULT_FPS = 30

CLIP_MODEL = "ViT-L-14"
CLIP_PRETRAINED = "laion400m_e32"
SLM_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# Seuls les jeux à structure temporelle vidéo servent à entraîner le LSTM.
# DoTA et CADP (images ou images extraites) restent utilisés pour la recherche texte vers vidéo.
LSTM_DATASETS = ("ucf_crime",)

DECISION_THRESHOLD = 0.4


@dataclass
class Paths:
    bdd100k: Path
    captions: Path
    ucf: Path
    dota: Path
    cadp: Path
    work: Path
    enabled: dict = field(default_factory=dict)

    @classmethod
    def for_mode(cls, mode: str = "local", work: str | Path | None = None) -> "Paths":
        if mode == "kaggle":
            p = cls(
                bdd100k=Path("/kaggle/input/bdd100k"),
                captions=Path("/kaggle/input/captions"),
                ucf=Path("/kaggle/input/ucaucf-crime-annotation-dataset"),
                dota=Path("/kaggle/input/dota-dataset"),
                cadp=Path("/kaggle/input/accident-detection-from-cctv-footage"),
                work=Path(work or "/kaggle/working"),
            )
        elif mode == "local":
            p = cls(
                bdd100k=Path("bdd100k"),
                captions=Path("captions"),
                ucf=Path("ucaucf-crime-annotation-dataset"),
                dota=Path("dota-dataset"),
                cadp=Path("cadp-dataset"),
                work=Path(work or "."),
            )
        else:
            raise ValueError(f"mode inconnu : {mode!r} (local ou kaggle)")
        p.refresh_enabled()
        return p

    def refresh_enabled(self) -> dict:
        """Un jeu est actif si ses fichiers sont présents sur le disque."""
        dota_frames = self.dota / "frames"
        self.enabled = {
            "ucf_crime": self.ucf.exists(),
            "dota": (self.dota / "metadata_train.json").exists()
                    and dota_frames.exists() and sum(1 for _ in dota_frames.iterdir()) > 100,
            "cadp": (self.cadp / "train" / "Accident").exists(),
        }
        return self.enabled

    # Artefacts produits par le pipeline
    @property
    def video_embeddings(self) -> Path:
        return self.work / "video_embeddings"

    @property
    def text_embeddings(self) -> Path:
        return self.work / "text_embeddings"

    @property
    def video_index(self) -> Path:
        return self.work / "ucf_video_index.faiss"

    @property
    def video_paths(self) -> Path:
        return self.work / "video_paths.pkl"

    @property
    def text_index(self) -> Path:
        return self.work / "index_text_full.faiss"

    @property
    def text_database(self) -> Path:
        return self.work / "text_database.pkl"

    @property
    def clip_weights(self) -> Path:
        return self.work / "vlm_phase0_finetuned.pth"

    @property
    def detector_weights(self) -> Path:
        return self.work / "detector_best.pth"

    @property
    def captions_train(self) -> Path:
        return self.captions / "urban_captions_train.json"

    @property
    def captions_val(self) -> Path:
        return self.captions / "urban_captions_val.json"


def device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int = SEED) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
