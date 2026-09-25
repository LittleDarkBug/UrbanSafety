"""Téléchargement automatique de CADP (kagglehub) et de DoTA (annotations GitHub, images Google Drive).

UCF-Crime (annotations UCA) et BDD100K sont à placer manuellement.
"""
from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

DOTA_ANNOTATIONS_URL = "https://raw.githubusercontent.com/MoonBlvd/Detection-of-Traffic-Anomaly/master/dataset/"
DOTA_ANNOTATION_FILES = ("metadata_train.json", "metadata_val.json", "train_split.txt", "val_split.txt")
DOTA_FRAMES_DRIVE_FOLDER = "1_WzhwZC2NIpzZIpX7YCvapq66rtBc67n"
CADP_KAGGLE_DATASET = "ckay16/accident-detection-from-cctv-footage"


def download_cadp(output_dir: Path) -> bool:
    output_dir = Path(output_dir)
    if (output_dir / "train" / "Accident").exists():
        print("CADP : déjà présent")
        return True
    import kagglehub

    print("CADP : téléchargement via kagglehub (~262 Mo)")
    src = Path(kagglehub.dataset_download(CADP_KAGGLE_DATASET))
    data_dir = src / "data" if (src / "data").exists() else src
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test", "val"):
        if (data_dir / split).exists() and not (output_dir / split).exists():
            shutil.copytree(data_dir / split, output_dir / split)
    return (output_dir / "train" / "Accident").exists()


def download_dota_annotations(output_dir: Path) -> bool:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in DOTA_ANNOTATION_FILES:
        dest = output_dir / name
        if not dest.exists():
            try:
                urllib.request.urlretrieve(DOTA_ANNOTATIONS_URL + name, dest)
            except OSError as e:
                print(f"DoTA : échec du téléchargement de {name} ({e})")
    return (output_dir / "metadata_train.json").exists()


def download_dota_frames(output_dir: Path) -> bool:
    output_dir = Path(output_dir)
    frames = output_dir / "frames"
    if frames.exists() and sum(1 for _ in frames.iterdir()) > 100:
        print("DoTA : images déjà présentes")
        return True
    import gdown

    print("DoTA : téléchargement des images (~55 Go)")
    output_dir.mkdir(parents=True, exist_ok=True)
    gdown.download_folder(id=DOTA_FRAMES_DRIVE_FOLDER, output=str(output_dir), quiet=False)
    for archive in sorted(output_dir.glob("*.zip")):
        with zipfile.ZipFile(archive) as z:
            z.extractall(output_dir)
        archive.unlink()
    return frames.exists() and any(frames.iterdir())


def download_all(paths) -> dict:
    """Télécharge ce qui manque et renvoie l'état des jeux."""
    download_cadp(paths.cadp)
    download_dota_annotations(paths.dota)
    download_dota_frames(paths.dota)
    return paths.refresh_enabled()
