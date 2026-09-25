"""Métadonnées BDD100K : extraction des attributs utiles et mise en forme pour le modèle de langage,
puis jeu de paires image-caption pour l'adaptation contrastive d'OpenCLIP."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

OBJECT_CATEGORIES = ("car", "truck", "bus", "train", "motorcycle", "bicycle", "pedestrian", "rider")


def parse_metadata(image: dict) -> dict:
    attrs = image.get("attributes", {})
    meta = {
        "weather": attrs.get("weather", "unknown"),
        "scene": attrs.get("scene", "unknown"),
        "timeofday": attrs.get("timeofday", "unknown"),
        "objects": {},
        "traffic_lights": [],
        "lanes": [],
        "drivable_areas": [],
    }
    for label in image.get("labels", []):
        cat = label.get("category", "")
        la = label.get("attributes", {})
        if cat in OBJECT_CATEGORIES:
            meta["objects"][cat] = meta["objects"].get(cat, 0) + 1
        elif cat == "traffic light":
            meta["traffic_lights"].append(la.get("trafficLightColor", "unknown"))
        elif cat == "lane":
            meta["lanes"].append({"type": la.get("laneType", "unknown"), "style": la.get("laneStyle", "unknown")})
        elif cat == "drivable area":
            area = la.get("areaType", "unknown")
            if area not in meta["drivable_areas"]:
                meta["drivable_areas"].append(area)
    return meta


def format_for_slm(meta: dict) -> str:
    lines = [f"Weather: {meta['weather']}", f"Scene: {meta['scene']}", f"Time: {meta['timeofday']}"]
    if meta["objects"]:
        lines.append("Objects: " + ", ".join(f"{n} {o}{'s' if n > 1 else ''}" for o, n in meta["objects"].items()))
    else:
        lines.append("Objects: none visible")
    if meta["traffic_lights"]:
        lines.append("Traffic lights: " + ", ".join(f"{n} {c}" for c, n in Counter(meta["traffic_lights"]).items()))
    lane_types = sorted({l["type"] for l in meta["lanes"] if l["type"] != "unknown"})
    if lane_types:
        lines.append("Lane markings: " + ", ".join(lane_types))
    if meta["drivable_areas"]:
        lines.append("Drivable areas: " + ", ".join(meta["drivable_areas"]))
    return "\n".join(lines)


def fallback_caption(image: dict) -> str:
    a = image.get("attributes", {})
    return f"A {a.get('weather', 'clear')} {a.get('timeofday', 'daytime')} scene in {a.get('scene', 'city street')}."


class CaptionDataset:
    """Paires (image transformée, caption) à partir des images BDD100K et des captions générées."""

    def __init__(self, images_dir: Path, captions_file: Path, transform=None):
        self.images_dir = Path(images_dir)
        self.transform = transform
        captions = json.loads(Path(captions_file).read_text(encoding="utf-8"))
        self.items = [(name, text) for name, text in captions.items() if (self.images_dir / name).exists()]
        if not self.items:
            raise RuntimeError(f"aucune image de {captions_file} trouvée dans {self.images_dir}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        from PIL import Image

        name, caption = self.items[idx]
        image = Image.open(self.images_dir / name).convert("RGB")
        return (self.transform(image) if self.transform else image), caption
