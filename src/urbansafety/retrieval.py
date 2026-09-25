"""Phase 6 : inférence. Recherche texte vers vidéo, description d'une vidéo par les phrases les plus proches,
et analyse temporelle d'une vidéo par le détecteur."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .embeddings import video_features
from .indexing import _as_row


@dataclass
class SurveillanceSystem:
    model: object          # OpenCLIP (éventuellement adapté par LoRA)
    preprocess: object
    tokenizer: object
    detector: torch.nn.Module
    video_index: object
    video_paths: list
    text_index: object
    text_database: list
    device: str = "cpu"

    @property
    def embed_dim(self) -> int:
        return self.model.visual.output_dim

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        emb = self.model.encode_text(self.tokenizer([text]).to(self.device))
        return F.normalize(emb, dim=-1).float().cpu().numpy().astype("float32")

    def search_videos(self, query: str, k: int = 5) -> list[dict]:
        scores, ids = self.video_index.search(self.encode_text(query), k)
        return [{"path": self.video_paths[i], "score": float(s)}
                for i, s in zip(ids[0], scores[0]) if 0 <= i < len(self.video_paths)]

    def search_sentences(self, vector: np.ndarray, k: int = 5) -> list[dict]:
        scores, ids = self.text_index.search(_as_row(vector), k)
        return [{**self.text_database[i], "score": float(s)}
                for i, s in zip(ids[0], scores[0]) if 0 <= i < len(self.text_database)]

    def features(self, video_path: Path):
        return video_features(Path(video_path), self.model, self.preprocess, self.device, self.embed_dim).cpu()

    def describe(self, video_path: Path, n: int = 3) -> list[dict]:
        """Phrases d'annotation les plus proches de la vidéo, une seule par vidéo source."""
        feats = self.features(video_path)
        seen, results = set(), []
        for hit in self.search_sentences(feats.mean(dim=0).numpy(), k=n * 3):
            if hit["video"] in seen:
                continue
            seen.add(hit["video"])
            results.append(hit)
            if len(results) == n:
                break
        return results

    @torch.no_grad()
    def analyze(self, features, threshold: float = 0.5) -> dict:
        self.detector.eval()
        scores = self.detector(features.unsqueeze(0).to(self.device)).view(-1).cpu().numpy()
        report = {"scores": scores, "max": float(scores.max()), "mean": float(scores.mean()),
                  "anomaly": bool(scores.max() > threshold), "peak_segment": int(scores.argmax()), "description": None}
        if report["anomaly"]:
            hits = self.search_sentences(features.mean(dim=0).numpy(), k=1)
            report["description"] = hits[0]["text"] if hits else None
        return report

    def recall_at_k(self, queries: list[str], relevant: list[set[int]], ks=(1, 5, 10)) -> dict[int, float]:
        found = {k: [] for k in ks}
        for query, gold in zip(queries, relevant):
            _, ids = self.video_index.search(self.encode_text(query), max(ks))
            for k in ks:
                found[k].append(len(set(ids[0][:k]) & gold) / len(gold) if gold else 0.0)
        return {k: float(np.mean(v)) for k, v in found.items()}
