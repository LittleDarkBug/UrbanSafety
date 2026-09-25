"""Phase 5 : métriques au niveau vidéo (score d'une vidéo = maximum de ses segments) et figures."""
from __future__ import annotations

import numpy as np
import torch

from .config import DECISION_THRESHOLD


@torch.no_grad()
def predict(model, loader, device: str = "cpu"):
    model.eval()
    scores, labels = [], []
    for x, y in loader:
        scores.append(model(x.to(device)).max(dim=1).values.view(-1).cpu())
        labels.append((y if y.dim() == 1 else y.max(dim=1).values).view(-1).cpu())
    return torch.cat(scores).numpy(), torch.cat(labels).numpy()


def metrics(scores: np.ndarray, labels: np.ndarray, threshold: float = DECISION_THRESHOLD) -> dict:
    from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_score,
                                 recall_score, roc_auc_score)

    two_classes = len(np.unique(labels)) > 1
    preds = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(labels, scores)) if two_classes else 0.0,
        "ap": float(average_precision_score(labels, scores)) if two_classes else 0.0,
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "threshold": threshold,
    }


def evaluate(model, loader, device: str = "cpu", threshold: float = DECISION_THRESHOLD) -> dict:
    scores, labels = predict(model, loader, device)
    return {**metrics(scores, labels, threshold), "scores": scores, "labels": labels}


def plot_report(history_by_model: dict, scores: np.ndarray, labels: np.ndarray, out_path,
                threshold: float = DECISION_THRESHOLD) -> None:
    """Courbes d'entraînement, courbe ROC et distribution des scores par classe, dans une seule figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for name, h in history_by_model.items():
        ax[0].plot(h["epoch"], h["val_auc"], marker="o", label=name)
    ax[0].set(title="AUC de validation", xlabel="epoch", ylabel="AUC")
    ax[0].legend()
    fpr, tpr, _ = roc_curve(labels, scores)
    ax[1].plot(fpr, tpr, label=f"AUC = {roc_auc_score(labels, scores):.4f}")
    ax[1].plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax[1].set(title="Courbe ROC (test)", xlabel="taux de faux positifs", ylabel="taux de vrais positifs")
    ax[1].legend(loc="lower right")
    bins = np.linspace(0, 1, 30)
    ax[2].hist(scores[labels == 0], bins=bins, alpha=0.6, label="normal")
    ax[2].hist(scores[labels == 1], bins=bins, alpha=0.6, label="anomalie")
    ax[2].axvline(threshold, color="k", linestyle="--", label=f"seuil {threshold}")
    ax[2].set(title="Distribution des scores (test)", xlabel="score d'anomalie")
    ax[2].legend()
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
