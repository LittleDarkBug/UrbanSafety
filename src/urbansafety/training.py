"""Phase 4 : entraînement du détecteur, en version de base ou sous attaque PGD, avec arrêt anticipé sur l'AUC de validation."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch

from .evaluation import evaluate
from .models.detector import AnomalyDetector, PGDAttack, ranking_hinge_loss


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-3
    accumulation_steps: int = 4
    patience: int = 5
    adversarial: bool = False
    pgd_epsilon: float = 0.005
    pgd_steps: int = 3
    pgd_alpha: float = 0.002


@dataclass
class TrainResult:
    model: AnomalyDetector
    best_val_auc: float
    history: dict = field(default_factory=lambda: {"epoch": [], "loss": [], "val_auc": []})


def train_detector(train_loader, val_loader, embed_dim: int, cfg: TrainConfig, device: str = "cpu") -> TrainResult:
    from tqdm import tqdm

    model = AnomalyDetector(input_dim=embed_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    attack = PGDAttack(model, cfg.pgd_epsilon, cfg.pgd_steps, cfg.pgd_alpha) if cfg.adversarial else None
    result = TrainResult(model=model, best_val_auc=0.0)
    best_state, stale = copy.deepcopy(model.state_dict()), 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = 0.0
        optimizer.zero_grad()
        for step, (x, y) in enumerate(tqdm(train_loader, desc=f"epoch {epoch}/{cfg.epochs}"), start=1):
            x, y = x.to(device), y.to(device)
            bags = y if y.dim() == 1 else y.max(dim=1).values
            if attack is not None:
                x = x + attack.perturb(x, bags.unsqueeze(1).expand(-1, x.size(1)))
            loss = ranking_hinge_loss(model(x), bags)
            (loss / cfg.accumulation_steps).backward()
            if step % cfg.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            total += loss.item()
        scheduler.step()

        val_auc = evaluate(model, val_loader, device)["auc"]
        result.history["epoch"].append(epoch)
        result.history["loss"].append(total / max(1, len(train_loader)))
        result.history["val_auc"].append(val_auc)
        print(f"epoch {epoch} : perte {result.history['loss'][-1]:.4f}, AUC validation {val_auc:.4f}")

        if val_auc > result.best_val_auc:
            result.best_val_auc, best_state, stale = val_auc, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= cfg.patience:
                print(f"arrêt anticipé à l'epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return result
