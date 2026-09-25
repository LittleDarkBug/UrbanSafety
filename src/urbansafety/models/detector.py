"""Phase 4 : détecteur temporel d'anomalies et entraînement adversarial."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnomalyDetector(nn.Module):
    """LSTM bidirectionnel qui produit un score d'anomalie dans [0, 1] pour chaque segment.

    Entrée : [lot, segments, dim] ; sortie : [lot, segments, 1].
    """

    def __init__(self, input_dim: int = 768, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return torch.sigmoid(self.fc(self.dropout(out)))


def ranking_hinge_loss(scores, bag_labels, margin: float = 0.1):
    """Perte de classement en Multiple Instance Learning.

    Chaque vidéo est un sac de segments représenté par son score maximal. Tout sac anormal doit dépasser
    tout sac normal du lot d'au moins ``margin`` : max(0, m - max(s_anormal) + max(s_normal)).
    Aucun horodatage n'est nécessaire.
    """
    top = scores.max(dim=1).values.view(-1)
    normal, abnormal = top[bag_labels == 0], top[bag_labels == 1]
    if len(normal) == 0 or len(abnormal) == 0:
        return top.sum() * 0.0  # lot à une seule classe : perte nulle mais rattachée au graphe
    return F.relu(margin - abnormal.unsqueeze(1) + normal.unsqueeze(0)).mean()


class PGDAttack:
    """Projected Gradient Descent sur les embeddings : perturbation bornée par ``epsilon`` en norme infinie,
    construite en ``steps`` pas de taille ``alpha`` dans le sens qui dégrade le plus la prédiction."""

    def __init__(self, model: nn.Module, epsilon: float = 0.005, steps: int = 3, alpha: float = 0.002):
        self.model, self.epsilon, self.steps, self.alpha = model, epsilon, steps, alpha

    def perturb(self, inputs, segment_labels):
        delta = torch.empty_like(inputs).uniform_(-self.epsilon, self.epsilon).requires_grad_(True)
        for _ in range(self.steps):
            scores = self.model(inputs + delta).view(-1)
            loss = F.binary_cross_entropy(scores, segment_labels.reshape(-1))
            # Gradient par rapport à la seule perturbation : les poids du modèle n'accumulent rien ici.
            (grad,) = torch.autograd.grad(loss, delta)
            with torch.no_grad():
                delta += self.alpha * grad.sign()
                delta.clamp_(-self.epsilon, self.epsilon)
        return delta.detach()
