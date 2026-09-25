"""Phase 2 : OpenCLIP ViT-L/14 adapté par LoRA sur les paires image-caption BDD100K."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from ..config import CLIP_MODEL, CLIP_PRETRAINED

LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ("c_fc", "c_proj")  # couches MLP des blocs du transformer


def load_clip(model_name: str = CLIP_MODEL, pretrained: str = CLIP_PRETRAINED, device: str = "cpu"):
    """Renvoie ``(model, preprocess, tokenizer)``."""
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    return model.to(device), preprocess, open_clip.get_tokenizer(model_name)


def apply_lora(model):
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, target_modules=list(LORA_TARGETS),
                        lora_dropout=LORA_DROPOUT, bias="none")
    return get_peft_model(model, config)


def clip_contrastive_loss(image_features, text_features, logit_scale):
    """InfoNCE symétrique : chaque image doit retrouver sa caption dans le lot, et inversement."""
    logits = logit_scale * image_features @ text_features.t()
    target = torch.arange(len(logits), device=logits.device)
    return (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target)) / 2


def finetune(model, tokenizer, loader, epochs: int = 12, lr: float = 5e-5, device: str = "cuda"):
    """Entraîne les adaptateurs LoRA en précision mixte bfloat16 sur GPU."""
    from tqdm import tqdm

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    use_amp = device.startswith("cuda")
    history = []
    model.train()
    for epoch in range(epochs):
        for images, texts in (bar := tqdm(loader, desc=f"LoRA epoch {epoch + 1}/{epochs}")):
            images = images.to(device, non_blocking=True)
            tokens = tokenizer(list(texts)).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda" if use_amp else "cpu", dtype=torch.bfloat16, enabled=use_amp):
                img_f, txt_f, scale = model(images, tokens)[:3]
                loss = clip_contrastive_loss(img_f, txt_f, scale)
            loss.backward()
            optimizer.step()
            bar.set_postfix(loss=f"{loss.item():.4f}")
        history.append(loss.item())
    model.eval()
    return history


def load_finetuned(model, weights: Path, device: str = "cpu") -> bool:
    """Charge les poids LoRA s'ils existent ; sinon le modèle reste sur ses poids pré-entraînés."""
    weights = Path(weights)
    if not weights.exists():
        return False
    model.load_state_dict(torch.load(weights, map_location=device))
    return True


def freeze(model):
    for p in model.parameters():
        p.requires_grad = False
    return model.eval()
