"""Phase 1 : génération de captions BDD100K par un petit modèle de langage (Qwen2.5-Instruct).

Les métadonnées structurées de chaque image deviennent une phrase courte, qui sert de texte apparié
pour l'adaptation contrastive d'OpenCLIP. La génération reprend là où elle s'est arrêtée.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

from .config import SLM_MODEL_ID
from .data.bdd100k import fallback_caption, format_for_slm, parse_metadata

SYSTEM_PROMPT = "Generate a single concise sentence (under 25 words) describing this driving scene. No preamble."


def load_slm(model_id: str = SLM_MODEL_ID):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto",
                                                 trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer


def build_prompt(metadata_text: str, tokenizer) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{metadata_text}\nDescription:"}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_captions(model, tokenizer, images: list[dict], output_file: Path,
                      batch_size: int = 32, save_every: int = 500, max_new_tokens: int = 40) -> dict[str, str]:
    import torch
    from tqdm import tqdm

    output_file = Path(output_file)
    captions = json.loads(output_file.read_text(encoding="utf-8")) if output_file.exists() else {}
    todo = [img for img in images if img["name"] not in captions]

    def save():
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(captions, indent=2, ensure_ascii=False), encoding="utf-8")

    for start in tqdm(range(0, len(todo), batch_size), desc="captions"):
        batch = todo[start:start + batch_size]
        prompts = [build_prompt(format_for_slm(parse_metadata(img)), tokenizer) for img in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        texts = tokenizer.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        for img, text in zip(batch, texts):
            text = text.strip()
            captions[img["name"]] = (text[:200] + "...") if len(text) > 200 else (text or fallback_caption(img))
        if (start // batch_size + 1) % max(1, save_every // batch_size) == 0:
            save()
    save()
    return captions


def release(model, tokenizer) -> None:
    import torch

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
