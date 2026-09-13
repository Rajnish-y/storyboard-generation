"""
identity_metrics.py
------------------------------------------------------------------------
Shared CLIP-I / DINO embedding + similarity utilities. Used by both
evaluate_consistency.py (offline reporting) and storyboard_generator.py's
self-correcting regeneration loop (online, during generation).

Kept as its own module so both consumers use identical embedding logic —
important for the self-correction threshold to mean the same thing as the
numbers you report in your results table.
"""

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, AutoImageProcessor, AutoModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_clip_model = None
_clip_processor = None
_dino_model = None
_dino_processor = None


def get_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return _clip_model, _clip_processor


def get_dino():
    global _dino_model, _dino_processor
    if _dino_model is None:
        _dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        _dino_model = AutoModel.from_pretrained("facebook/dinov2-base").to(DEVICE)
    return _dino_model, _dino_processor


@torch.no_grad()
def clip_embedding(image: Image.Image):
    model, processor = get_clip()
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    feats = model.get_image_features(**inputs)
    if not torch.is_tensor(feats):
        feats = getattr(feats, "image_embeds", None) or feats.pooler_output
    return feats


@torch.no_grad()
def dino_embedding(image: Image.Image):
    model, processor = get_dino()
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]  # CLS token


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a, b).item()


def identity_scores(reference: Image.Image, candidate: Image.Image) -> dict:
    """Returns {'clip_i': float, 'dino': float} between two images."""
    clip_score = cosine_sim(clip_embedding(reference), clip_embedding(candidate))
    dino_score = cosine_sim(dino_embedding(reference), dino_embedding(candidate))
    return {"clip_i": clip_score, "dino": dino_score}


@torch.no_grad()
def clip_text_embedding(text: str):
    model, processor = get_clip()
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    feats = model.get_text_features(**inputs)
    return feats


def clip_t_score(prompt: str, candidate: Image.Image) -> float:
    """CLIP-T: text-image alignment. Measures scene/prompt compliance —
    the counterpart to CLIP-I/DINO, which only measure identity."""
    return cosine_sim(clip_text_embedding(prompt), clip_embedding(candidate))