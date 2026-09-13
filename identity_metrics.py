"""
identity_metrics.py
------------------------------------------------------------------------
Shared CLIP-I / CLIP-T / DINO embedding + similarity utilities.

Used by:
- evaluate_consistency.py
- storyboard_generator.py
- run_ablation.py

Keeping all metric logic here ensures that the same embedding and
similarity implementation is used for both runtime self-correction
and final reported evaluation results.
"""

import torch
from PIL import Image
from transformers import (
    CLIPModel,
    CLIPProcessor,
    AutoImageProcessor,
    AutoModel,
)


# ---------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------
# Lazy-loaded models
# ---------------------------------------------------------------------

_clip_model = None
_clip_processor = None

_dino_model = None
_dino_processor = None


# ---------------------------------------------------------------------
# CLIP
# ---------------------------------------------------------------------

def get_clip():
    """Load and cache the CLIP model and processor."""
    global _clip_model, _clip_processor

    if _clip_model is None:
        _clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).to(DEVICE)

        _clip_model.eval()

        _clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

    return _clip_model, _clip_processor


# ---------------------------------------------------------------------
# DINO
# ---------------------------------------------------------------------

def get_dino():
    """Load and cache the DINOv2 model and processor."""
    global _dino_model, _dino_processor

    if _dino_model is None:
        _dino_processor = AutoImageProcessor.from_pretrained(
            "facebook/dinov2-base"
        )

        _dino_model = AutoModel.from_pretrained(
            "facebook/dinov2-base"
        ).to(DEVICE)

        _dino_model.eval()

    return _dino_model, _dino_processor


# ---------------------------------------------------------------------
# CLIP image embedding
# ---------------------------------------------------------------------

@torch.no_grad()
def clip_embedding(image: Image.Image) -> torch.Tensor:
    """
    Return normalized CLIP image embedding.

    Args:
        image: PIL image.

    Returns:
        Tensor of shape [1, embedding_dim].
    """
    model, processor = get_clip()

    image = image.convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    outputs = model.get_image_features(**inputs)

    # Transformers versions differ in what get_image_features returns.
    if torch.is_tensor(outputs):
        features = outputs

    elif hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
        features = outputs.image_embeds

    elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        features = outputs.pooler_output

    elif hasattr(outputs, "last_hidden_state"):
        features = outputs.last_hidden_state[:, 0, :]

    else:
        raise TypeError(
            f"Unexpected CLIP image output type: {type(outputs)}"
        )

    # Normalize for cosine similarity.
    features = torch.nn.functional.normalize(features, p=2, dim=-1)

    return features


# ---------------------------------------------------------------------
# DINO embedding
# ---------------------------------------------------------------------

@torch.no_grad()
def dino_embedding(image: Image.Image) -> torch.Tensor:
    """
    Return normalized DINOv2 CLS embedding.

    Args:
        image: PIL image.

    Returns:
        Tensor of shape [1, embedding_dim].
    """
    model, processor = get_dino()

    image = image.convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    outputs = model(**inputs)

    # DINOv2 CLS token.
    features = outputs.last_hidden_state[:, 0, :]

    # Normalize for cosine similarity.
    features = torch.nn.functional.normalize(features, p=2, dim=-1)

    return features


# ---------------------------------------------------------------------
# CLIP text embedding
# ---------------------------------------------------------------------

@torch.no_grad()
def clip_text_embedding(text: str) -> torch.Tensor:
    """
    Return normalized CLIP text embedding.

    Handles both older and newer Transformers return types.
    """
    if not isinstance(text, str):
        raise TypeError(
            f"text must be a string, got {type(text)}"
        )

    model, processor = get_clip()

    inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    outputs = model.get_text_features(**inputs)

    # Newer/older Transformers versions can return different objects.
    if torch.is_tensor(outputs):
        features = outputs

    elif hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
        features = outputs.text_embeds

    elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        features = outputs.pooler_output

    elif hasattr(outputs, "last_hidden_state"):
        features = outputs.last_hidden_state[:, 0, :]

    else:
        raise TypeError(
            f"Unexpected CLIP text output type: {type(outputs)}"
        )

    # Normalize for cosine similarity.
    features = torch.nn.functional.normalize(features, p=2, dim=-1)

    return features


# ---------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Compute cosine similarity between two embeddings.

    Returns:
        Float similarity score in approximately [-1, 1].
    """
    if not torch.is_tensor(a):
        raise TypeError(
            f"Expected first argument to be Tensor, got {type(a)}"
        )

    if not torch.is_tensor(b):
        raise TypeError(
            f"Expected second argument to be Tensor, got {type(b)}"
        )

    # Make sure embeddings have compatible shapes.
    if a.ndim == 1:
        a = a.unsqueeze(0)

    if b.ndim == 1:
        b = b.unsqueeze(0)

    if a.shape != b.shape:
        raise ValueError(
            f"Embedding shape mismatch: {a.shape} vs {b.shape}"
        )

    return torch.nn.functional.cosine_similarity(
        a,
        b,
        dim=-1
    ).mean().item()


# ---------------------------------------------------------------------
# CLIP-I + DINO
# ---------------------------------------------------------------------

def identity_scores(
    reference: Image.Image,
    candidate: Image.Image
) -> dict:
    """
    Compute image-to-image identity/visual consistency metrics.

    Returns:
        {
            "clip_i": float,
            "dino": float
        }
    """
    if not isinstance(reference, Image.Image):
        raise TypeError(
            f"reference must be PIL.Image.Image, got {type(reference)}"
        )

    if not isinstance(candidate, Image.Image):
        raise TypeError(
            f"candidate must be PIL.Image.Image, got {type(candidate)}"
        )

    clip_score = cosine_sim(
        clip_embedding(reference),
        clip_embedding(candidate)
    )

    dino_score = cosine_sim(
        dino_embedding(reference),
        dino_embedding(candidate)
    )

    return {
        "clip_i": clip_score,
        "dino": dino_score
    }


# ---------------------------------------------------------------------
# CLIP-T
# ---------------------------------------------------------------------

def clip_t_score(
    prompt: str,
    candidate: Image.Image
) -> float:
    """
    Compute CLIP text-image alignment.

    Args:
        prompt: Scene description/prompt.
        candidate: Generated scene image.

    Returns:
        CLIP-T cosine similarity score.
    """
    if not isinstance(prompt, str):
        raise TypeError(
            f"prompt must be a string, got {type(prompt)}"
        )

    if not isinstance(candidate, Image.Image):
        raise TypeError(
            f"candidate must be PIL.Image.Image, got {type(candidate)}"
        )

    text_features = clip_text_embedding(prompt)
    image_features = clip_embedding(candidate)

    return cosine_sim(
        text_features,
        image_features
    )


# ---------------------------------------------------------------------
# Optional convenience function
# ---------------------------------------------------------------------

def all_metrics(
    reference: Image.Image,
    candidate: Image.Image,
    prompt: str
) -> dict:
    """
    Compute all current metrics together.

    Returns:
        {
            "clip_i": ...,
            "dino": ...,
            "clip_t": ...
        }
    """
    identity = identity_scores(
        reference,
        candidate
    )

    clip_t = clip_t_score(
        prompt,
        candidate
    )

    return {
        **identity,
        "clip_t": clip_t
    }