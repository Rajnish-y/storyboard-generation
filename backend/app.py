"""FastAPI bridge for SDXL + IP-Adapter storyboard generation.

Run from the repository root with:
    uvicorn backend.app:app --host 0.0.0.0 --port 8000

The generator is initialized once at startup. GPU-heavy calls are serialized.
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from diffusers.image_processor import IPAdapterMaskProcessor
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from storyboard_generator import (
    Scene,
    StoryboardGenerator,
    classify_scene_dynamics,
    compute_adaptive_scale,
)
from identity_metrics import identity_scores, clip_t_score

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(os.getenv("STORYBOARD_OUTPUT_DIR", str(ROOT_DIR / "generated_outputs")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, deformed, extra limbs, low quality, portrait, close-up, "
    "plain background, studio background, illustration, comic, cartoon, sketch"
)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("storyboard-api")


def _cors_origins() -> list[str]:
    raw = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173,https://storyframe-eta.vercel.app",
    )
    return [item.strip() for item in raw.split(",") if item.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.generator = None
    app.state.model_error = None
    app.state.gpu_lock = asyncio.Lock()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    allow_cpu = os.getenv("ALLOW_CPU_INFERENCE", "0").lower() in {"1", "true", "yes"}
    if device == "cpu" and not allow_cpu:
        app.state.model_error = (
            "No CUDA GPU is available. Run the generation backend in a Colab T4/A100 "
            "runtime or explicitly enable ALLOW_CPU_INFERENCE=1 (CPU generation is very slow)."
        )
        logger.error(app.state.model_error)
    else:
        try:
            dtype = torch.float16 if device == "cuda" else torch.float32
            logger.info("Loading StoryboardGenerator on %s; this may take several minutes.", device)
            app.state.generator = StoryboardGenerator(device=device, dtype=dtype)
            logger.info("StoryboardGenerator loaded successfully on %s.", device)
        except Exception as exc:  # keep API health available so UI can show the error
            app.state.model_error = f"Model initialization failed: {exc}"
            logger.exception("Model initialization failed")

    yield


app = FastAPI(
    title="StoryFrame API",
    description="Reference-conditioned scene generation with adaptive conditioning and DINO feedback.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-Demo-Token"],
)


def verify_demo_token(x_demo_token: str | None = Header(default=None, alias="X-Demo-Token")) -> None:
    """Require a demo token only when DEMO_API_TOKEN is configured."""
    expected = os.getenv("DEMO_API_TOKEN", "").strip()
    if expected and not secrets.compare_digest(x_demo_token or "", expected):
        raise HTTPException(status_code=401, detail="Invalid or missing demo token.")


def get_generator() -> StoryboardGenerator:
    gen = getattr(app.state, "generator", None)
    if gen is None:
        reason = getattr(app.state, "model_error", None) or "The generation model is not ready."
        raise HTTPException(status_code=503, detail=reason)
    return gen


async def load_reference(file: UploadFile) -> Image.Image:
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Reference image must be 10 MB or smaller.")
    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")
    if image.width < 64 or image.height < 64:
        raise HTTPException(status_code=400, detail="Reference image must be at least 64×64 pixels.")
    return image


def _save_image(image: Image.Image) -> tuple[str, str]:
    image_id = uuid.uuid4().hex
    filename = f"{image_id}.png"
    path = OUTPUT_DIR / filename
    image.save(path, format="PNG", optimize=True)
    return image_id, filename


def _safe_float(value: Any) -> float:
    return round(float(value), 6)


def _compose_contact_sheet(results: list[dict[str, Any]]) -> str | None:
    """Create a downloadable contact sheet from successful storyboard frames."""
    successful = [item for item in results if item.get("image_url")]
    if not successful:
        return None

    thumb_w = 480
    margin = 22
    caption_h = 58
    gap = 18
    columns = 2 if len(successful) > 1 else 1
    loaded: list[tuple[dict[str, Any], Image.Image]] = []
    for item in successful:
        image_id = str(item["image_url"]).rsplit("/", 1)[-1]
        path = OUTPUT_DIR / f"{image_id}.png"
        if not path.is_file():
            continue
        image = Image.open(path).convert("RGB")
        thumb_h = int(thumb_w * image.height / image.width)
        image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        loaded.append((item, image.copy()))
    if not loaded:
        return None

    frame_h = max(image.height for _, image in loaded) + caption_h
    rows = (len(loaded) + columns - 1) // columns
    width = margin * 2 + columns * thumb_w + (columns - 1) * gap
    height = margin * 2 + rows * frame_h + (rows - 1) * gap + 42
    sheet = Image.new("RGB", (width, height), (12, 16, 23))
    draw = ImageDraw.Draw(sheet)
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
        label_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 15)
        caption_font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except OSError:
        title_font = label_font = caption_font = ImageFont.load_default()
    draw.text((margin, 12), "STORYFRAME | GENERATED STORYBOARD", font=title_font, fill=(226, 237, 252))
    top = 48
    for index, (item, image) in enumerate(loaded):
        col = index % columns
        row = index // columns
        x = margin + col * (thumb_w + gap)
        y = top + row * (frame_h + gap)
        sheet.paste(image, (x, y))
        label_y = y + image.height + 8
        draw.text((x, label_y), f"{index + 1:02d}  {item.get('scene_id', '')}", font=label_font, fill=(151, 190, 255))
        caption = str(item.get("prompt", ""))
        if len(caption) > 90:
            caption = caption[:87] + "..."
        draw.text((x, label_y + 24), caption, font=caption_font, fill=(174, 188, 208))

    sheet_id = uuid.uuid4().hex
    png_path = OUTPUT_DIR / f"{sheet_id}.png"
    pdf_path = OUTPUT_DIR / f"{sheet_id}.pdf"
    sheet.save(png_path, format="PNG", optimize=True)
    sheet.save(pdf_path, format="PDF", resolution=144.0)
    return sheet_id


def _run_one(
    gen: StoryboardGenerator,
    reference: Image.Image,
    prompt: str,
    mode: str,
    threshold: float,
    max_retries: int,
    steps: int,
    guidance: float,
    seed: int,
    scene_id: str,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
) -> dict[str, Any]:
    """Run one method and report the selected result and trace accurately."""
    scene = Scene(
        scene_id=scene_id,
        prompt=prompt.strip(),
        negative_prompt=negative_prompt,
        ip_adapter_scale=0.5,
    )
    scene_type = classify_scene_dynamics(scene)
    started = time.perf_counter()
    attempts: list[dict[str, Any]] = []
    best_image: Image.Image | None = None
    best_identity: dict[str, float] | None = None
    best_scale = 0.5
    best_attempt = 0
    initial_scale = 0.5
    threshold_met = False

    if mode == "fixed":
        initial_scale = 0.5
        trial = Scene(scene_id, prompt, negative_prompt, initial_scale)
        candidate = gen.generate_scene(
            reference, trial, num_inference_steps=steps, guidance_scale=guidance, seed=seed
        )
        scores = identity_scores(reference, candidate)
        attempts.append({"attempt": 0, "scale": initial_scale, "clip_i": _safe_float(scores["clip_i"]), "dino": _safe_float(scores["dino"]), "threshold_met": None})
        best_image, best_identity, best_scale, best_attempt = candidate, scores, initial_scale, 0

    elif mode == "adaptive_only":
        initial_scale = compute_adaptive_scale(scene)
        trial = Scene(scene_id, prompt, negative_prompt, initial_scale)
        candidate = gen.generate_scene(
            reference, trial, num_inference_steps=steps, guidance_scale=guidance, seed=seed
        )
        scores = identity_scores(reference, candidate)
        attempts.append({"attempt": 0, "scale": initial_scale, "clip_i": _safe_float(scores["clip_i"]), "dino": _safe_float(scores["dino"]), "threshold_met": None})
        best_image, best_identity, best_scale, best_attempt = candidate, scores, initial_scale, 0

    elif mode == "proposed":
        initial_scale = compute_adaptive_scale(scene)
        current_scale = initial_scale
        for attempt_index in range(max_retries + 1):
            trial = Scene(scene_id, prompt, negative_prompt, current_scale)
            candidate = gen.generate_scene(
                reference,
                trial,
                num_inference_steps=steps,
                guidance_scale=guidance,
                seed=seed + attempt_index,
            )
            scores = identity_scores(reference, candidate)
            passed = float(scores["dino"]) >= threshold
            attempts.append({
                "attempt": attempt_index,
                "scale": round(current_scale, 2),
                "clip_i": _safe_float(scores["clip_i"]),
                "dino": _safe_float(scores["dino"]),
                "threshold_met": passed,
            })
            if best_identity is None or float(scores["dino"]) > float(best_identity["dino"]):
                best_image = candidate
                best_identity = dict(scores)
                best_scale = current_scale
                best_attempt = attempt_index
            if passed:
                threshold_met = True
                # Stop when the threshold is reached, but retain the best result so far.
                break
            current_scale = min(current_scale + 0.10, 1.0)
    else:
        raise HTTPException(status_code=400, detail="Unknown method. Choose fixed, adaptive_only, or proposed.")

    if best_image is None or best_identity is None:
        raise RuntimeError("Generation completed without producing an image.")

    final_metrics = {
        "clip_i": _safe_float(best_identity["clip_i"]),
        "dino": _safe_float(best_identity["dino"]),
        "clip_t": _safe_float(clip_t_score(prompt, best_image)),
    }
    image_id, _ = _save_image(best_image)
    elapsed = round(time.perf_counter() - started, 2)
    actual_retries = max(0, len(attempts) - 1)
    return {
        "scene_id": scene_id,
        "prompt": prompt,
        "mode": mode,
        "scene_type": scene_type,
        "image_url": f"/api/images/{image_id}",
        "metrics": {
            **final_metrics,
            "retries": actual_retries,
            "attempt_count": len(attempts),
            "best_attempt": best_attempt,
            "initial_scale": _safe_float(initial_scale),
            "selected_scale": _safe_float(best_scale),
            "last_attempt_scale": _safe_float(attempts[-1]["scale"]),
            "threshold": _safe_float(threshold),
            "threshold_met": bool(threshold_met) if mode == "proposed" else None,
            "generation_seconds": elapsed,
            "guidance_scale": guidance,
            "steps": steps,
            "seed": seed,
        },
        "attempts": attempts,
    }


MULTI_CHARACTER_NEGATIVE_PROMPT = (
    "merged faces, duplicate person, extra people, third person, crowd, group, "
    "identity blending, extra limbs, malformed hands, deformed face, blurry, low quality, text, watermark"
)


def _check_sdxl_prompt_length(gen: StoryboardGenerator, prompt: str) -> None:
    """Reject prompts that SDXL's 77-token text encoders would silently truncate."""
    for tokenizer_name in ("tokenizer", "tokenizer_2"):
        tokenizer = getattr(gen.pipe, tokenizer_name, None)
        if tokenizer is None:
            continue
        encoded = tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"]
        token_count = len(encoded)
        if token_count > 77:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Prompt is too long for SDXL ({token_count} tokens in {tokenizer_name}; "
                    "maximum 77). Shorten the scene description and keep character placement concise."
                ),
            )


def _run_multi_character(
    gen: StoryboardGenerator,
    reference_a: Image.Image,
    reference_b: Image.Image,
    prompt: str,
    scale: float,
    steps: int,
    guidance: float,
    seed: int,
) -> dict[str, Any]:
    """Generate one scene using left/right masked IP-Adapter reference conditioning."""
    started = time.perf_counter()
    width = 1024
    height = 1024

    # Keep the user's prompt concise; the suffix explicitly binds each reference
    # to a region, while masks provide the actual spatial conditioning.
    full_prompt = (
        f"{prompt.strip()} Exactly two characters: A on the left; "
        "B on the right. No other people."
    )
    _check_sdxl_prompt_length(gen, full_prompt)

    left_edge = int(width * 0.50)
    right_edge = int(width * 0.50)
    left_mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(left_mask).rectangle((0, 0, left_edge, height), fill=255)
    right_mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(right_mask).rectangle((right_edge, 0, width, height), fill=255)

    processor = IPAdapterMaskProcessor()
    masks = processor.preprocess([left_mask, right_mask], height=height, width=width)
    # The adapter expects [batch, image_count, height, width] inside a list
    # corresponding to the loaded IP-Adapter checkpoint.
    masks = masks.reshape(1, masks.shape[0], masks.shape[2], masks.shape[3])

    try:
        gen.pipe.set_ip_adapter_scale([[scale, scale]])
        generator = torch.Generator(device=gen.device).manual_seed(seed)
        result = gen.pipe(
            prompt=full_prompt,
            negative_prompt=MULTI_CHARACTER_NEGATIVE_PROMPT,
            ip_adapter_image=[[reference_a, reference_b]],
            cross_attention_kwargs={"ip_adapter_masks": [masks]},
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=height,
            width=width,
            generator=generator,
        ).images[0]
        image_id, _ = _save_image(result)
        return {
            "image_url": f"/api/images/{image_id}",
            "mode": "multi_character",
            "reference_count": 2,
            "placement": {"character_a": "left", "character_b": "right"},
            "conditioning_scale": scale,
            "steps": steps,
            "guidance_scale": guidance,
            "seed": seed,
            "generation_seconds": round(time.perf_counter() - started, 2),
            "identity_metrics_available": False,
            "note": (
                "Prototype result. Whole-scene CLIP-I/DINO scores are not reported as per-character "
                "identity scores; evaluate each character region separately."
            ),
        }
    finally:
        # Existing single-character requests set their own scale, but reset the
        # adapter to its normal scalar configuration to avoid leaking state.
        try:
            gen.pipe.set_ip_adapter_scale(0.5)
        except Exception:
            logger.exception("Could not restore the default IP-Adapter scale")


@app.get("/api/health")
def health() -> dict[str, Any]:
    gen = getattr(app.state, "generator", None)
    return {
        "status": "ok" if gen is not None else "not_ready",
        "model_loaded": gen is not None,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_error": getattr(app.state, "model_error", None),
        "auth_required": bool(os.getenv("DEMO_API_TOKEN", "").strip()),
    }


@app.get("/api/images/{image_id}")
def get_image(image_id: str):
    if not image_id.isalnum():
        raise HTTPException(status_code=400, detail="Invalid image identifier.")
    path = OUTPUT_DIR / f"{image_id}.png"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Image not found. The backend runtime may have reset.")
    return FileResponse(path, media_type="image/png")


@app.post("/api/generate", dependencies=[Depends(verify_demo_token)])
async def generate_scene(
    reference_image: UploadFile = File(...),
    prompt: str = Form(...),
    mode: str = Form("proposed"),
    threshold: float = Form(0.54),
    max_retries: int = Form(2),
    steps: int = Form(30),
    guidance: float = Form(7.5),
    seed: int = Form(42),
    negative_prompt: str = Form(DEFAULT_NEGATIVE_PROMPT),
):
    gen = get_generator()
    clean_prompt = prompt.strip()
    if not clean_prompt:
        raise HTTPException(status_code=400, detail="Enter a scene prompt.")
    if len(clean_prompt) > 1200:
        raise HTTPException(status_code=400, detail="Prompt must be 1,200 characters or fewer.")
    if mode not in {"fixed", "adaptive_only", "proposed"}:
        raise HTTPException(status_code=400, detail="Invalid generation method.")
    if not 0.0 <= threshold <= 1.0:
        raise HTTPException(status_code=400, detail="DINO threshold must be between 0 and 1.")
    if max_retries not in {0, 1, 2}:
        raise HTTPException(status_code=400, detail="Maximum retries must be 0, 1, or 2.")
    if not 10 <= steps <= 40:
        raise HTTPException(status_code=400, detail="Inference steps must be between 10 and 40.")
    if not 1.0 <= guidance <= 12.0:
        raise HTTPException(status_code=400, detail="Guidance scale must be between 1 and 12.")

    reference = await load_reference(reference_image)
    async with app.state.gpu_lock:
        try:
            return await run_in_threadpool(
                _run_one, gen, reference, clean_prompt, mode, threshold, max_retries,
                steps, guidance, seed, "live_scene", negative_prompt,
            )
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.exception("CUDA out of memory during generation")
            raise HTTPException(status_code=503, detail="GPU ran out of memory. Restart the Colab runtime and ensure only one generation backend is running.")
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Scene generation failed")
            raise HTTPException(status_code=500, detail=f"Scene generation failed: {exc}")


@app.post("/api/generate-multi", dependencies=[Depends(verify_demo_token)])
async def generate_multi_character_scene(
    character_a: UploadFile = File(...),
    character_b: UploadFile = File(...),
    prompt: str = Form(...),
    scale: float = Form(0.75),
    steps: int = Form(25),
    guidance: float = Form(7.5),
    seed: int = Form(42),
):
    """Generate a single two-character scene using masked multi-reference IP-Adapter conditioning.

    This is deliberately separate from /api/generate and /api/storyboard so the
    existing single-reference API behavior remains unchanged.
    """
    gen = get_generator()
    clean_prompt = prompt.strip()
    if not clean_prompt:
        raise HTTPException(status_code=400, detail="Enter a scene prompt.")
    if len(clean_prompt) > 1200:
        raise HTTPException(status_code=400, detail="Prompt must be 1,200 characters or fewer.")
    if not 0.1 <= scale <= 1.0:
        raise HTTPException(status_code=400, detail="IP-Adapter scale must be between 0.1 and 1.0.")
    if not 10 <= steps <= 40:
        raise HTTPException(status_code=400, detail="Inference steps must be between 10 and 40.")
    if not 1.0 <= guidance <= 12.0:
        raise HTTPException(status_code=400, detail="Guidance scale must be between 1 and 12.")

    reference_a = await load_reference(character_a)
    reference_b = await load_reference(character_b)
    if reference_a.size == reference_b.size and reference_a.tobytes() == reference_b.tobytes():
        raise HTTPException(status_code=400, detail="Upload two distinct reference images.")

    async with app.state.gpu_lock:
        try:
            return await run_in_threadpool(
                _run_multi_character, gen, reference_a, reference_b, clean_prompt,
                scale, steps, guidance, seed,
            )
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.exception("CUDA out of memory during multi-character generation")
            raise HTTPException(
                status_code=503,
                detail="GPU ran out of memory. Restart the Colab runtime and ensure only one generation backend is running.",
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Multi-character scene generation failed")
            raise HTTPException(status_code=500, detail=f"Multi-character generation failed: {exc}")


@app.get("/api/files/{artifact_id}.pdf")
def get_pdf(artifact_id: str):
    if not artifact_id.isalnum():
        raise HTTPException(status_code=400, detail="Invalid artifact identifier.")
    path = OUTPUT_DIR / f"{artifact_id}.pdf"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="PDF not found. The backend runtime may have reset.")
    return FileResponse(path, media_type="application/pdf", filename="storyframe_storyboard.pdf")


@app.post("/api/storyboard", dependencies=[Depends(verify_demo_token)])
async def generate_storyboard(
    reference_image: UploadFile = File(...),
    scenes_text: str = Form(...),
    mode: str = Form("proposed"),
    threshold: float = Form(0.54),
    max_retries: int = Form(2),
    steps: int = Form(30),
    guidance: float = Form(7.5),
    seed: int = Form(42),
):
    gen = get_generator()
    prompts = [line.strip().lstrip("-•0123456789. ").strip() for line in scenes_text.splitlines()]
    prompts = [p for p in prompts if p]
    if not prompts:
        raise HTTPException(status_code=400, detail="Enter at least one scene prompt, one per line.")
    if len(prompts) > 10:
        raise HTTPException(status_code=400, detail="Storyboard generation is limited to 10 scenes per request.")
    if any(len(p) > 1200 for p in prompts):
        raise HTTPException(status_code=400, detail="Each scene prompt must be 1,200 characters or fewer.")
    if mode not in {"fixed", "adaptive_only", "proposed"}:
        raise HTTPException(status_code=400, detail="Invalid generation method.")
    reference = await load_reference(reference_image)
    results: list[dict[str, Any]] = []
    async with app.state.gpu_lock:
        for index, prompt in enumerate(prompts, start=1):
            scene_id = f"scene_{index:02d}"
            try:
                result = await run_in_threadpool(
                    _run_one, gen, reference, prompt, mode, threshold, max_retries,
                    steps, guidance, seed + index - 1, scene_id, DEFAULT_NEGATIVE_PROMPT,
                )
                results.append(result)
            except Exception as exc:
                logger.exception("Storyboard scene %s failed", scene_id)
                results.append({"scene_id": scene_id, "prompt": prompt, "error": str(exc)})
                if isinstance(exc, torch.cuda.OutOfMemoryError) and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    break
    successful = sum("image_url" in item for item in results)
    contact_sheet_id = _compose_contact_sheet(results)
    return {
        "requested_scenes": len(prompts),
        "generated_scenes": successful,
        "results": results,
        "storyboard_sheet_url": f"/api/images/{contact_sheet_id}" if contact_sheet_id else None,
        "storyboard_pdf_url": f"/api/files/{contact_sheet_id}.pdf" if contact_sheet_id else None,
    }


@app.post("/api/plan", dependencies=[Depends(verify_demo_token)])
def plan_scenes(
    premise: str = Form(...),
    num_scenes: int = Form(4),
):
    """Optional LLM narrative/scene planning; requires ANTHROPIC_API_KEY on backend."""
    if not os.getenv("ANTHROPIC_API_KEY", "").strip():
        raise HTTPException(
            status_code=503,
            detail="LLM scene planning is optional and requires ANTHROPIC_API_KEY on the backend. Enter scene prompts manually to continue.",
        )
    premise = premise.strip()
    if not premise:
        raise HTTPException(status_code=400, detail="Enter a story premise.")
    if not 2 <= num_scenes <= 10:
        raise HTTPException(status_code=400, detail="Choose between 2 and 10 scenes.")
    try:
        from narrative_generator import generate_scene_script
        data = generate_scene_script(premise, num_scenes)
        scenes = []
        for index, item in enumerate(data.get("scenes", []), start=1):
            prompt = str(item.get("prompt", "")).strip()
            if not prompt:
                continue
            scene = Scene(
                scene_id=str(item.get("scene_id") or f"scene_{index:02d}"),
                prompt=prompt,
                ip_adapter_scale=0.5,
            )
            scenes.append({
                "scene_id": scene.scene_id,
                "prompt": scene.prompt,
                "scene_type": classify_scene_dynamics(scene),
                "initial_scale": compute_adaptive_scale(scene),
            })
        if not scenes:
            raise ValueError("Narrative model returned no valid scene prompts.")
        return {"premise": premise, "scenes": scenes}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Scene planning failed")
        raise HTTPException(status_code=502, detail=f"Scene planning failed: {exc}")
