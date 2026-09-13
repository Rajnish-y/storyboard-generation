"""
Multimodal Generation with IP-Adapter for Narrative and Character Consistency using SDXL
------------------------------------------------------------------------------------------
Core pipeline: generates a sequence of storyboard frames from a single character
reference image + a per-scene text script, using joint IP-Adapter (identity) +
text (action/emotion) conditioning on SDXL.

Run environment: needs a CUDA GPU (Colab T4/A100, or local GPU with >=12GB VRAM).
Install:
    pip install diffusers transformers accelerate safetensors pillow torch torchvision --break-system-packages
    pip install timm scikit-learn --break-system-packages   # for DINO/CLIP-I eval
"""

import os
import json
from dataclasses import dataclass, field
from typing import List

import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline, AutoencoderKL

from identity_metrics import identity_scores


# ---------------------------------------------------------------------------
# 1. Scene script format
# ---------------------------------------------------------------------------

@dataclass
class Scene:
    scene_id: str
    prompt: str            # e.g. "character sprinting through rain, panicked expression"
    # Actively discourages the model from defaulting back to the reference's
    # tight portrait crop / plain studio background.
    negative_prompt: str = (
        "blurry, deformed, extra limbs, low quality, "
        "portrait, close-up, plain background, studio background, "
        "illustration, comic, cartoon, sketch, line art, black and white, monochrome"
    )
    ip_adapter_scale: float = 0.5   # moderate — 0.3 let identity collapse under high guidance


@dataclass
class Storyboard:
    character_ref_path: str
    scenes: List[Scene] = field(default_factory=list)

    @classmethod
    def from_json(cls, path: str) -> "Storyboard":
        with open(path, "r") as f:
            data = json.load(f)
        scenes = [Scene(**s) for s in data["scenes"]]
        return cls(character_ref_path=data["character_ref_path"], scenes=scenes)


# ---------------------------------------------------------------------------
# 2. Pipeline wrapper: SDXL + IP-Adapter, joint conditioning
# ---------------------------------------------------------------------------

class StoryboardGenerator:
    def __init__(
        self,
        base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
        ip_adapter_repo: str = "h94/IP-Adapter",
        ip_adapter_subfolder: str = "sdxl_models",
        # Using the base SDXL IP-Adapter checkpoint. The face-focused variant
        # (ip-adapter-plus-face_sdxl_vit-h) needs a different, mismatched
        # image encoder and errors out — not worth the risk this close to
        # the deadline. We instead fix composition-override purely via
        # ip_adapter_scale (kept low) and guidance_scale (kept higher below).
        ip_adapter_weight_name: str = "ip-adapter_sdxl.bin",
        device: str = "cuda",
        dtype=torch.float16,
    ):
        self.device = device
        self.dtype = dtype

        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype
        )
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            base_model, vae=vae, torch_dtype=dtype
        ).to(device)

        # Colab's T4 (~15GB VRAM) comfortably fits SDXL + IP-Adapter without
        # CPU-offload, so we load directly onto the GPU for full speed.
        # (If you ever run this on a smaller local GPU, swap this block for
        # enable_model_cpu_offload() + enable_attention_slicing() instead.)

        # Loads the IP-Adapter weights and wires the extra cross-attention path
        # that injects the reference-image embedding alongside the text embedding.
        self.pipe.load_ip_adapter(
            ip_adapter_repo,
            subfolder=ip_adapter_subfolder,
            weight_name=ip_adapter_weight_name,
        )

    def generate_scene(
        self,
        ref_image: Image.Image,
        scene: Scene,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,  # moderate — 10.0 caused style drift into illustration
        seed: int = 42,
    ) -> Image.Image:
        # This is the joint-conditioning call: ip_adapter_image (identity) and
        # prompt (scene action/emotion) are passed into the SAME forward pass,
        # not generated separately and composited.
        self.pipe.set_ip_adapter_scale(scene.ip_adapter_scale)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        result = self.pipe(
            prompt=scene.prompt,
            negative_prompt=scene.negative_prompt,
            ip_adapter_image=ref_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        return result.images[0]

    def generate_scene_with_self_correction(
        self,
        ref_image: Image.Image,
        scene: Scene,
        dino_threshold: float = 0.35,   # our balanced-config average from the ablation
        max_retries: int = 2,
        scale_step: float = 0.1,        # how much to raise scale per retry
    ) -> tuple[Image.Image, dict]:
        """
        Closed-loop self-correction: generate, score against the reference via
        DINO, and if the score falls below threshold, retry with a boosted
        ip_adapter_scale (favoring identity over scene compliance) up to
        max_retries times. Returns the best-scoring frame and its scores
        (scores dict includes 'retries': number of retries actually used).
        """
        best_image = None
        best_scores = {"clip_i": -1.0, "dino": -1.0}
        best_attempt = 0
        current_scale = scene.ip_adapter_scale

        for attempt in range(max_retries + 1):
            trial_scene = Scene(
                scene_id=scene.scene_id,
                prompt=scene.prompt,
                negative_prompt=scene.negative_prompt,
                ip_adapter_scale=current_scale,
            )
            image = self.generate_scene(ref_image, trial_scene, seed=42 + attempt)
            scores = identity_scores(ref_image, image)

            print(
                f"  [{scene.scene_id}] attempt {attempt}: scale={current_scale:.2f} "
                f"CLIP-I={scores['clip_i']:.4f} DINO={scores['dino']:.4f}"
            )

            if scores["dino"] > best_scores["dino"]:
                best_image, best_scores, best_attempt = image, scores, attempt

            if scores["dino"] >= dino_threshold:
                break  # good enough, stop retrying

            current_scale = min(current_scale + scale_step, 1.0)

        best_scores = dict(best_scores)
        best_scores["retries"] = best_attempt
        return best_image, best_scores

    def generate_storyboard(self, storyboard: Storyboard, out_dir: str, self_correct: bool = True) -> List[str]:
        os.makedirs(out_dir, exist_ok=True)
        ref_image = Image.open(storyboard.character_ref_path).convert("RGB")

        output_paths = []
        for scene in storyboard.scenes:
            if self_correct:
                frame, scores = self.generate_scene_with_self_correction(ref_image, scene)
                print(f"[final] {scene.scene_id}: CLIP-I={scores['clip_i']:.4f} DINO={scores['dino']:.4f}")
            else:
                frame = self.generate_scene(ref_image, scene)

            out_path = os.path.join(out_dir, f"{scene.scene_id}.png")
            frame.save(out_path)
            output_paths.append(out_path)
            print(f"[generated] {scene.scene_id} -> {out_path}")

        return output_paths


# ---------------------------------------------------------------------------
# 3. Example scene script (save as scenes.json and edit freely)
# ---------------------------------------------------------------------------

EXAMPLE_SCRIPT = {
    "character_ref_path": "character_ref.png",
    "scenes": [
        {
            "scene_id": "scene_01",
            "prompt": "photorealistic, wide shot, full scene visible, character standing confidently in a busy crowded market with visible stalls and people, determined expression, bright daylight, dynamic pose",
            "ip_adapter_scale": 0.5,
        },
        {
            "scene_id": "scene_02",
            "prompt": "photorealistic, wide shot, full scene visible, character sprinting through a rain-soaked narrow alley with visible wet cobblestones and buildings, panicked wide-eyed expression, dark night lighting, motion, dynamic pose",
            "ip_adapter_scale": 0.5,
        },
        {
            "scene_id": "scene_03",
            "prompt": "photorealistic, wide shot, full scene visible, character kneeling on the ground beside a wounded ally lying down, visible grief and tears, dim flickering candlelight, dramatic shadows, dynamic pose",
            "ip_adapter_scale": 0.5,
        },
    ],
}


if __name__ == "__main__":
    # Write the example script once, if not present
    if not os.path.exists("scenes.json"):
        with open("scenes.json", "w") as f:
            json.dump(EXAMPLE_SCRIPT, f, indent=2)
        print("Wrote scenes.json — edit it, add character_ref.png, then re-run.")
    else:
        sb = Storyboard.from_json("scenes.json")
        gen = StoryboardGenerator()
        gen.generate_storyboard(sb, out_dir="storyboard_output")