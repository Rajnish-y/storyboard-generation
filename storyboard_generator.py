"""
Multimodal Generation with IP-Adapter for Narrative and Character Consistency using SDXL
----------------------------------------------------------------------------------------
Core pipeline:
    character reference + per-scene text prompt
        -> scene-adaptive IP-Adapter conditioning
        -> SDXL
        -> optional DINO-guided self-correction

Main research mechanism:
1. Scene-Adaptive Conditioning:
   determine an initial IP-Adapter scale from scene dynamics.

2. DINO-Guided Self-Correction:
   generate -> measure DINO identity consistency -> if below threshold,
   increase reference conditioning -> regenerate, up to a fixed retry limit.

The Fixed, Adaptive-only, and Proposed methods can therefore be compared
experimentally.
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
    prompt: str

    negative_prompt: str = (
        "blurry, deformed, extra limbs, low quality, "
        "portrait, close-up, plain background, studio background, "
        "illustration, comic, cartoon, sketch, line art, "
        "black and white, monochrome"
    )

    # Used as a fallback/default.
    # Adaptive experiments should use compute_adaptive_scale().
    ip_adapter_scale: float = 0.5


@dataclass
class Storyboard:
    character_ref_path: str
    scenes: List[Scene] = field(default_factory=list)

    @classmethod
    def from_json(cls, path: str) -> "Storyboard":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        scenes = [
            Scene(**scene)
            for scene in data["scenes"]
        ]

        return cls(
            character_ref_path=data["character_ref_path"],
            scenes=scenes,
        )


# ---------------------------------------------------------------------------
# 2. Scene-Adaptive Conditioning
# ---------------------------------------------------------------------------

# These are INITIAL experimental values.
# They should be validated through the ablation experiment.
STATIC_SCALE = 0.60
NEUTRAL_SCALE = 0.50
DYNAMIC_SCALE = 0.40


def classify_scene_dynamics(scene: Scene) -> str:
    """
    Deterministically classify a scene as:
        - static
        - dynamic
        - neutral

    Classification is based on action/dynamics keywords in the scene prompt.
    """

    prompt = scene.prompt.lower()

    dynamic_keywords = [
        "running",
        "sprinting",
        "jumping",
        "fighting",
        "falling",
        "dancing",
        "chasing",
        "escaping",
        "attacking",
        "shooting",
        "motion",
        "running",
        "rush",
        "rushing",
        "combat",
        "explosion",
        "climbing",
        "throwing",
        "kicking",
        "sliding",
    ]

    static_keywords = [
        "standing",
        "sitting",
        "talking",
        "looking",
        "waiting",
        "calm",
        "posing",
        "kneeling",
        "grieving",
        "portrait",
        "observing",
        "resting",
        "thinking",
        "smiling",
        "crying",
    ]

    dynamic_score = sum(
        1 for word in dynamic_keywords if word in prompt
    )

    static_score = sum(
        1 for word in static_keywords if word in prompt
    )

    if dynamic_score > static_score and dynamic_score > 0:
        return "dynamic"

    if static_score > dynamic_score:
        return "static"

    return "neutral"


def compute_adaptive_scale(scene: Scene) -> float:
    """
    Scene-Adaptive Conditioning (SACS).

    The initial IP-Adapter scale is selected deterministically from the
    scene's estimated dynamics.

    Static scene:
        stronger reference conditioning

    Dynamic scene:
        weaker reference conditioning to allow more freedom for motion/action

    Neutral/ambiguous scene:
        balanced conditioning

    Returns:
        float: initial IP-Adapter scale
    """

    scene_type = classify_scene_dynamics(scene)

    if scene_type == "static":
        return STATIC_SCALE

    if scene_type == "dynamic":
        return DYNAMIC_SCALE

    return NEUTRAL_SCALE


# ---------------------------------------------------------------------------
# 3. Pipeline wrapper: SDXL + IP-Adapter
# ---------------------------------------------------------------------------

class StoryboardGenerator:

    def __init__(
        self,
        base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
        ip_adapter_repo: str = "h94/IP-Adapter",
        ip_adapter_subfolder: str = "sdxl_models",
        ip_adapter_weight_name: str = "ip-adapter_sdxl.bin",
        device: str = "cuda",
        dtype=torch.float16,
    ):
        self.device = device
        self.dtype = dtype

        # SDXL VAE optimized for fp16 inference.
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=dtype,
        )

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            base_model,
            vae=vae,
            torch_dtype=dtype,
        ).to(device)

        # Loads IP-Adapter reference-conditioning weights.
        self.pipe.load_ip_adapter(
            ip_adapter_repo,
            subfolder=ip_adapter_subfolder,
            weight_name=ip_adapter_weight_name,
        )

    # -----------------------------------------------------------------------
    # Basic generation
    # -----------------------------------------------------------------------

    def generate_scene(
        self,
        ref_image: Image.Image,
        scene: Scene,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: int = 42,
    ) -> Image.Image:
        """
        Generate one scene using joint:
            identity reference + scene text
        conditioning.
        """

        self.pipe.set_ip_adapter_scale(
            scene.ip_adapter_scale
        )

        generator = torch.Generator(
            device=self.device
        ).manual_seed(seed)

        result = self.pipe(
            prompt=scene.prompt,
            negative_prompt=scene.negative_prompt,
            ip_adapter_image=ref_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        return result.images[0]

    # -----------------------------------------------------------------------
    # Proposed method: Adaptive + DINO self-correction
    # -----------------------------------------------------------------------

    def generate_scene_with_self_correction(
        self,
        ref_image: Image.Image,
        scene: Scene,
        dino_threshold: float = 0.54,
        max_retries: int = 2,
        scale_step: float = 0.10,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
    ):
        """
        Proposed method:

            Scene
              ↓
        adaptive initial scale
              ↓
           generate
              ↓
          DINO score
              ↓
        score >= threshold?
           /          \
         YES            NO
          |              |
        accept       increase scale
                         |
                     regenerate
                         |
                    max retries

        Returns:
            best_image
            best_scores
        """

        # IMPORTANT:
        # Do not depend on a manually entered scale from scenes.json.
        # Calculate the initial scale from the scene itself.
        initial_scale = compute_adaptive_scale(scene)

        current_scale = initial_scale

        best_image = None

        best_scores = {
            "clip_i": -1.0,
            "dino": -1.0,
        }

        best_attempt = 0

        scene_type = classify_scene_dynamics(scene)

        print(
            f"[SACS] {scene.scene_id}: "
            f"type={scene_type}, "
            f"initial_scale={initial_scale:.2f}"
        )

        for attempt in range(max_retries + 1):

            trial_scene = Scene(
                scene_id=scene.scene_id,
                prompt=scene.prompt,
                negative_prompt=scene.negative_prompt,
                ip_adapter_scale=current_scale,
            )

            image = self.generate_scene(
                ref_image,
                trial_scene,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=42 + attempt,
            )

            scores = identity_scores(
                ref_image,
                image
            )

            print(
                f"  [{scene.scene_id}] "
                f"attempt={attempt} "
                f"scale={current_scale:.2f} "
                f"CLIP-I={scores['clip_i']:.4f} "
                f"DINO={scores['dino']:.4f}"
            )

            # Keep the best result even if none passes the threshold.
            if scores["dino"] > best_scores["dino"]:
                best_image = image
                best_scores = dict(scores)
                best_attempt = attempt

            # Accept immediately if identity consistency is sufficient.
            if scores["dino"] >= dino_threshold:
                break

            # Otherwise increase reference conditioning.
            current_scale = min(
                current_scale + scale_step,
                1.0,
            )

        best_scores["retries"] = best_attempt
        best_scores["initial_scale"] = initial_scale
        best_scores["final_scale"] = current_scale

        return best_image, best_scores

    # -----------------------------------------------------------------------
    # Full storyboard generation
    # -----------------------------------------------------------------------

    def generate_storyboard(
        self,
        storyboard: Storyboard,
        out_dir: str,
        self_correct: bool = True,
    ) -> List[str]:

        os.makedirs(
            out_dir,
            exist_ok=True
        )

        ref_image = Image.open(
            storyboard.character_ref_path
        ).convert("RGB")

        output_paths = []

        for scene in storyboard.scenes:

            if self_correct:

                frame, scores = (
                    self.generate_scene_with_self_correction(
                        ref_image,
                        scene,
                    )
                )

                print(
                    f"[final] {scene.scene_id}: "
                    f"CLIP-I={scores['clip_i']:.4f} "
                    f"DINO={scores['dino']:.4f} "
                    f"initial_scale={scores['initial_scale']:.2f} "
                    f"final_scale={scores['final_scale']:.2f} "
                    f"retries={scores['retries']}"
                )

            else:

                # Non-self-correcting mode.
                # Still use the adaptive initial scale.
                adaptive_scale = compute_adaptive_scale(scene)

                adaptive_scene = Scene(
                    scene_id=scene.scene_id,
                    prompt=scene.prompt,
                    negative_prompt=scene.negative_prompt,
                    ip_adapter_scale=adaptive_scale,
                )

                frame = self.generate_scene(
                    ref_image,
                    adaptive_scene,
                )

            out_path = os.path.join(
                out_dir,
                f"{scene.scene_id}.png",
            )

            frame.save(out_path)

            output_paths.append(out_path)

            print(
                f"[generated] {scene.scene_id} -> {out_path}"
            )

        return output_paths


# ---------------------------------------------------------------------------
# 4. Example scene script
# ---------------------------------------------------------------------------

EXAMPLE_SCRIPT = {
    "character_ref_path": "character_ref.png",

    "scenes": [
        {
            "scene_id": "scene_01",

            "prompt": (
                "photorealistic, wide shot, full scene visible, "
                "character standing confidently in a busy crowded market "
                "with visible stalls and people, determined expression, "
                "bright daylight, dynamic pose"
            ),

            "ip_adapter_scale": 0.5,
        },

        {
            "scene_id": "scene_02",

            "prompt": (
                "photorealistic, wide shot, full scene visible, "
                "character sprinting through a rain-soaked narrow alley "
                "with visible wet cobblestones and buildings, "
                "panicked wide-eyed expression, dark night lighting, "
                "motion, dynamic pose"
            ),

            "ip_adapter_scale": 0.5,
        },

        {
            "scene_id": "scene_03",

            "prompt": (
                "photorealistic, wide shot, full scene visible, "
                "character kneeling on the ground beside a wounded ally "
                "lying down, visible grief and tears, "
                "dim flickering candlelight, dramatic shadows, "
                "dynamic pose"
            ),

            "ip_adapter_scale": 0.5,
        },
    ],
}


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    if not os.path.exists("scenes.json"):

        with open(
            "scenes.json",
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                EXAMPLE_SCRIPT,
                f,
                indent=2,
            )

        print(
            "Wrote scenes.json. "
            "Edit it and provide character_ref.png before running again."
        )

    else:

        storyboard = Storyboard.from_json(
            "scenes.json"
        )

        generator = StoryboardGenerator()

        generator.generate_storyboard(
            storyboard,
            out_dir="storyboard_output",
            self_correct=True,
        )