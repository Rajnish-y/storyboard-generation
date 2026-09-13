"""
narrative_generator.py
------------------------------------------------------------------------
Pipeline Stages 1-3: Narrative Generation -> Scene Segmentation -> Scene/Shot
Prompt Building.

Takes a one-line story premise and produces a scenes.json fully compatible
with storyboard_generator.py (Stage 5). Runs entirely on CPU — no GPU
needed for this stage, so it's meant to run on your LOCAL machine, not
Colab. You only need to move scenes.json (not this script) into Colab
afterward.

Setup (local machine):
    pip install anthropic

    Get an API key from https://console.anthropic.com/ and set it as an
    environment variable (see the step-by-step guide in chat).

Usage:
    python narrative_generator.py "A soldier searches for his missing sister during a city siege"
    python narrative_generator.py "A detective chases a thief through a rainy city" --num_scenes 4
"""

import argparse
import json
import os
import re

import anthropic


# System prompt encodes everything we learned from tuning storyboard_generator.py:
# - always demand a wide shot (fixes the portrait-copying problem)
# - always anchor style with "photorealistic" (fixes the illustration-drift problem)
# - use ip_adapter_scale 0.5 as the empirically-tuned balanced default,
#   nudged slightly per scene based on how static vs. dynamic it is
SYSTEM_PROMPT = """You are a professional storyboard writer and cinematographer assistant.
Given a one-line story premise, break it into a sequence of visually distinct
scenes suitable for AI image generation with a fixed character reference image.

For each scene, write ONE detailed image-generation prompt describing, in this order:
1. Always start with the word "photorealistic" (keeps visual style consistent across scenes)
2. Always specify "wide shot, full scene visible" (prevents the model from defaulting to a tight portrait)
3. The setting/environment, described specifically and visually
4. The character's action and body pose
5. The character's facial expression / emotion
6. Lighting and time of day

Also classify each scene's scene_type as either:
- "static"  — calm, still, emotionally intimate, minimal body motion
- "dynamic" — action, motion, running/fighting/fast movement

Do NOT assign any numeric conditioning parameters yourself — only classify scene_type.
The conditioning scale is computed separately by our own adaptive-scaling function.

Return ONLY valid JSON, no other text, no markdown fences, in exactly this shape:
{
  "scenes": [
    {
      "scene_id": "scene_01",
      "prompt": "photorealistic, wide shot, full scene visible, ...",
      "scene_type": "static"
    }
  ]
}"""


# --- Scene-Adaptive Conditioning (SACS) -----------------------------------
# This is the actual technical contribution: a deterministic mapping from
# scene type to ip_adapter_scale, derived from our own ablation showing that
# a single fixed scale cannot satisfy both identity fidelity and scene
# compliance across all scene types. Kept as an explicit, inspectable
# function (not an LLM judgment call) so it's reproducible and defensible
# as "our method" rather than a black-box heuristic.
def compute_adaptive_scale(scene_type: str, base_scale: float = 0.5) -> float:
    if scene_type == "static":
        return round(base_scale + 0.05, 2)   # more room to favor identity when there's little motion
    elif scene_type == "dynamic":
        return round(base_scale - 0.05, 2)   # favor scene compliance when motion/action dominates
    return base_scale


def generate_scene_script(premise: str, num_scenes: int = 5) -> dict:
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from your environment

    user_message = (
        f"Story premise: {premise}\n\n"
        f"Break this into exactly {num_scenes} scenes, in narrative order, "
        f"building rising tension across the sequence."
    )

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    raw_text = response.content[0].text.strip()
    # Defensive cleanup in case the model wraps output in ```json fences anyway
    raw_text = re.sub(r"^```(json)?\s*|\s*```$", "", raw_text.strip())

    return json.loads(raw_text)


def save_scene_script(data: dict, character_ref_path: str, out_path: str = "scenes.json"):
    scenes = []
    for scene in data["scenes"]:
        scene_type = scene.get("scene_type", "static")
        scale = compute_adaptive_scale(scene_type)
        scenes.append({
            "scene_id": scene["scene_id"],
            "prompt": scene["prompt"],
            "ip_adapter_scale": scale,
            "scene_type": scene_type,  # kept for transparency / your results table
        })

    output = {"character_ref_path": character_ref_path, "scenes": scenes}
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {len(output['scenes'])} scenes -> {out_path}\n")
    for scene in output["scenes"]:
        print(f"[{scene['scene_id']}] type={scene['scene_type']} scale={scene['ip_adapter_scale']} — {scene['prompt']}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 1-3: turn a story premise into a scenes.json for storyboard_generator.py"
    )
    parser.add_argument("premise", type=str, help="One-line story premise")
    parser.add_argument("--num_scenes", type=int, default=5, help="Number of scenes to generate")
    parser.add_argument(
        "--character_ref_path",
        type=str,
        default="character_ref.png",
        help="Filename of the character reference image (used later in Colab)",
    )
    parser.add_argument("--out", type=str, default="scenes.json")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set. See the setup guide before running this script."
        )

    data = generate_scene_script(args.premise, args.num_scenes)
    save_scene_script(data, args.character_ref_path, args.out)