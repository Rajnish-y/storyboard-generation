"""
run_ablation.py
------------------------------------------------------------------------
Runs the 3-arm comparison for the proposed contribution:

Method              Conditioning               Self-correction
----------------------------------------------------------------
Fixed (baseline)    single fixed scale         No
Adaptive-only       scene-dependent (SACS)     No
Proposed            scene-dependent (SACS)     DINO-guided retry loop

For every scene, the experiment records:
    - CLIP-I
    - DINO
    - CLIP-T
    - retry count
    - generation time
    - initial/final conditioning scale

This allows us to determine whether the proposed method actually improves
the identity-vs-scene-compliance trade-off.
"""

import csv
import json
import time

from PIL import Image

from storyboard_generator import (
    StoryboardGenerator,
    Scene,
    compute_adaptive_scale,
)
from identity_metrics import (
    identity_scores,
    clip_t_score,
)


# ---------------------------------------------------------------------------
# Experimental baseline
# ---------------------------------------------------------------------------

# Current balanced fixed configuration from preliminary experiments.
FIXED_SCALE = 0.5


# ---------------------------------------------------------------------------
# Method 1: Fixed baseline
# ---------------------------------------------------------------------------

def run_fixed(
    gen: StoryboardGenerator,
    ref_image: Image.Image,
    scene: Scene,
):
    """
    Fixed baseline:
    Every scene uses exactly the same IP-Adapter scale.
    No adaptive logic and no self-correction.
    """

    trial_scene = Scene(
        scene_id=scene.scene_id,
        prompt=scene.prompt,
        negative_prompt=scene.negative_prompt,
        ip_adapter_scale=FIXED_SCALE,
    )

    print(
        f"[fixed] {scene.scene_id}: "
        f"scale={FIXED_SCALE:.2f}"
    )

    t0 = time.time()

    image = gen.generate_scene(
        ref_image,
        trial_scene,
    )

    elapsed = time.time() - t0

    scores = identity_scores(
        ref_image,
        image,
    )

    scores["clip_t"] = clip_t_score(
        scene.prompt,
        image,
    )

    scores["retries"] = 0
    scores["initial_scale"] = FIXED_SCALE
    scores["final_scale"] = FIXED_SCALE
    scores["time_sec"] = round(elapsed, 2)

    return image, scores


# ---------------------------------------------------------------------------
# Method 2: Adaptive-only
# ---------------------------------------------------------------------------

def run_adaptive_only(
    gen: StoryboardGenerator,
    ref_image: Image.Image,
    scene: Scene,
):
    """
    Adaptive-only:
    Compute the initial IP-Adapter scale from the scene using SACS.

    No DINO feedback and no regeneration.
    """

    adaptive_scale = compute_adaptive_scale(scene)

    trial_scene = Scene(
        scene_id=scene.scene_id,
        prompt=scene.prompt,
        negative_prompt=scene.negative_prompt,
        ip_adapter_scale=adaptive_scale,
    )

    print(
        f"[adaptive-only] {scene.scene_id}: "
        f"scale={adaptive_scale:.2f}"
    )

    t0 = time.time()

    image = gen.generate_scene(
        ref_image,
        trial_scene,
    )

    elapsed = time.time() - t0

    scores = identity_scores(
        ref_image,
        image,
    )

    scores["clip_t"] = clip_t_score(
        scene.prompt,
        image,
    )

    scores["retries"] = 0
    scores["initial_scale"] = adaptive_scale
    scores["final_scale"] = adaptive_scale
    scores["time_sec"] = round(elapsed, 2)

    return image, scores


# ---------------------------------------------------------------------------
# Method 3: Proposed
# ---------------------------------------------------------------------------

def run_proposed(
    gen: StoryboardGenerator,
    ref_image: Image.Image,
    scene: Scene,
):
    """
    Proposed method:
        1. Compute scene-adaptive initial scale.
        2. Generate image.
        3. Evaluate DINO consistency.
        4. Increase reference conditioning if DINO is below threshold.
        5. Regenerate up to the configured retry limit.
    """

    t0 = time.time()

    image, scores = gen.generate_scene_with_self_correction(
        ref_image,
        scene,
    )

    elapsed = time.time() - t0

    scores = dict(scores)

    # CLIP-T is evaluated on the final accepted/best image.
    scores["clip_t"] = clip_t_score(
        scene.prompt,
        image,
    )

    scores["time_sec"] = round(
        elapsed,
        2,
    )

    return image, scores


# ---------------------------------------------------------------------------
# Complete ablation experiment
# ---------------------------------------------------------------------------

def run_ablation(
    scenes_json_path: str = "scenes.json",
    out_csv: str = "ablation_results.csv",
):
    """
    Run all three methods on all scenes and save results to CSV.
    """

    # -----------------------------------------------------------------------
    # Load scenes
    # -----------------------------------------------------------------------

    with open(
        scenes_json_path,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    ref_image = Image.open(
        data["character_ref_path"]
    ).convert("RGB")

    scenes = []

    for scene_data in data["scenes"]:

        scene = Scene(
            scene_id=scene_data["scene_id"],
            prompt=scene_data["prompt"],
            negative_prompt=scene_data.get(
                "negative_prompt",
                Scene(
                    scene_id="temp",
                    prompt="temp",
                ).negative_prompt,
            ),
            # This value is intentionally not used by the adaptive logic.
            ip_adapter_scale=scene_data.get(
                "ip_adapter_scale",
                0.5,
            ),
        )

        scenes.append(scene)

    # -----------------------------------------------------------------------
    # Initialize generator once
    # -----------------------------------------------------------------------

    gen = StoryboardGenerator()

    rows = []

    # -----------------------------------------------------------------------
    # Run all methods
    # -----------------------------------------------------------------------

    methods = [
        ("fixed", run_fixed),
        ("adaptive_only", run_adaptive_only),
        ("proposed", run_proposed),
    ]

    for scene in scenes:

        # Print adaptive scale for transparency.
        adaptive_scale = compute_adaptive_scale(scene)

        print("\n" + "=" * 70)
        print(f"SCENE: {scene.scene_id}")
        print(f"Adaptive initial scale: {adaptive_scale:.2f}")
        print("=" * 70)

        for method_name, method_fn in methods:

            print(
                f"\n=== {scene.scene_id} | {method_name} ==="
            )

            image, scores = method_fn(
                gen,
                ref_image,
                scene,
            )

            # Save every generated result separately.
            output_path = (
                f"ablation_"
                f"{scene.scene_id}_"
                f"{method_name}.png"
            )

            image.save(output_path)

            row = {
                "scene_id": scene.scene_id,
                "method": method_name,
                **scores,
            }

            rows.append(row)

            print(row)

    # -----------------------------------------------------------------------
    # Save CSV
    # -----------------------------------------------------------------------

    if not rows:
        raise RuntimeError(
            "No experiment results were produced."
        )

    with open(
        out_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"\nAblation results saved to: {out_csv}"
    )

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_ablation()