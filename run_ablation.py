"""
run_ablation.py
------------------------------------------------------------------------
Runs the 3-arm comparison your contribution claim depends on:

    Method              Conditioning            Self-correction
    ------------------  ----------------------  ----------------
    Fixed (baseline)    single fixed scale      No
    Adaptive-only       scene-dependent (SACS)  No
    Proposed            scene-dependent (SACS)  DINO-guided retry loop

For every scene, logs CLIP-I, DINO, CLIP-T, retry count, and generation
time — everything needed to say, with evidence, whether the proposed
method actually improves the identity/scene-compliance tradeoff, rather
than merely asserting it.

Usage (run in Colab, same environment as storyboard_generator.py):
    python run_ablation.py
"""

import json
import time
import csv

from PIL import Image

from storyboard_generator import Storyboard, Scene, StoryboardGenerator
from identity_metrics import identity_scores, clip_t_score

FIXED_SCALE = 0.6  # matches your original Review 2-era default, for a fair baseline


def run_fixed(gen: StoryboardGenerator, ref_image, scene: Scene):
    trial = Scene(scene.scene_id, scene.prompt, scene.negative_prompt, FIXED_SCALE)
    t0 = time.time()
    image = gen.generate_scene(ref_image, trial)
    elapsed = time.time() - t0
    scores = identity_scores(ref_image, image)
    scores["clip_t"] = clip_t_score(scene.prompt, image)
    scores["retries"] = 0
    scores["time_sec"] = round(elapsed, 2)
    return image, scores


def run_adaptive_only(gen: StoryboardGenerator, ref_image, scene: Scene):
    # scene.ip_adapter_scale already carries the SACS-assigned value from scenes.json
    t0 = time.time()
    image = gen.generate_scene(ref_image, scene)
    elapsed = time.time() - t0
    scores = identity_scores(ref_image, image)
    scores["clip_t"] = clip_t_score(scene.prompt, image)
    scores["retries"] = 0
    scores["time_sec"] = round(elapsed, 2)
    return image, scores


def run_proposed(gen: StoryboardGenerator, ref_image, scene: Scene):
    t0 = time.time()
    image, id_scores = gen.generate_scene_with_self_correction(ref_image, scene)
    elapsed = time.time() - t0
    scores = dict(id_scores)  # already includes 'retries' from the method
    scores["clip_t"] = clip_t_score(scene.prompt, image)
    scores["time_sec"] = round(elapsed, 2)
    return image, scores


def run_ablation(scenes_json_path: str = "scenes.json", out_csv: str = "ablation_results.csv"):
    with open(scenes_json_path) as f:
        data = json.load(f)

    ref_image = Image.open(data["character_ref_path"]).convert("RGB")
    scenes = [Scene(**{k: v for k, v in s.items() if k in ("scene_id", "prompt", "ip_adapter_scale")})
              for s in data["scenes"]]

    gen = StoryboardGenerator()

    rows = []
    for scene in scenes:
        for method_name, method_fn in [
            ("fixed", run_fixed),
            ("adaptive_only", run_adaptive_only),
            ("proposed", run_proposed),
        ]:
            print(f"\n=== {scene.scene_id} | {method_name} ===")
            image, scores = method_fn(gen, ref_image, scene)
            image.save(f"ablation_{scene.scene_id}_{method_name}.png")

            row = {"scene_id": scene.scene_id, "method": method_name, **scores}
            rows.append(row)
            print(row)

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved -> {out_csv}")
    return rows


if __name__ == "__main__":
    run_ablation()