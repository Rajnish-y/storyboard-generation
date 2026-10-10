# Self-correction reporting fix

During the live demo we observed that `generate_scene_with_self_correction()` returned `retries=1` after three generation attempts when attempt 1 had the highest DINO score and attempt 2 had a lower score. The implementation stored `best_attempt` under the `retries` key. That conflates two different values.

## Correct meanings

- `attempt_count`: number of generations performed (1 to `max_retries + 1`).
- `retries`: number of additional generations actually performed (`max(0, attempt_count - 1)`).
- `best_attempt`: zero-based index of the generated image that was retained.
- `selected_scale`: conditioning scale used to generate the retained image.
- `threshold_met`: whether any accepted attempt reached the DINO threshold.
- `last_attempt_scale`: scale used by the final actual generation, not the next scale value that would have been tried.

## Change required in the core class

In `storyboard_generator.py`, initialize and track `best_scale`, `attempt_count`, `threshold_met`, and the attempt trace. After the loop, replace the existing return metadata logic that currently assigns `best_scores["retries"] = best_attempt` and `best_scores["final_scale"] = current_scale` with metadata that records the values above. Do not label the next scale that was computed after the last failed attempt as the scale used for the selected image.

The FastAPI bridge in `backend/app.py` already uses corrected semantics and keeps the attempt trace for the user interface. For consistency across the UI and CLI experiment runner, the core generator should be patched before the final report/code freeze.

## Ready-to-use replacement method

Replace the existing `generate_scene_with_self_correction()` method in `StoryboardGenerator` with the following version. It uses the same conditioning behavior; the key change is accurate, separate bookkeeping.

```python
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
    initial_scale = compute_adaptive_scale(scene)
    current_scale = initial_scale
    best_image = None
    best_scores = {"clip_i": -1.0, "dino": -1.0}
    best_attempt = 0
    best_scale = initial_scale
    attempt_trace = []
    threshold_met = False

    scene_type = classify_scene_dynamics(scene)
    print(
        f"[SACS] {scene.scene_id}: type={scene_type}, "
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
        scores = identity_scores(ref_image, image)
        passed = float(scores["dino"]) >= dino_threshold
        attempt_trace.append({
            "attempt": attempt,
            "scale": round(current_scale, 2),
            "clip_i": float(scores["clip_i"]),
            "dino": float(scores["dino"]),
            "threshold_met": passed,
        })
        print(
            f"  [{scene.scene_id}] attempt={attempt} "
            f"scale={current_scale:.2f} "
            f"CLIP-I={scores['clip_i']:.4f} "
            f"DINO={scores['dino']:.4f}"
        )

        if float(scores["dino"]) > float(best_scores["dino"]):
            best_image = image
            best_scores = dict(scores)
            best_attempt = attempt
            best_scale = current_scale

        if passed:
            threshold_met = True
            break

        current_scale = min(current_scale + scale_step, 1.0)

    best_scores = dict(best_scores)
    best_scores["retries"] = max(0, len(attempt_trace) - 1)
    best_scores["attempt_count"] = len(attempt_trace)
    best_scores["best_attempt"] = best_attempt
    best_scores["initial_scale"] = float(initial_scale)
    best_scores["selected_scale"] = float(best_scale)
    # Compatibility with older reports: final_scale is the scale of the selected image.
    best_scores["final_scale"] = float(best_scale)
    best_scores["last_attempt_scale"] = float(attempt_trace[-1]["scale"])
    best_scores["threshold"] = float(dino_threshold)
    best_scores["threshold_met"] = bool(threshold_met)
    best_scores["attempts"] = attempt_trace
    return best_image, best_scores
```

Ensure the class imports `Image`, `Scene`, `compute_adaptive_scale`, `classify_scene_dynamics`, and `identity_scores` already present in the repository. Run the unit/smoke test and a single-scene trial after replacing it. If `run_ablation.py` writes the `attempts` list into CSV, you may remove `best_scores["attempts"]` for a compact CSV or serialize it using `json.dumps()`.
