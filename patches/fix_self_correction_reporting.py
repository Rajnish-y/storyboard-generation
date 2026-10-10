"""Patch storyboard_generator.py to report retry metadata accurately.

Run from repository root:
    python patches/fix_self_correction_reporting.py

A backup is created before the file is modified.
"""
from pathlib import Path
import sys

path = Path("storyboard_generator.py")
if not path.is_file():
    raise SystemExit("Run this script from the repository root; storyboard_generator.py was not found.")
text = path.read_text(encoding="utf-8")
start_marker = "    def generate_scene_with_self_correction("
end_marker = "    # -----------------------------------------------------------------------\n    # Full storyboard generation"
start = text.find(start_marker)
end = text.find(end_marker, start + len(start_marker))
if start < 0 or end < 0:
    raise SystemExit("Could not locate self-correction method and its end marker. No file was changed.")
current_method = text[start:end]
if '"attempt_count"' in current_method and '"selected_scale"' in current_method and '"threshold_met"' in current_method:
    print("Self-correction metadata already appears patched; no change made.")
    raise SystemExit(0)

replacement = '''    def generate_scene_with_self_correction(
        self,
        ref_image: Image.Image,
        scene: Scene,
        dino_threshold: float = 0.54,
        max_retries: int = 2,
        scale_step: float = 0.10,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
    ):
        """Generate, score, and retry; report actual retry and selection metadata."""
        max_retries = min(max(0, int(max_retries)), 2)
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
        # Keep the legacy key but define it as the scale of the selected image.
        best_scores["final_scale"] = float(best_scale)
        best_scores["last_attempt_scale"] = float(attempt_trace[-1]["scale"])
        best_scores["threshold"] = float(dino_threshold)
        best_scores["threshold_met"] = bool(threshold_met)
        best_scores["attempts"] = attempt_trace
        return best_image, best_scores

'''
backup = path.with_suffix(path.suffix + ".before_retry_fix.bak")
if not backup.exists():
    backup.write_text(text, encoding="utf-8")
path.write_text(text[:start] + replacement + text[end:], encoding="utf-8")
print(f"Patched {path}. Backup: {backup}")
print("Now run a single-scene smoke test before rerunning the full ablation.")
