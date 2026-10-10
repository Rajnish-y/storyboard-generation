"""API smoke test using stub generator/metrics; no model downloads required."""
import io
import sys
import types
from dataclasses import dataclass
from PIL import Image

# Make test independent from large model dependencies and actual CUDA hardware.
seq = {"dino": [0.20, 0.523, 0.7535], "i": 0}

@dataclass
class Scene:
    scene_id: str
    prompt: str
    negative_prompt: str = "negative"
    ip_adapter_scale: float = 0.5

class FakeGenerator:
    def __init__(self, **kwargs):
        self.calls = 0
    def generate_scene(self, ref_image, scene, **kwargs):
        self.calls += 1
        shade = int(90 + scene.ip_adapter_scale * 100) % 255
        return Image.new("RGB", (96, 80), (shade, 75, 125))

sg = types.ModuleType("storyboard_generator")
sg.Scene = Scene
sg.StoryboardGenerator = FakeGenerator
sg.classify_scene_dynamics = lambda scene: "dynamic" if any(k in scene.prompt.lower() for k in ["run", "sprint", "jump", "fight"]) else "static"
sg.compute_adaptive_scale = lambda scene: 0.4 if sg.classify_scene_dynamics(scene) == "dynamic" else 0.6
sys.modules["storyboard_generator"] = sg

metrics = types.ModuleType("identity_metrics")
def identity_scores(ref, candidate):
    values = seq["dino"]
    pos = min(seq["i"], len(values)-1)
    dino = values[pos]
    seq["i"] += 1
    return {"clip_i": 0.8 + (dino / 100), "dino": dino}
metrics.identity_scores = identity_scores
metrics.clip_t_score = lambda prompt, candidate: 0.301
sys.modules["identity_metrics"] = metrics

import torch
torch.cuda.is_available = lambda: True
torch.cuda.get_device_name = lambda index=0: "Fake T4"

# Stub the Diffusers symbol required during backend import.
# The smoke test uses a fake generator and does not run real inference.
fake_diffusers = types.ModuleType("diffusers")
fake_image_processor = types.ModuleType("diffusers.image_processor")

class FakeIPAdapterMaskProcessor:
    pass

fake_image_processor.IPAdapterMaskProcessor = FakeIPAdapterMaskProcessor

sys.modules["diffusers"] = fake_diffusers
sys.modules["diffusers.image_processor"] = fake_image_processor

from fastapi.testclient import TestClient
from backend import app as app_module

with TestClient(app_module.app) as client:
    health = client.get("/api/health")
    assert health.status_code == 200, health.text
    assert health.json()["model_loaded"] is True

    image = Image.new("RGB", (96, 96), (20, 40, 60))
    buf = io.BytesIO(); image.save(buf, format="PNG")
    seq["i"] = 0
    response = client.post(
        "/api/generate",
        files={"reference_image": ("reference.png", buf.getvalue(), "image/png")},
        data={"prompt": "character sprinting through a rainy alley", "mode": "proposed", "threshold": "0.54", "max_retries": "2", "steps": "10", "guidance": "7.5", "seed": "42"},
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["metrics"]["attempt_count"] == 3, result
    assert result["metrics"]["retries"] == 2, result
    assert result["metrics"]["threshold_met"] is True, result
    assert result["metrics"]["selected_scale"] == 0.6, result
    assert client.get(result["image_url"]).status_code == 200

    seq["i"] = 0
    buf = io.BytesIO(); image.save(buf, format="PNG")
    storyboard_response = client.post(
        "/api/storyboard",
        files={"reference_image": ("reference.png", buf.getvalue(), "image/png")},
        data={"scenes_text": "character standing in a room\ncharacter running in rain", "mode": "fixed", "steps": "10", "guidance": "7.5"},
    )
    assert storyboard_response.status_code == 200, storyboard_response.text
    storyboard = storyboard_response.json()
    assert storyboard["generated_scenes"] == 2, storyboard
    assert storyboard["storyboard_sheet_url"], storyboard
    assert client.get(storyboard["storyboard_sheet_url"]).status_code == 200
    assert client.get(storyboard["storyboard_pdf_url"]).status_code == 200

print("API smoke test passed: health, proposed correction trace, image retrieval, storyboard contact sheet and PDF export.")
