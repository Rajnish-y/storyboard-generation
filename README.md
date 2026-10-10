# StoryFrame - React frontend + FastAPI bridge

This extension adds a usable web interface to the existing `storyboard-generation` repository without replacing the research modules. It expects the repository's existing `storyboard_generator.py`, `identity_metrics.py`, `narrative_generator.py` (optional), and `storyboard_composer.py` to remain in the repository root.

## What the app can do

- Upload a character reference image.
- Generate one scene with the **Fixed**, **Adaptive-only**, or **Proposed** method.
- Show CLIP-I, DINO, and CLIP-T results, selected scale, actual retry count, and the full DINO attempt trace.
- Generate a sequence of up to 10 scene prompts in one request.
- Optionally generate scene prompts from a one-line premise. This feature requires `ANTHROPIC_API_KEY` on the backend; manual prompts do not.
- Download each generated PNG. Backend outputs are saved under `generated_outputs/`.
- Reuse the existing generator, initialized once when the backend starts.

## Repository layout after merging

Copy the extension's `frontend/`, `backend/`, and `patches/` directories into the existing repository root. Keep the existing research files in the root. Before a final ablation rerun, apply the two small reporting patches described below.

```text
storyboard-generation/
├── backend/
│   ├── __init__.py
│   ├── app.py
│   └── requirements-api.txt
├── patches/
│   ├── fix_self_correction_reporting.py
│   └── fix_ablation_csv_fields.py
├── frontend/
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   └── src/
│       ├── App.jsx
│       ├── main.jsx
│       └── styles.css
├── identity_metrics.py
├── narrative_generator.py        # optional LLM planning
├── run_ablation.py
├── storyboard_composer.py
└── storyboard_generator.py
```

## Before freezing the code: two reporting patches

From the repository root, run:

```powershell
python patches\fix_self_correction_reporting.py
python patches\fix_ablation_csv_fields.py
python -m py_compile storyboard_generator.py run_ablation.py backend\app.py
```

The first patch distinguishes retries actually performed from the best attempt selected and records the selected scale. The second lets the ablation CSV contain additional attempt metadata without failing on varying method fields. Each script creates a `.bak` backup before modifying a file. Review the diffs before committing. **Do not rerun the 10-scene evaluation until the patch passes a single-scene smoke test**, because new metadata fields are added to the output.

## A. Frontend on Windows / VS Code

Install Node.js 20+ if not already installed. In a PowerShell terminal:

```powershell
cd D:\storyboard-project\frontend
npm install
npm run dev
```

Open the Vite URL, normally `http://localhost:5173`.

The app defaults to API URL `http://127.0.0.1:8000`. Use the gear icon at the top right to change the API URL and, if configured, enter the temporary demo token. The frontend can run locally while the GPU backend runs in Colab.

## B. Backend on a CUDA machine

Run from the repository root, not from inside `backend/`:

```powershell
cd D:\storyboard-project
python -m pip install -r backend\requirements-api.txt
python -m pip install diffusers transformers accelerate safetensors pillow
uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

The backend expects a CUDA-capable GPU for practical generation. Your 6 GB local RTX 3050 may not have enough VRAM for SDXL + IP-Adapter + metric models together. For your demo, use Colab's T4 and keep the frontend on Windows.

## C. Colab GPU backend + temporary tunnel

Do not load SDXL manually in one notebook cell and then start a second model server in a separate process; that can load two model copies and run out of VRAM. Use one backend process that owns the model.

1. Clone the repository and merge the `backend/` folder into it (upload the extension files to Colab if necessary).
2. Run this setup cell:

```python
%cd /content/storyboard-generation
!pip install -q fastapi "uvicorn[standard]" python-multipart diffusers transformers accelerate safetensors

import os, secrets, subprocess, sys, time, pathlib

# Temporary token protects generation endpoints while the tunnel is open.
demo_token = secrets.token_urlsafe(24)
os.environ["DEMO_API_TOKEN"] = demo_token
os.environ["CORS_ORIGINS"] = "http://localhost:5173,http://127.0.0.1:5173"

backend_log = open("/content/storyframe_backend.log", "w")
backend_process = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"],
    cwd="/content/storyboard-generation",
    stdout=backend_log,
    stderr=subprocess.STDOUT,
)
print("Backend process started. Model initialization can take several minutes.")
print("Temporary demo token (enter this in the frontend settings):", demo_token)
```

3. Wait for backend startup to finish. Check the last lines of the log:

```python
!tail -n 40 /content/storyframe_backend.log
```

The API starts serving after SDXL + IP-Adapter load. If health reports `not_ready`, resolve the `model_error` before continuing.

4. Start a temporary Cloudflare quick tunnel in another cell:

```python
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /content/cloudflared
!chmod +x /content/cloudflared

import subprocess, time, re, pathlib
cloudflared_log = open("/content/storyframe_tunnel.log", "w")
tunnel_process = subprocess.Popen(
    ["/content/cloudflared", "tunnel", "--url", "http://127.0.0.1:8000", "--no-autoupdate"],
    stdout=cloudflared_log,
    stderr=subprocess.STDOUT,
)
time.sleep(8)
log_text = pathlib.Path("/content/storyframe_tunnel.log").read_text(errors="ignore")
matches = re.findall(r"https://[a-z0-9-]+\.trycloudflare\.com", log_text)
print("Temporary backend URL:", matches[-1] if matches else "Tunnel URL not found yet; run tail on storyframe_tunnel.log")
```

5. In the local frontend's settings, set **API base URL** to the printed `https://….trycloudflare.com` URL and enter the temporary token.

Security notes:
- The quick-tunnel URL and runtime are temporary. Do not use them as a production deployment.
- Never put an Anthropic key or other permanent secret into frontend code or commit it to GitHub.
- Stop the backend and tunnel when the demo is over. Regenerating the token is easy: restart the backend with a new one.

## D. Optional LLM scene planning

`/api/plan` calls the project's existing `narrative_generator.py`. To enable it, set `ANTHROPIC_API_KEY` only in the backend environment and install the optional package:

```python
!pip install -q anthropic
```

Manual prompt entry works without this optional API.

## API quick reference

- `GET /api/health` - backend/model status.
- `POST /api/generate` - one scene; multipart fields: `reference_image`, `prompt`, `mode`, `threshold`, `max_retries`, `steps`, `guidance`, `seed`.
- `POST /api/storyboard` - up to 10 prompts in `scenes_text`, one scene per line.
- `POST /api/plan` - optional premise-to-scene planning; requires `ANTHROPIC_API_KEY`.
- `GET /api/images/{image_id}` - generated PNG.

Generation is intentionally serialized to one active request to reduce GPU-memory failures. Run Uvicorn with one worker only; multiple workers could load multiple SDXL copies.

## Research reporting note

The UI's `retries` means the number of additional generations actually performed. `best_attempt` identifies the attempt whose image was selected. They are intentionally separate because the last attempt is not guaranteed to score best. If the threshold was not reached, the UI explicitly reports that and shows the best-DINO result rather than claiming success.

The 0.54 threshold is the threshold used in the team's current calibrated experiment, not a universal identity boundary. CLIP-I and DINO are similarity proxies, not proof of identity; CLIP-T is a text-image alignment proxy.
