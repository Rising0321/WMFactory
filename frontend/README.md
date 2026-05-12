# WMFactory Frontend Gateway

This directory hosts the **unified gateway**: it does not run model inference itself. Each model worker lives under **`WMBackend/services/<model>/app.py`**, with default listen ports matching **`WMBackend/vllm_wm/registry.py`** (`MODEL_SPECS`).

- Web UI: `frontend/web/`
- Gateway API: `frontend/server.py`
- Per-model subprocess adapters: `frontend/adapters/`

On **Load model**, the gateway **lazily starts** a uvicorn worker on the configured port if nothing is listening yet. The worker working directory defaults to **`WMFactory/WMBackend/services/<name>/`** (see `adapters/runtime_utils.py`).

## Quick start

```bash
cd frontend
python -m uvicorn server:app --host 0.0.0.0 --port 8080
```

Open `http://127.0.0.1:8080` in a browser.

## Model IDs and default ports

Aligned with WMBackend `MODEL_SPECS` (each worker binds to localhost by default):

| model_id | default_port |
| --- | --- |
| matrixgame | 9003 |
| matrixgame3 | 9016 |
| yume | 9008 |
| diamond | 9001 |
| open-oasis | 9005 |
| wham | 9007 |
| vid2world | 9010 |
| infinite-world | 9011 |
| worldplay | 9009 |
| mineworld | 9012 |
| lingbot-world-fast | 9013 |

## Gateway HTTP API

- `GET /api/models`
- `POST /api/models/load`
- `GET /api/datasets`
- `POST /api/datasets/random-image`
- `POST /api/sessions/start`
- `POST /api/sessions/step`
- `POST /api/sessions/reset`
- `POST /api/sessions/progress`
