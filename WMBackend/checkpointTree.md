# WMBackend checkpoints layout

Large model weights and auxiliary assets live under **`WMBackend/checkpoints/`**. Each **top-level subdirectory** usually corresponds to one **service family** (the name on disk may differ slightly from the unified **`model_id`** in `vllm_wm/registry.py`).

This folder only holds **documentation**; it does not contain weights.

## `model_id` → on-disk directory

| `model_id` (registry) | Default checkpoint root on disk |
| --- | --- |
| `matrixgame` | `checkpoints/matrixgame2/` |
| `matrixgame3` | `checkpoints/matrixgame3/` |
| `yume` | `checkpoints/yume/` |
| `diamond` | `checkpoints/diamond/` |
| `open-oasis` | `checkpoints/openoasis/` |
| `wham` | `checkpoints/wham/` |
| `vid2world` | `checkpoints/vid2world/` |
| `infinite-world` | `checkpoints/infiniteworld/` |
| `worldplay` | `checkpoints/worldplay/` |
| `mineworld` | `checkpoints/mineworld/` |
| `lingbot-world-fast` | `checkpoints/lingbotworld/lingbot-world-base-cam/` (default; override with `WM_LINGBOTWORLDFAST_CHECKPOINT_ROOT`) |

Exact filenames and optional Hugging Face **`.cache/`** trees are defined in each service under `WMBackend/services/<service>/app.py` (search for `CHECKPOINT_ROOT` / `CKPT`).

## Optional: `blobs/`

Some clones use a shared **Hugging Face Hub** blob store at `checkpoints/blobs/`. It can grow very large and is not required for every model. You may omit it in fresh installs if you only use manually copied checkpoints.

---

## Directory tree (folders only)

Below is a **directory-only** snapshot (depth ≤ 3 under each top-level model folder). **`.cache/`**, **`__pycache__/`**, and individual weight files are omitted for readability.

```
checkpoints/
├── README/                    # this documentation
├── .gitkeep
├── blobs/                     # optional HF hub blob store (content omitted)
├── diamond/
│   └── csgo/
│       ├── config/
│       ├── model/
│       └── spawn/
├── infiniteworld/
│   └── models/
│       └── google/
├── lingbotworld/
│   └── lingbot-world-base-cam/
│       ├── assets/
│       ├── examples/
│       ├── google/
│       ├── high_noise_model/
│       └── lingbot_world_fast/
├── matrixgame2/
│   ├── base_distilled_model/
│   ├── base_model/
│   ├── gta_distilled_model/
│   ├── templerun_distilled_model/
│   └── xlm-roberta-large/
├── matrixgame3/
│   ├── base_model/
│   └── google/
│       └── umt5-xxl/
├── mineworld/
│   └── vae/
├── openoasis/                 # layout depends on what you install; often HF-style root
├── vid2world/
│   └── Vid2World-CSGO/
├── wham/                      # e.g. WHAM_200M.ckpt at shallow depth when populated
├── worldplay/
│   ├── HY-WorldPlay/
│   │   ├── ar_distilled_action_model/
│   │   ├── ar_model/
│   │   ├── ar_rl_model/
│   │   ├── bidirectional_model/
│   │   ├── wan_distilled_model/
│   │   └── wan_transformer/
│   └── Wan2.2-TI2V-5B-Diffusers/
│       ├── assets/
│       ├── examples/
│       ├── scheduler/
│       ├── text_encoder/
│       ├── tokenizer/
│       ├── transformer/
│       └── vae/
└── yume/
    ├── InternVL3-2B-Instruct/
    └── Yume-5B-720P/
        └── google/
```

## Refreshing this tree

After you add or reorganize checkpoints, regenerate a folder-only view (example):

```bash
cd WMBackend/checkpoints
find . -type d \( -path '*/.cache/*' -o -path './blobs/*' \) -prune -o -type d -print | sort | sed 's|[^/]*/|  |g'
```

Then paste the relevant portion into this README or a sibling `TREE.snapshot.txt` if you prefer to keep the prose stable.
