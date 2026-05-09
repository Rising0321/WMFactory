from __future__ import annotations

from dataclasses import asdict

from vllm_wm.backends.base import ModelSpec
from vllm_wm.backends.managed_service_backend import ManagedServiceBackend
from vllm_wm.config import EngineConfig


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        model_id="matrixgame",
        label="Matrix-Game 2.0",
        description="Streaming interactive world model with latent history, native WASD and camera control.",
        family="latent-diffusion-streaming",
        service_dir_name="matrixgame",
        env_prefix="MATRIXGAME",
        default_port=9003,
        aliases=("matrix-game", "matrix-game-2", "matrix-game2", "matrixgame2"),
        notes=("Wan-style causal diffusion", "stateful latent KV cache"),
        start_timeout=3600.0,
        priority=10,
    ),
    ModelSpec(
        model_id="matrixgame3",
        label="Matrix-Game 3.0",
        description="Interactive process-driven world model with chunked video rollout and native prompt loop.",
        family="interactive-process",
        service_dir_name="matrixgame3",
        env_prefix="MATRIXGAME3",
        default_port=9016,
        aliases=("matrix-game-3", "matrix-game3"),
        notes=("torchrun subprocess backend", "progress reporting supported"),
        supports_progress=True,
        request_timeout=3600.0,
        preferred_visible_devices=2,
        priority=20,
    ),
    ModelSpec(
        model_id="yume",
        label="YUME 1.5",
        description="Chunked first-person image-to-video world model with discrete movement and camera prompts.",
        family="chunked-image-to-video",
        service_dir_name="yume",
        env_prefix="YUME",
        default_port=9008,
        aliases=("yume1.5", "yume-1.5"),
        notes=("Wan-based chunk generator", "InternVL prompt refinement path"),
        request_timeout=900.0,
        priority=30,
    ),
    ModelSpec(
        model_id="diamond",
        label="Diamond",
        description="Conditional diffusion world-model environment with CSGO action space and reward/end signals.",
        family="world-model-env",
        service_dir_name="diamond",
        env_prefix="DIAMOND",
        default_port=9001,
        notes=("WorldModelEnv wrapper", "supports dataset spawn fallback"),
        supports_random_dataset=True,
        default_dataset_ids=("CSGO",),
        load_timeout=1800.0,
        start_timeout=300.0,
        step_timeout=300.0,
        priority=40,
    ),
    ModelSpec(
        model_id="open-oasis",
        label="Open-Oasis 500M",
        description="Action-conditional diffusion world model with direct latent autoregression over frames.",
        family="action-conditional-diffusion",
        service_dir_name="openoasis",
        env_prefix="OPENOASIS",
        default_port=9005,
        aliases=("openoasis", "open_oasis"),
        load_timeout=1800.0,
        start_timeout=1800.0,
        reset_timeout=1800.0,
        priority=50,
    ),
    ModelSpec(
        model_id="wham",
        label="WHAM",
        description="Autoregressive token world model with 10-frame context and 16D gamepad-like action conditioning.",
        family="autoregressive-token-world-model",
        service_dir_name="wham",
        env_prefix="WHAM",
        default_port=9007,
        start_timeout=300.0,
        reset_timeout=300.0,
        priority=60,
    ),
    ModelSpec(
        model_id="vid2world",
        label="Vid2World",
        description="Academic causal video diffusion world model for CSGO with history-conditioned action rollout.",
        family="causal-video-diffusion",
        service_dir_name="vid2world",
        env_prefix="VID2WORLD",
        default_port=9010,
        aliases=("vid-2-world",),
        notes=("supports seed_meta for dataset-history reuse",),
        supports_seed_meta=True,
        supports_random_dataset=True,
        default_dataset_ids=("CSGO",),
        request_timeout=900.0,
        priority=70,
    ),
    ModelSpec(
        model_id="infinite-world",
        label="Infinite-World",
        description="Long-horizon latent-history diffusion world model with move/view action streams.",
        family="latent-history-diffusion",
        service_dir_name="infiniteworld",
        env_prefix="INFINITEWORLD",
        default_port=9011,
        aliases=("infiniteworld", "infinite_world"),
        notes=("separate generation and decode devices",),
        request_timeout=1800.0,
        preferred_visible_devices=2,
        use_dual_device_hint=True,
        priority=80,
    ),
    ModelSpec(
        model_id="worldplay",
        label="HY-WorldPlay 5B",
        description="WAN-based interactive first-person world model with rolling latent memory and camera-conditioned motion.",
        family="wan-interactive-world-model",
        service_dir_name="worldplay",
        env_prefix="WORLDPLAY",
        default_port=9009,
        aliases=("hy-worldplay", "hyworldplay"),
        notes=("HY-WorldPlay distilled action checkpoint", "prefers dual-GPU aux-device split"),
        request_timeout=3600.0,
        load_timeout=3600.0,
        start_timeout=1800.0,
        step_timeout=1800.0,
        preferred_visible_devices=2,
        priority=90,
    ),
    ModelSpec(
        model_id="mineworld",
        label="MineWorld 1200M 32f",
        description="Minecraft autoregressive world model with diagonal decoding and explicit Transformer KV cache refresh.",
        family="autoregressive-token-world-model",
        service_dir_name="mineworld",
        env_prefix="MINEWORLD",
        default_port=9012,
        aliases=("mine-world",),
        notes=("Minecraft-only domain", "uses frame/action token cache"),
        request_timeout=1800.0,
        load_timeout=1800.0,
        start_timeout=300.0,
        step_timeout=300.0,
        supports_random_dataset=True,
        default_dataset_ids=("minecraft",),
        priority=100,
    ),
    ModelSpec(
        model_id="lingbot-world-fast",
        label="LingBot-World-Fast",
        description="Causal autoregressive LingBot world model with per-call KV caching over camera-conditioned chunks.",
        family="causal-kv-world-model",
        service_dir_name="lingbotworldfast",
        env_prefix="LINGBOTWORLDFAST",
        default_port=9013,
        aliases=("lingbotworld-fast", "lingbot-fast", "lingbotworldfast"),
        notes=("camera trajectory synthesized from WASD/IJKL", "prefers dual-GPU device_map dispatch"),
        request_timeout=3600.0,
        load_timeout=3600.0,
        start_timeout=300.0,
        step_timeout=1800.0,
        preferred_visible_devices=2,
        priority=110,
    ),
)


_SPEC_BY_ID = {spec.model_id: spec for spec in MODEL_SPECS}
_SPEC_BY_ALIAS = {alias: spec.model_id for spec in MODEL_SPECS for alias in spec.aliases}


def normalize_model_id(model_id: str) -> str:
    key = model_id.strip().lower()
    if key in _SPEC_BY_ID:
        return key
    if key in _SPEC_BY_ALIAS:
        return _SPEC_BY_ALIAS[key]
    raise KeyError(f"Unsupported model '{model_id}'")


def get_model_spec(model_id: str) -> ModelSpec:
    return _SPEC_BY_ID[normalize_model_id(model_id)]


def build_backend(model_id: str, config: EngineConfig | None = None) -> ManagedServiceBackend:
    return ManagedServiceBackend(get_model_spec(model_id), config or EngineConfig())


def list_model_cards() -> list[dict]:
    cards = []
    for spec in sorted(MODEL_SPECS, key=lambda item: (item.priority, item.label.lower())):
        card = asdict(spec)
        cards.append(
            {
                "id": card["model_id"],
                "label": card["label"],
                "description": card["description"],
                "family": card["family"],
                "aliases": list(card["aliases"]),
                "notes": list(card["notes"]),
                "default_dataset_ids": list(card["default_dataset_ids"]),
                "worker_mode": "managed-service",
                "status": "available",
            }
        )
    return cards
