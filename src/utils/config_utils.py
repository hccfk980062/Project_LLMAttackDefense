"""Load and merge YAML configs with environment variable overrides."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return _apply_env_overrides(config)


def _apply_env_overrides(config: dict) -> dict:
    """Override config values with environment variables (uppercase dotted paths)."""
    env_map = {
        "OPENAI_API_KEY": None,
        "ANTHROPIC_API_KEY": None,
        "WANDB_API_KEY": None,
        "LLM_PROVIDER": ("llm", "provider"),
        "LLM_MODEL": ("llm", "model"),
        "LLM_TEMPERATURE": ("llm", "temperature"),
        "VECTOR_STORE_PATH": ("rag", "vector_store_path"),
        "EMBEDDING_MODEL": ("rag", "embedding_model"),
        "DEFENSE_ENABLED": ("defense", "enabled"),
        "DETECTOR_MODEL_PATH": ("defense", "layers", "layer2_injection_detector", "model_path"),
    }
    for env_key, config_path in env_map.items():
        val = os.environ.get(env_key)
        if val is not None and config_path:
            _set_nested(config, config_path, _coerce(val))
    return config


def _set_nested(d: dict, keys: tuple, value: Any) -> None:
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def _coerce(val: str) -> Any:
    if val.lower() in ("true", "false"):
        return val.lower() == "true"
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val
