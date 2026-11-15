from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"
EXAMPLE_PATH = ROOT / "config.example.yaml"

DEFAULT_BACKEND = "typesafe"
DEFAULT_TYPESAFE_MODEL = "jev-latest"
DEFAULT_OPENAI_MODEL = "gpt-6-luna"
DEFAULT_CONFIDENCE = 0.55
DEFAULT_TIMEOUT_SECONDS = 20.0


def _as_float(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_config(data: dict[str, Any] | None) -> dict[str, Any]:
    """Fill defaults without changing an explicit typesafe configuration."""
    cfg = dict(data or {})
    cfg.setdefault("enabled", True)
    backend = str(cfg.get("backend") or DEFAULT_BACKEND).strip().lower()
    cfg["backend"] = backend or DEFAULT_BACKEND
    cfg.setdefault("mode", "shadow")
    cfg.setdefault("model", DEFAULT_TYPESAFE_MODEL)
    cfg.setdefault("thresholds", {})
    cfg.setdefault("limits", {})
    cfg.setdefault("logging", {"path": "logs/runs.jsonl"})

    raw_openai = cfg.get("openai")
    openai_cfg = dict(raw_openai) if isinstance(raw_openai, dict) else {}
    openai_cfg.setdefault("model", DEFAULT_OPENAI_MODEL)
    threshold_default = _as_float(
        (cfg.get("thresholds") or {}).get("min_choice_confidence"),
        DEFAULT_CONFIDENCE,
    )
    if openai_cfg.get("confidence_threshold") is None:
        openai_cfg["confidence_threshold"] = threshold_default
    else:
        openai_cfg["confidence_threshold"] = _as_float(
            openai_cfg.get("confidence_threshold"),
            threshold_default,
        )
    timeout = _as_float(openai_cfg.get("timeout_seconds"), DEFAULT_TIMEOUT_SECONDS)
    openai_cfg["timeout_seconds"] = timeout if timeout > 0 else DEFAULT_TIMEOUT_SECONDS
    cfg["openai"] = openai_cfg
    return cfg


def load_config() -> dict[str, Any]:
    path = CONFIG_PATH if CONFIG_PATH.exists() else EXAMPLE_PATH
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return normalize_config(data)


def resolve_log_path(cfg: dict[str, Any]) -> Path:
    rel = (cfg.get("logging") or {}).get("path", "logs/runs.jsonl")
    path = Path(rel)
    if not path.is_absolute():
        path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
