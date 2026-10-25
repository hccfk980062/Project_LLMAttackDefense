"""Weights & Biases experiment tracking with a silent no-op fallback.

Experiments must run the same with or without W&B: when logging is disabled
(``wandb.log_enabled: false`` in configs/default.yaml, ``--no-wandb``,
``WANDB_DISABLED`` / ``WANDB_MODE=disabled``), when ``wandb`` isn't installed, or
when no API key is configured, every method is a no-op.
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def wandb_enabled(use_wandb: bool) -> bool:
    """wandb is on only when asked for AND not disabled via environment."""
    if not use_wandb:
        return False
    if os.environ.get("WANDB_DISABLED", "").lower() in ("true", "1", "yes"):
        return False
    if os.environ.get("WANDB_MODE", "").lower() == "disabled":
        return False
    return True


def _flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        elif isinstance(v, (int, float, bool)):
            out[key] = v
    return out


class ExperimentTracker:
    def __init__(
        self,
        cfg: dict | None,
        name: str,
        config: dict | None = None,
        use_wandb: bool = True,
    ):
        wb_cfg = (cfg or {}).get("wandb", {}) or {}
        self.name = name
        self._run = None
        if not wandb_enabled(use_wandb and wb_cfg.get("log_enabled", False)):
            return
        # Offline mode needs no key; online mode without one would prompt for
        # login, which hangs a non-interactive run.
        if not os.environ.get("WANDB_API_KEY") and os.environ.get("WANDB_MODE", "").lower() != "offline":
            print("[tracking] WANDB_API_KEY not set — W&B logging skipped (use --no-wandb to silence).")
            return
        try:
            import wandb

            self._run = wandb.init(
                project=wb_cfg.get("project", "prompt-injection-defense"),
                name=name,
                config=config or {},
                reinit=True,
            )
        except Exception as e:  # never let tracking break an experiment
            logger.warning("W&B init failed (%s); continuing without tracking.", e)
            self._run = None

    @property
    def active(self) -> bool:
        return self._run is not None

    def log_metrics(self, metrics: dict, prefix: str = "") -> None:
        """Log numeric values (nested dicts are flattened to ``a/b`` keys)."""
        if self._run is None:
            return
        flat = _flatten(metrics, prefix)
        if flat:
            self._run.log(flat)

    def log_table(self, name: str, rows: list[dict]) -> None:
        if self._run is None or not rows:
            return
        import wandb

        columns = list(rows[0].keys())
        table = wandb.Table(columns=columns, data=[[r.get(c) for c in columns] for r in rows])
        self._run.log({name: table})

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None

    def __enter__(self) -> "ExperimentTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.finish()
