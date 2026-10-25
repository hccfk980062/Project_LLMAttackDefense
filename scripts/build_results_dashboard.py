"""
Aggregate every experiments/results/*.json + the trained L2 detector's metrics
into one normalized file for the results dashboard artifact.

Every experiment's output schema was kept stable across the DefensePipeline
refactor (see scripts/experiment_utils.py), so this reads the same way whether
a file was produced by the original scripts or the current ones — freshness is
communicated purely via each file's mtime, shown on the dashboard, rather than
guessed at here.

Usage:
  python scripts/build_results_dashboard.py
Output:
  experiments/results/dashboard_data.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import get_logger

logger = get_logger("build_results_dashboard")

DEFAULT_RESULTS_DIR = Path("experiments/results")
DEFAULT_DETECTOR_METRICS = Path("models/detector/checkpoints/best_model/metrics.json")


def _mtime(p: Path) -> str | None:
    return datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="minutes") if p.exists() else None


def _load(p: Path) -> Any:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        logger.warning("Could not parse %s: %s", p, e)
        return None


def _attack_runner_summary(records: list[dict]) -> dict:
    """Summarize an AttackRunner.save_results() list (Experience_Init / baseline_quick.json)."""
    n = len(records)
    if n == 0:
        return {"n": 0}
    n_success = sum(bool(r.get("is_success")) for r in records)
    n_blocked = sum(bool(r.get("was_blocked")) for r in records)
    latencies = [r["latency_ms"] for r in records if r.get("latency_ms") is not None]
    by_type: dict[str, dict[str, int]] = {}
    for r in records:
        t = r.get("attack_type", "unknown")
        bucket = by_type.setdefault(t, {"n": 0, "success": 0})
        bucket["n"] += 1
        bucket["success"] += int(bool(r.get("is_success")))
    return {
        "n": n,
        "n_success": n_success,
        "n_blocked": n_blocked,
        "asr": n_success / n,
        "blocked_rate": n_blocked / n,
        "mean_latency_ms": (sum(latencies) / len(latencies)) if latencies else None,
        "asr_by_type": {k: v["success"] / v["n"] for k, v in by_type.items()},
    }


def _section(path: Path, fields: dict[str, Any] | None) -> dict:
    return {
        "available": fields is not None,
        "source_file": str(path),
        "generated_at": _mtime(path),
        **(fields or {}),
    }


def build(results_dir: Path, detector_metrics: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"generated_at": datetime.now().isoformat(timespec="seconds")}

    # ── L2 detector: in-distribution test vs held-out OOD (queries / documents) ─
    det = _load(detector_metrics)
    out["detector"] = _section(detector_metrics, {
        "test": (det or {}).get("test"),
        "ood": (det or {}).get("ood"),
        "ood_doc": (det or {}).get("ood_doc"),
    } if det else None)

    # ── Experience Init — direct injection, no defense (smoke test) ────────
    # Two possible locations: experiments/results/Experience_Init.json (legacy)
    # and the current script's actual default, baseline_quick.json at repo
    # root. Prefer whichever one is newer rather than a fixed lookup order —
    # otherwise a stale Experience_Init.json left over from before that
    # default changed silently wins forever, even after a fresh rerun.
    init_path = results_dir / "Experience_Init.json"
    init_records = _load(init_path)
    fallback_path = results_dir.parent.parent / "baseline_quick.json"
    fallback_records = _load(fallback_path)
    if fallback_records is not None and (
        init_records is None or fallback_path.stat().st_mtime > init_path.stat().st_mtime
    ):
        init_path, init_records = fallback_path, fallback_records
    out["experience_init"] = _section(
        init_path, _attack_runner_summary(init_records) if init_records else None
    )

    # ── Experience 1 — indirect injection baseline (clean vs poisoned) ─────
    p = results_dir / "indirect_baseline.json"
    d = _load(p)
    out["experience_1"] = _section(p, {
        "clean_asr": d["clean_asr"], "poisoned_asr": d["poisoned_asr"],
        "poison_ratio": d["poison_ratio"], "n_queries": d["n_queries"],
        "n_trials": d.get("n_trials", d["n_queries"]),
        "poisoned_asr_ci": d.get("poisoned_asr_ci"),
    } if d else None)

    # ── Experience 2 — indirect injection: baseline vs defended ────────────
    p = results_dir / "indirect_defense_comparison.json"
    d = _load(p)
    out["experience_2"] = _section(p, {
        "baseline_asr": d["baseline_asr"], "defended_asr": d["defended_asr"],
        "blocked_rate": d["blocked_rate"], "layers_active": d.get("layers_active", []),
        "n_queries": d["n_queries"],
        "n_trials": d.get("n_trials", d["n_queries"]),
        "defended_asr_ci": d.get("defended_asr_ci"),
        "clean_doc_drop_rate": d.get("clean_doc_drop_rate"),
    } if d else None)

    # ── Experience 3 — ablation (per-layer contribution) ───────────────────
    p = results_dir / "ablation_results.json"
    d = _load(p)
    configs = [
        {"name": name, "label": c.get("label", name), "asr": c.get("asr"), "blocked_rate": c.get("blocked_rate"),
         "asr_ci": c.get("asr_ci"), "useful_answer_rate": c.get("useful_answer_rate")}
        for name, c in d.items()
    ] if d else []
    out["experience_3"] = _section(p, {"configs": configs} if d else None)

    # ── Experience 4 — false positive rate ──────────────────────────────────
    p = results_dir / "fpr_test.json"
    d = _load(p)
    out["experience_4"] = _section(p, {
        "n_queries": d["n_queries"], "n_blocked": d["n_blocked"], "fpr": d["fpr"],
        "fpr_ci": d.get("fpr_ci"), "benign_source": d.get("benign_source", "train_dist"),
        "target_fpr": d.get("target_fpr", 0.05),
        "blocks_by_layer": d.get("blocks_by_layer", {}),
        "fpr_by_category": d.get("fpr_by_category", {}),
        "l2_active": d.get("l2_active"),
    } if d else None)

    # ── Experience 5 — self-built zh-TW attack set ──────────────────────────
    p = results_dir / "chinese_attack_set_eval.json"
    d = _load(p)
    out["experience_5"] = _section(p, {
        "baseline_asr": d["baseline_asr"], "defended_asr": d["defended_asr"],
        "blocked_rate": d["blocked_rate"], "n_samples": d["n_samples"],
        "l2_active": d.get("l2_active"),
        "blocks_by_layer": d.get("blocks_by_layer", {}),
        "asr_by_category": d.get("asr_by_category", {}),
    } if d else None)

    # ── Full evaluation (run_evaluation.py, ASR/FPR/latency by attack type) ─
    p = results_dir / "full_evaluation_summary.json"
    d = _load(p)
    out["full_evaluation"] = _section(p, {"reports": d} if d else None)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--detector-metrics", default=str(DEFAULT_DETECTOR_METRICS))
    parser.add_argument("--out", default=None, help="Defaults to <results-dir>/dashboard_data.json")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out = build(results_dir, Path(args.detector_metrics))

    out_path = Path(args.out) if args.out else results_dir / "dashboard_data.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    available = [k for k, v in out.items() if isinstance(v, dict) and v.get("available")]
    logger.info("Wrote %s — sections available: %s", out_path, ", ".join(available))
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
