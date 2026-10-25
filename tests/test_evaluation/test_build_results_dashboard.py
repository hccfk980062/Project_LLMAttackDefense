"""Tests for scripts/build_results_dashboard.py (offline, no API)."""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from build_results_dashboard import build  # noqa: E402


def test_missing_files_are_marked_unavailable(tmp_path):
    out = build(tmp_path, tmp_path / "nope.json")
    for key in ("detector", "experience_init", "experience_1", "experience_2",
                "experience_3", "experience_4", "experience_5", "full_evaluation"):
        assert out[key]["available"] is False


def test_detector_metrics_pass_through(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({
        "test": {"eval_f1": 1.0, "eval_precision": 1.0, "eval_recall": 1.0},
        "ood": {"ood_f1": 0.83, "ood_precision": 1.0, "ood_recall": 0.71},
    }), encoding="utf-8")

    out = build(tmp_path, metrics_path)
    assert out["detector"]["available"] is True
    assert out["detector"]["test"]["eval_f1"] == 1.0
    assert out["detector"]["ood"]["ood_recall"] == 0.71
    assert out["detector"]["generated_at"] is not None


def test_experience_init_computes_asr_from_attack_runner_records(tmp_path):
    records = [
        {"attack_type": "direct_injection", "is_success": True, "was_blocked": False, "latency_ms": 100.0},
        {"attack_type": "direct_injection", "is_success": False, "was_blocked": True, "latency_ms": 50.0},
        {"attack_type": "jailbreak", "is_success": False, "was_blocked": False, "latency_ms": 150.0},
    ]
    (tmp_path / "Experience_Init.json").write_text(json.dumps(records), encoding="utf-8")

    out = build(tmp_path, tmp_path / "nope.json")
    section = out["experience_init"]
    assert section["available"] is True
    assert section["n"] == 3
    assert section["asr"] == 1 / 3
    assert section["blocked_rate"] == 1 / 3
    assert section["mean_latency_ms"] == 100.0
    assert section["asr_by_type"]["direct_injection"] == 0.5
    assert section["asr_by_type"]["jailbreak"] == 0.0


def test_experience_init_prefers_the_newer_file_regardless_of_lookup_order(tmp_path):
    # Regression test: the aggregator used to always prefer
    # experiments/results/Experience_Init.json (checked first) even when a
    # fresher baseline_quick.json (the current script's actual default output
    # path) existed at repo root — silently showing stale data after a rerun.
    results_dir = tmp_path / "experiments" / "results"
    results_dir.mkdir(parents=True)

    stale = results_dir / "Experience_Init.json"
    stale.write_text(json.dumps([
        {"attack_type": "direct_injection", "is_success": True, "was_blocked": False, "latency_ms": 1.0},
    ]), encoding="utf-8")

    fresh_path = tmp_path / "baseline_quick.json"
    fresh_path.write_text(json.dumps([
        {"attack_type": "direct_injection", "is_success": False, "was_blocked": False, "latency_ms": 1.0},
        {"attack_type": "direct_injection", "is_success": False, "was_blocked": False, "latency_ms": 1.0},
    ]), encoding="utf-8")

    now = time.time()
    os.utime(stale, (now - 100, now - 100))   # older
    os.utime(fresh_path, (now, now))          # newer

    out = build(results_dir, tmp_path / "nope.json")
    section = out["experience_init"]
    assert section["source_file"].replace("\\", "/").endswith("baseline_quick.json")
    assert section["n"] == 2
    assert section["asr"] == 0.0


def test_ablation_configs_survive_either_schema_version(tmp_path):
    # Old scripts wrote 5 configs, new ones write 8 (adds l1_only/l2_only/full) —
    # the aggregator must not assume either count.
    (tmp_path / "ablation_results.json").write_text(json.dumps({
        "no_defense": {"label": "No Defense", "asr": 0.5, "blocked_rate": 0.0},
        "full": {"label": "Full Pipeline (L1-L5)", "asr": 0.0, "blocked_rate": 0.375},
    }), encoding="utf-8")

    out = build(tmp_path, tmp_path / "nope.json")
    names = {c["name"] for c in out["experience_3"]["configs"]}
    assert names == {"no_defense", "full"}
