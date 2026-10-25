"""
Experience 3 — Ablation study: each defense layer's individual contribution.

Every configuration runs the SAME poisoned stores (one per poisoning seed) and
the same trigger queries through `DefensePipeline` with a different
`enabled_layers` subset, so the ablation uses the exact production code path
(no re-implemented defense logic). Trials per configuration = seeds × queries.

Indirect injection lives in the retrieved context. L1/L2 inspect both the user
query and (since the context scan was added) every retrieved document, dropping
flagged documents without blocking the query — "filtered" below. "Clean drop"
is the share of unpoisoned retrieved documents the scan threw away, and
"Useful" the share of queries answered (not blocked) with at least one
document left — together they expose a defense that wins by discarding
everything.
"""
import argparse
import json
import shutil
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

from experiment_utils import (
    RESULTS_DIR, add_scale_args, get_trigger_queries, rate_with_ci,
    load_all_configs, load_and_chunk, build_poisoned_store,
    get_top_k, make_retriever, build_defense_pipeline, check_marker,
)
from src.utils.tracking import ExperimentTracker


@dataclass
class Config:
    name: str
    layers: set
    label: str
    results: list = field(default_factory=list)
    # Extra DefensePipeline kwargs, e.g. L4 output_format="json".
    overrides: dict = field(default_factory=dict)


CONFIGS = [
    Config("no_defense", set(), "No Defense"),
    Config("l1_only", {"L1"}, "L1 Only (Input Sanitizer)"),
    Config("l2_only", {"L2"}, "L2 Only (Injection Detector)"),
    Config("l3_only", {"L3"}, "L3 Only (Prompt Segmentation)"),
    Config("l5_only", {"L5"}, "L5 Only (Output Validator)"),
    Config("l3_l5", {"L3", "L5"}, "L3 + L5"),
    Config("l3_l4_l5", {"L3", "L4", "L5"}, "L3 + L4 + L5"),
    # L4 structured output (proposal: 格式約束／輸出結構限制) — the answer has to
    # fit {answer, sources, refused}, and L5 checks that format.
    Config("l3_l4json_l5", {"L3", "L4", "L5"}, "L3 + L4(json) + L5", overrides={"output_format": "json"}),
    Config("full", {"L1", "L2", "L3", "L4", "L5"}, "Full Pipeline (L1-L5)"),
]


def run_config(cfg: Config, stores: dict[int, str], queries, default_cfg: dict) -> list[dict]:
    overrides = dict(cfg.overrides)
    if "L2" not in cfg.layers:
        overrides["detector_model_path"] = None
    top_k = get_top_k(default_cfg)
    rows = []
    for seed, store_path in stores.items():
        retriever = make_retriever(store_path, top_k=top_k)
        pipeline = build_defense_pipeline(
            retriever, default_cfg, enabled_layers=cfg.layers, use_llm_audit=True, **overrides,
        )
        for query, doc_tag in queries:
            res = pipeline.run(query)
            marker_found, marker = check_marker(res.final_answer)
            n_retrieved = next((t.detail.get("num_docs", 0) for t in res.traces if t.layer == "Retrieval"), 0)
            rows.append({
                "seed": seed,
                "query": query,
                "doc": doc_tag,
                "success": marker_found and not res.is_blocked,
                "blocked": res.is_blocked,
                "block_layer": res.block_layer,
                "n_retrieved": n_retrieved,
                "dropped_docs": res.dropped_docs,
                "marker": marker,
                "response": (res.final_answer or "")[:200],
                "latency_ms": res.total_latency_ms,
            })
    return rows


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    k_success = sum(r["success"] for r in rows)
    k_blocked = sum(r["blocked"] for r in rows)
    k_filtered = sum(bool(r["dropped_docs"]) for r in rows)
    n_clean_dropped = sum(not d["metadata"].get("poisoned") for r in rows for d in r["dropped_docs"])
    n_retrieved = sum(r["n_retrieved"] for r in rows)
    k_useful = sum(1 for r in rows if not r["blocked"] and len(r["dropped_docs"]) < max(1, r["n_retrieved"]))
    return {
        "asr": k_success / n, "blocked_rate": k_blocked / n, "context_filtered_rate": k_filtered / n,
        "asr_ci": rate_with_ci(k_success, n),
        "blocked_ci": rate_with_ci(k_blocked, n),
        # Denominator is every retrieved doc (poisoned ones are a minority, so
        # this is close to the clean-doc false-drop rate and never overstates it).
        "docs_dropped_clean": n_clean_dropped,
        "docs_retrieved": n_retrieved,
        "useful_answer_rate": rate_with_ci(k_useful, n),
        "mean_latency_ms": statistics.mean(r["latency_ms"] for r in rows),
        "n_trials": n,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-wandb", dest="use_wandb", action="store_false")
    parser.add_argument("--configs", nargs="*", default=None,
                        help=f"Subset of configuration names (default: all): {[c.name for c in CONFIGS]}")
    add_scale_args(parser)
    args = parser.parse_args()

    default_cfg, attack_cfg, _ = load_all_configs()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # See scripts/Experience_1/run_indirect_baseline.py — intentionally NOT
    # read from attack.yaml's general poison_ratio (0.1); Experience 1/2/3
    # are a matched set that has always used 0.30.
    poison_ratio = 0.30
    queries = get_trigger_queries(args.queries)
    configs = [c for c in CONFIGS if args.configs is None or c.name in args.configs]

    print("=" * 65)
    print("Experience 3 — Ablation Study (Indirect Injection Defense)")
    print("=" * 65)
    print(f"  {len(queries)} trigger queries ({args.queries}) × seeds {args.seeds}")

    print("\n[1/3] Building poisoned vector stores...")
    chunks = load_and_chunk()
    stores: dict[int, str] = {}
    for seed in args.seeds:
        stores[seed], attack_samples = build_poisoned_store(chunks, poison_ratio=poison_ratio, seed=seed)
        print(f"      seed {seed}: {len(attack_samples)}/{len(chunks)} chunks poisoned")

    print("\n[2/3] Running ablation configurations...\n")
    print(f"{'Config':<32} {'ASR (95% CI)':>18}  {'Blocked':>8}  {'Filtered':>8}  "
          f"{'CleanDrop':>9}  {'Useful':>7}  {'Latency':>9}")
    print("-" * 104)

    all_results = {}
    for cfg in configs:
        cfg.results = run_config(cfg, stores, queries, default_cfg)
        s = summarize(cfg.results)
        lo, hi = s["asr_ci"]["ci95"]
        clean_drop = s["docs_dropped_clean"] / max(1, s["docs_retrieved"])
        print(f"{cfg.label:<32} {s['asr']:>5.0%} [{lo:>3.0%}–{hi:>4.0%}]  {s['blocked_rate']:>7.0%}   "
              f"{s['context_filtered_rate']:>7.0%}   {clean_drop:>8.0%}   "
              f"{s['useful_answer_rate']['rate']:>6.0%}   {s['mean_latency_ms']:>7.0f}ms")
        all_results[cfg.name] = {"label": cfg.label, **s, "results": cfg.results}

    if "no_defense" in all_results:
        print("\n[3/3] Layer contribution vs. baseline\n")
        base_asr = all_results["no_defense"]["asr"]
        print(f"  Baseline ASR: {base_asr:.0%}")
        for name, data in all_results.items():
            if name == "no_defense":
                continue
            print(f"  {data['label']:<32}: {data['asr']:.0%}  (↓{base_asr - data['asr']:.0%})")

    with ExperimentTracker(default_cfg, "experience3_ablation", use_wandb=args.use_wandb) as tracker:
        tracker.log_metrics({name: {k: v for k, v in d.items() if k != "results"}
                             for name, d in all_results.items()})

    out_path = RESULTS_DIR / "ablation_results.json"
    out_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Saved → {out_path}")

    for store_dir in stores.values():
        shutil.rmtree(store_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
