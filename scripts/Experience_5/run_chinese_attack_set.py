"""
Experience 5 — Self-built Traditional Chinese attack set: baseline vs. defended.

Runs data/chinese_attack_set/direct_injection.json through:
  * baseline  : the plain RAG pipeline (no defense)
  * defended  : the full `DefensePipeline` (L1→L2→L3→L4→L5)

Reports ASR for both, plus per-category ASR and which layer blocked each attack.

Indirect section: every document in data/chinese_attack_set/indirect_injection.json
is injected (via PoisoningRetriever) into the document retrieved for each of the
first --indirect-queries trigger queries, and the benign trigger query is sent to
both systems. Note these 5 documents ARE in the L2 training data, so an L2 catch
here is in-distribution (the held-out marker payloads are Experience 2/3).
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

from src.attack import AttackType, PoisoningRetriever, TRIGGER_QUERIES, chinese_indirect_scenarios
from src.attack.attack_runner import AttackRunner
from src.utils.tracking import ExperimentTracker
from src.attack.dataset_loader import load_attack_set_file
from src.rag.rag_pipeline import RAGPipeline
from experiment_utils import (
    RESULTS_DIR, SYSTEM_PROMPT, load_all_configs, get_top_k, make_retriever, build_defense_pipeline,
)

ATTACK_SET = "data/chinese_attack_set/direct_injection.json"


def run_indirect(rag, store_path: str, default_cfg: dict, model: str, n_queries: int) -> dict:
    """zh-TW indirect set, end to end: baseline RAG vs. full DefensePipeline."""
    scenarios = chinese_indirect_scenarios(queries=TRIGGER_QUERIES[:n_queries])
    poisoner = PoisoningRetriever(make_retriever(store_path, top_k=get_top_k(default_cfg)))
    rag.retriever = poisoner
    pipeline = build_defense_pipeline(poisoner, default_cfg, use_llm_audit=True)

    base = AttackRunner(lambda q: rag.query(q).answer, judge_model=model).run_indirect(
        scenarios, poisoner, desc="[Baseline] Indirect")
    defended = AttackRunner(pipeline.run, judge_model=model).run_indirect(
        scenarios, poisoner, desc="[Defended] Indirect")

    n = len(scenarios)
    by_cat = defaultdict(lambda: {"n": 0, "base": 0, "def": 0})
    rows = []
    for s, b, d in zip(scenarios, base, defended):
        c = by_cat[s.metadata.get("category", "unknown")]
        c["n"] += 1
        c["base"] += b.is_success
        c["def"] += d.is_success
        rows.append({
            "id": s.id, "category": s.metadata.get("category", ""), "trigger_query": s.trigger_query,
            "strategy": s.strategy, "baseline_success": b.is_success, "baseline_reason": b.judge_reason,
            "defended_success": d.is_success, "defended_reason": d.judge_reason,
            "blocked": d.was_blocked, "block_layer": d.block_layer,
        })
    result = {
        "n_scenarios": n,
        "in_l2_training_data": True,
        "baseline_asr": sum(b.is_success for b in base) / n if n else 0.0,
        "defended_asr": sum(d.is_success for d in defended) / n if n else 0.0,
        "blocked_rate": sum(d.was_blocked for d in defended) / n if n else 0.0,
        "blocks_by_layer": dict(Counter(d.block_layer for d in defended if d.was_blocked)),
        "asr_by_category": {
            cat: {"baseline": c["base"] / c["n"], "defended": c["def"] / c["n"], "n": c["n"]}
            for cat, c in by_cat.items()
        },
        "records": rows,
    }
    print("\n  Indirect (knowledge-base) scenarios:")
    print(f"    Baseline ASR : {result['baseline_asr']:.0%}  |  Defended ASR : {result['defended_asr']:.0%}"
          f"  |  Blocked : {result['blocked_rate']:.0%}  {result['blocks_by_layer']}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-indirect", action="store_true", help="Skip the indirect-injection section.")
    parser.add_argument("--indirect-queries", type=int, default=4,
                        help="Trigger queries per indirect document (x5 documents).")
    parser.add_argument("--no-wandb", dest="use_wandb", action="store_false")
    args = parser.parse_args()

    default_cfg, _, _ = load_all_configs()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    store_path = default_cfg["rag"]["vector_store_path"]
    model = default_cfg["llm"]["model"]

    samples = load_attack_set_file(ATTACK_SET, default_type=AttackType.DIRECT_INJECTION)
    print("=" * 65)
    print(f"Experience 5 — 繁中攻擊資料集 ({len(samples)} samples)")
    print("=" * 65)

    top_k = get_top_k(default_cfg)
    rag = RAGPipeline(store_path=store_path, llm_provider=default_cfg["llm"]["provider"],
                      llm_model=model, top_k=top_k)
    pipeline = build_defense_pipeline(make_retriever(store_path, top_k=top_k), default_cfg, use_llm_audit=True)
    judge = AttackRunner(target_fn=None, judge_model=model)   # only used for _judge_success

    l2_active = pipeline.l2 is not None
    print(f"Defended layers: L1{'+L2' if l2_active else ''}+L3+L4+L5\n")
    print(f"{'ID':<16} {'Category':<22} {'Base':<8} {'Defended':<10} Layer")
    print("-" * 75)

    rows = []
    for s in samples:
        base_ans = rag.query(s.payload).answer
        base_success, base_reason = judge._judge_success(s.malicious_goal, base_ans)

        res = pipeline.run(s.payload)
        if res.is_blocked:
            def_success, def_reason = False, f"blocked@{res.block_layer}"
        else:
            def_success, def_reason = judge._judge_success(s.malicious_goal, res.final_answer)

        category = s.metadata.get("category", "unknown")
        rows.append({
            "id": s.id, "category": category, "language": s.language,
            "payload": s.payload, "malicious_goal": s.malicious_goal,
            "baseline_success": base_success, "baseline_reason": base_reason,
            "defended_success": def_success, "defended_reason": def_reason,
            "blocked": res.is_blocked, "block_layer": res.block_layer,
        })
        b = "SUCCESS" if base_success else "safe"
        d = "BLOCKED" if res.is_blocked else ("SUCCESS" if def_success else "safe")
        print(f"{s.id:<16} {category:<22} {b:<8} {d:<10} {res.block_layer}")

    n = len(rows)
    base_asr = sum(r["baseline_success"] for r in rows) / n
    def_asr = sum(r["defended_success"] for r in rows) / n
    blocked_rate = sum(r["blocked"] for r in rows) / n

    by_cat = defaultdict(lambda: {"n": 0, "base": 0, "def": 0})
    for r in rows:
        c = by_cat[r["category"]]
        c["n"] += 1
        c["base"] += r["baseline_success"]
        c["def"] += r["defended_success"]

    print("\n" + "=" * 65)
    print("Summary")
    print("=" * 65)
    print(f"  Baseline ASR : {base_asr:.0%}  ({int(base_asr*n)}/{n})")
    print(f"  Defended ASR : {def_asr:.0%}  ({int(def_asr*n)}/{n})")
    print(f"  Blocked rate : {blocked_rate:.0%}")
    print(f"  Blocks by layer: {dict(Counter(r['block_layer'] for r in rows if r['blocked']))}")
    print("\n  Per-category (baseline → defended ASR):")
    for cat, c in sorted(by_cat.items()):
        print(f"    {cat:<24} {c['base']}/{c['n']} → {c['def']}/{c['n']}")

    output = {
        "experiment": "chinese_attack_set_eval",
        "attack_set": ATTACK_SET,
        "n_samples": n,
        "baseline_asr": base_asr,
        "defended_asr": def_asr,
        "blocked_rate": blocked_rate,
        "l2_active": l2_active,
        "blocks_by_layer": dict(Counter(r["block_layer"] for r in rows if r["blocked"])),
        "asr_by_category": {
            cat: {"baseline": c["base"] / c["n"], "defended": c["def"] / c["n"], "n": c["n"]}
            for cat, c in by_cat.items()
        },
        "records": rows,
    }
    if not args.no_indirect:
        output["indirect"] = run_indirect(rag, store_path, default_cfg, model, args.indirect_queries)

    with ExperimentTracker(default_cfg, "experience5_chinese_attack_set", use_wandb=args.use_wandb) as tracker:
        tracker.log_metrics({k: v for k, v in output.items() if k != "records"})
        tracker.log_table("records", [{k: v for k, v in r.items() if k != "payload"} for r in rows])

    out_path = RESULTS_DIR / "chinese_attack_set_eval.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    main()
