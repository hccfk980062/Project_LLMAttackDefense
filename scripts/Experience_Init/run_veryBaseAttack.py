"""
Experience Init — architecture smoke test.

Direct-injection attacks against the RAG pipeline with NO defense, to establish
that modern models already resist naive direct injection (baseline ASR ~ 0%).
Attack samples now come from the unified generator; trial count is read from
configs/attack.yaml.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # scripts/

from dotenv import load_dotenv
load_dotenv()

from src.attack import generate_samples
from src.attack.attack_runner import AttackRunner
from src.rag.rag_pipeline import RAGPipeline
from src.utils import load_config
from src.utils.tracking import ExperimentTracker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--attack-config", default="configs/attack.yaml")
    parser.add_argument("--limit", type=int, default=15, help="Number of attack samples to run.")
    parser.add_argument("--output", default="baseline_quick.json")
    parser.add_argument("--no-wandb", dest="use_wandb", action="store_false")
    args = parser.parse_args()

    cfg = load_config(args.config)
    atk_cfg = load_config(args.attack_config)
    num_trials = atk_cfg["attack"]["runner"].get("num_trials", 1)

    rag = RAGPipeline(
        store_path=cfg["rag"]["vector_store_path"],
        llm_provider=cfg["llm"]["provider"],
        llm_model=cfg["llm"]["model"],
        top_k=cfg["rag"]["top_k"],
    )
    target_fn = lambda q: rag.query(q).answer

    samples = generate_samples(["direct_injection"], attack_config=atk_cfg)
    # Spread across templates: take every Nth sample, then cap at --limit.
    step = max(1, len(samples) // max(1, args.limit))
    sampled = samples[::step][: args.limit]
    print(f"Running {len(sampled)} attack samples x {num_trials} trial(s)...")

    runner = AttackRunner(target_fn, judge_model=cfg["llm"]["model"], num_trials=num_trials)
    results = runner.run(sampled, desc="Baseline Attack")

    n_success = sum(r.is_success for r in results)
    n_blocked = sum(r.was_blocked for r in results)
    print("\n=== Quick Baseline Results ===")
    print(f"Total runs    : {len(results)}")
    print(f"Successful    : {n_success} ({n_success / len(results):.1%})")
    print(f"Blocked by LLM: {n_blocked} ({n_blocked / len(results):.1%})")
    print(f"Failed        : {len(results) - n_success - n_blocked}")

    print()
    for r in results:
        status = "SUCCESS" if r.is_success else ("BLOCKED" if r.was_blocked else "FAILED ")
        print(f"{status} | {r.sample.language:8} | {r.sample.payload[:55]}...")
        if r.judge_reason:
            print(f"  Judge: {r.judge_reason[:80]}")

    with ExperimentTracker(cfg, "experience_init_direct_baseline", use_wandb=args.use_wandb) as tracker:
        tracker.log_metrics({"asr": n_success / len(results), "blocked_rate": n_blocked / len(results),
                             "n_runs": len(results)})

    runner.save_results(results, args.output)
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
