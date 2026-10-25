"""Run the full evaluation pipeline (defended vs. baseline).

Attack sources (user-input vector):
  --attack-set generated   program-generated direct / jailbreak / prompt-leaking /
                           multilingual samples (default)
  --attack-set chinese     self-built zh-TW direct set
  --attack-set external    AdvBench + JailbreakBench only
  --include-external       add the external sets to generated/chinese

Indirect injection (knowledge-base vector) is evaluated end-to-end: benign
trigger queries are sent while a PoisoningRetriever injects the payload into the
retrieved document, for BOTH the baseline and the defended system. Disable with
--no-indirect.
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.attack import (
    PoisoningRetriever,
    default_indirect_scenarios,
    generate_samples,
    load_chinese_attack_set,
    load_external_attack_datasets,
)
from src.attack.attack_types import AttackType
from src.attack.dataset_loader import EXTERNAL_BENCHMARKS
from src.defense import DefensePipeline
from src.evaluation import Evaluator
from src.evaluation.benign import heldout_benign_queries, sample_benign, to_benign_samples
from src.rag import RAGPipeline
from src.rag.rag_pipeline import SYSTEM_PROMPT
from src.utils import get_logger, load_config
from src.utils.tracking import ExperimentTracker

logger = get_logger("run_evaluation")


def load_external(atk_cfg: dict, max_n: int | None, seed: int) -> list:
    ext = load_external_attack_datasets(atk_cfg, names=EXTERNAL_BENCHMARKS)
    if not ext:
        logger.warning("No external datasets found — run scripts/download_external_datasets.py first.")
    if max_n and len(ext) > max_n:
        ext = random.Random(seed).sample(ext, max_n)
    return ext


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--attack-config", default="configs/attack.yaml")
    parser.add_argument("--experiment-name", default="full_evaluation")
    parser.add_argument("--output-dir", default="experiments/results")
    parser.add_argument("--attack-set", choices=["generated", "chinese", "external"], default="generated")
    parser.add_argument("--attack-types", nargs="*", default=None,
                        help="Subset of attack types for --attack-set generated (default: all).")
    parser.add_argument("--include-external", action="store_true",
                        help="Also run AdvBench / JailbreakBench samples.")
    parser.add_argument("--max-external", type=int, default=50,
                        help="Seeded sample size from the external sets (0 = all). Controls API cost.")
    parser.add_argument("--max-attacks", type=int, default=0,
                        help="Seeded sample size of the user-input attack samples (0 = all).")
    parser.add_argument("--no-indirect", action="store_true",
                        help="Skip the end-to-end indirect-injection scenarios.")
    parser.add_argument("--max-indirect", type=int, default=0,
                        help="Seeded sample size of indirect scenarios (0 = all).")
    parser.add_argument("--benign-sample", type=int, default=0,
                        help="Extra benign queries sampled from benign.json. These are L2 training "
                             "rows (in-distribution), so the default FPR uses only the held-out set.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", dest="use_wandb", action="store_false")
    args = parser.parse_args()

    cfg = load_config(args.config)
    atk_cfg = load_config(args.attack_config)

    rag = RAGPipeline(
        store_path=cfg["rag"]["vector_store_path"],
        llm_provider=cfg["llm"]["provider"],
        llm_model=cfg["llm"]["model"],
        top_k=cfg["rag"]["top_k"],
    )
    # Baseline and defended system retrieve through the SAME wrapper, so an
    # active indirect scenario poisons both identically.
    poisoner = PoisoningRetriever(rag.retriever)
    rag.retriever = poisoner
    defense = DefensePipeline.from_config(cfg, poisoner, SYSTEM_PROMPT, use_llm_audit=True)

    # Pass .run directly (not `lambda q: defense.run(q).final_answer`) so
    # AttackRunner/Evaluator capture block_layer + per-layer latency.
    defended_fn = defense.run
    baseline_fn = lambda q: rag.query(q).answer

    max_ext = args.max_external or None
    if args.attack_set == "chinese":
        # The zh-TW indirect records are run as real indirect scenarios below.
        attack_samples = [s for s in load_chinese_attack_set()
                          if s.attack_type != AttackType.INDIRECT_INJECTION]
    elif args.attack_set == "external":
        attack_samples = load_external(atk_cfg, max_ext, args.seed)
    else:
        # indirect_injection's generator only yields bare poison strings; sent
        # as a user query that is a *direct* injection, so it is excluded here
        # and indirect injection is measured through the scenarios instead.
        types = [t for t in (args.attack_types or [t.value for t in AttackType])
                 if t != AttackType.INDIRECT_INJECTION.value]
        attack_samples = generate_samples(types, attack_config=atk_cfg)
    if args.max_attacks and len(attack_samples) > args.max_attacks:
        attack_samples = random.Random(args.seed).sample(attack_samples, args.max_attacks)
    # External samples have their own cap (--max-external).
    if args.include_external and args.attack_set != "external":
        attack_samples += load_external(atk_cfg, max_ext, args.seed)
    logger.info("Loaded %d user-input attack samples (%s)", len(attack_samples), args.attack_set)

    scenarios = [] if args.no_indirect else default_indirect_scenarios(
        max_scenarios=args.max_indirect or None, seed=args.seed,
    )
    logger.info("Indirect scenarios: %d", len(scenarios))

    # FPR + latency overhead are measured on queries L2 never trained on.
    benign_samples = to_benign_samples(
        [q for q, _ in heldout_benign_queries()] + sample_benign(args.benign_sample, args.seed)
    )

    tracker = ExperimentTracker(cfg, args.experiment_name, config={
        "attack_set": args.attack_set, "include_external": args.include_external,
        "n_attacks": len(attack_samples), "n_indirect": len(scenarios),
        "n_benign": len(benign_samples), "llm": cfg["llm"],
        "defense": cfg["defense"]["layers"],
    }, use_wandb=args.use_wandb)

    evaluator = Evaluator(
        defended_fn=defended_fn,
        undefended_fn=baseline_fn,
        output_dir=args.output_dir,
        # Reuse the pipeline's own L2 instance (None if no checkpoint yet) so
        # detector_f1 in the report is a real, independently-measured number
        # instead of always 0.0.
        detector=defense.l2,
    )
    try:
        evaluator.evaluate(
            attack_samples, benign_samples, experiment_name=args.experiment_name,
            indirect_scenarios=scenarios, poisoner=poisoner, tracker=tracker,
        )
    finally:
        tracker.finish()


if __name__ == "__main__":
    main()
