"""Run attack experiments and save results."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.attack import generate_samples
from src.attack.attack_runner import AttackRunner
from src.defense import DefensePipeline
from src.rag import RAGPipeline
from src.rag.rag_pipeline import SYSTEM_PROMPT
from src.utils import get_logger, load_config

logger = get_logger("run_attack")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--attack-config", default="configs/attack.yaml")
    parser.add_argument("--output", default="experiments/results/attack_results.json")
    parser.add_argument("--attack-types", nargs="*", default=None,
                        help="Subset of attack types (default: all).")
    parser.add_argument("--no-defense", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    atk_cfg = load_config(args.attack_config)

    rag = RAGPipeline(
        store_path=cfg["rag"]["vector_store_path"],
        llm_provider=cfg["llm"]["provider"],
        llm_model=cfg["llm"]["model"],
    )

    if args.no_defense:
        target_fn = lambda q: rag.query(q).answer
    else:
        defense = DefensePipeline.from_config(cfg, rag.retriever, SYSTEM_PROMPT, use_llm_audit=True)
        # Pass .run directly (not `lambda q: defense.run(q).final_answer`) so
        # AttackRunner captures block_layer + per-layer latency instead of
        # collapsing the result down to a bare string.
        target_fn = defense.run

    samples = generate_samples(args.attack_types, attack_config=atk_cfg)
    logger.info(f"Running {len(samples)} attack samples...")

    runner = AttackRunner(target_fn, batch_size=atk_cfg["attack"]["runner"]["batch_size"])
    results = runner.run(samples)
    runner.save_results(results, args.output)

    n_success = sum(r.is_success for r in results)
    logger.info(f"ASR: {n_success}/{len(results)} = {n_success/len(results):.1%}")


if __name__ == "__main__":
    main()
