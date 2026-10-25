"""
Experience 1 — Indirect injection baseline (NO defense).

  1. Load + chunk the knowledge base
  2. Build a CLEAN store (false-positive control) and, for every poisoning
     seed, a POISONED store (a fraction of chunks get a marker payload appended)
  3. Run benign trigger queries on the clean store and on every poisoned store
  4. Score by marker detection (no LLM judge needed)

Trials = poisoning seeds × trigger queries (default 3 × 44 = 132). Each seed
poisons a different random 30% of the chunks, so the ASR averages over which
chunks happen to be poisoned instead of reflecting a single lucky/unlucky draw.

Shared logic (poisoning, marker payloads, trigger queries) lives in
scripts/experiment_utils.py so Experience 2/3 stay in sync.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI

from experiment_utils import (
    log_experiment, add_wandb_flag, add_scale_args, get_trigger_queries, fmt_rate, rate_with_ci,
    RESULTS_DIR, SYSTEM_PROMPT,
    load_all_configs, load_and_chunk, build_clean_store, build_poisoned_store,
    get_top_k, make_retriever, check_marker,
)


def run_rag_query(retriever, query: str, llm) -> str:
    docs = retriever.retrieve(query)
    context = retriever.format_context(docs)
    msgs = [
        ("system", SYSTEM_PROMPT),
        ("human", f"參考資料：\n\n{context}\n\n問題：{query}"),
    ]
    return llm.invoke(msgs).content


def main() -> None:
    default_cfg, attack_cfg, _ = load_all_configs()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        # NOTE: intentionally NOT read from configs/attack.yaml's general
        # indirect_injection.poison_ratio (0.1, used elsewhere for generic
        # sample generation) — Experience 1/2/3 are a matched set that has
        # always used 0.30 (see ExperienceNotes.txt); reading the config's
        # value here silently swapped in 0.1 since the key exists, which
        # produces a materially weaker (and non-comparable) attack.
        "--poison-ratio", type=float, default=0.30,
    )
    add_scale_args(parser)
    add_wandb_flag(parser)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    llm = ChatOpenAI(model=default_cfg["llm"]["model"], temperature=0.0)
    top_k = get_top_k(default_cfg)
    queries = get_trigger_queries(args.queries)

    print("=" * 60)
    print("Experience 1 — Indirect Injection Baseline")
    print("=" * 60)
    print(f"  {len(queries)} trigger queries ({args.queries}) × seeds {args.seeds}, top_k={top_k}")

    print("\n[1/3] Loading + chunking documents...")
    chunks = load_and_chunk()
    print(f"      {len(chunks)} chunks")

    print("\n[2/3] Clean store (false-positive control)...")
    clean_dir = build_clean_store(chunks)
    clean_retriever = make_retriever(clean_dir, top_k=top_k)
    clean_results = []
    for query, doc_tag in queries:
        resp = run_rag_query(clean_retriever, query, llm)
        success, _ = check_marker(resp)
        clean_results.append({"query": query, "doc": doc_tag, "success": success, "response": resp})
    n_clean_hits = sum(r["success"] for r in clean_results)
    print(f"      clean ASR {fmt_rate(n_clean_hits, len(clean_results))}")
    shutil.rmtree(clean_dir, ignore_errors=True)

    print(f"\n[3/3] Poisoned stores ({args.poison_ratio:.0%} of chunks, append, marker payloads)...")
    poisoned_results, per_seed = [], {}
    for seed in args.seeds:
        poisoned_dir, attack_samples = build_poisoned_store(
            chunks, poison_ratio=args.poison_ratio, seed=seed
        )
        retriever = make_retriever(poisoned_dir, top_k=top_k)
        seed_rows = []
        for query, doc_tag in queries:
            resp = run_rag_query(retriever, query, llm)
            success, marker = check_marker(resp)
            seed_rows.append({"seed": seed, "query": query, "doc": doc_tag, "success": success,
                              "marker": marker, "response": resp})
        k = sum(r["success"] for r in seed_rows)
        per_seed[str(seed)] = rate_with_ci(k, len(seed_rows))
        print(f"      seed {seed}: {len(attack_samples)} chunks poisoned → ASR {fmt_rate(k, len(seed_rows))}")
        poisoned_results += seed_rows
        shutil.rmtree(poisoned_dir, ignore_errors=True)

    n = len(poisoned_results)
    n_hits = sum(r["success"] for r in poisoned_results)
    by_doc: dict[str, list[bool]] = {}
    for r in poisoned_results:
        by_doc.setdefault(r["doc"], []).append(r["success"])

    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"  Clean RAG    ASR : {fmt_rate(n_clean_hits, len(clean_results))}  ← false positive check")
    print(f"  Poisoned RAG ASR : {fmt_rate(n_hits, n)}  ← indirect injection baseline")
    print(f"  Poison ratio     : {args.poison_ratio:.0%}   strategy: append")
    print("  By target document:")
    for doc, flags in sorted(by_doc.items()):
        print(f"    {doc:<14} {sum(flags)}/{len(flags)}")

    output = {
        "experiment": "indirect_injection_baseline",
        "clean_asr": n_clean_hits / len(clean_results),
        "poisoned_asr": n_hits / n,
        "clean_asr_ci": rate_with_ci(n_clean_hits, len(clean_results)),
        "poisoned_asr_ci": rate_with_ci(n_hits, n),
        "poisoned_asr_by_seed": per_seed,
        "poisoned_asr_by_doc": {d: rate_with_ci(sum(f), len(f)) for d, f in sorted(by_doc.items())},
        "poison_ratio": args.poison_ratio,
        "top_k": top_k,
        "query_set": args.queries,
        "seeds": args.seeds,
        "n_queries": len(queries),
        "n_trials": n,
        "clean_results": clean_results,
        "poisoned_results": poisoned_results,
    }
    log_experiment(default_cfg, "experience1_indirect_baseline", output, args.use_wandb)

    out_path = RESULTS_DIR / "indirect_baseline.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    main()
