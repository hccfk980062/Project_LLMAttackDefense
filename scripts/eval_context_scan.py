"""
Offline evaluation of the L1/L2 retrieved-context scan — no API calls.

Runs the same knowledge-base chunks the experiments use (512 / 64) through the
pipeline's context scan and reports:

  * false-drop rate — clean chunks wrongly removed, split into documents the
    detector trained on vs. DOC_HOLDOUT documents it never saw;
  * detection rate  — chunks poisoned with the Experience MARKER_PAYLOADS
    (never in training) at append / prepend / embed positions, per layer;
  * scan latency    — L2 time for one top-k batch;
  * threshold sweep — L2 false-drop / detection at several thresholds from the
    same scores, split by trained vs. held-out documents. Pick
    context_threshold from the held-out columns, not the trained ones.

Output: experiments/results/context_scan_eval.json

Usage:
  python scripts/eval_context_scan.py
  python scripts/eval_context_scan.py --context-threshold 0.9
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_detector_dataset import DOC_HOLDOUT
from experiment_utils import RESULTS_DIR, load_all_configs, load_and_chunk

from src.attack.indirect_injection import MARKER_PAYLOADS
from src.defense.layer1_input_sanitizer import InputSanitizer
from src.defense.layer2_injection_detector import InjectionDetector

POSITIONS = ("append", "prepend", "embed")
SWEEP_THRESHOLDS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99)


def poison(text: str, payload: str, position: str) -> str:
    if position == "append":
        return text + payload
    if position == "prepend":
        return payload.strip() + "\n\n" + text
    mid = len(text) // 2
    return text[:mid] + payload + "\n\n" + text[mid:]


def scan(texts: list[str], l1: InputSanitizer, l2: InjectionDetector | None, threshold: float | None):
    """Per text: (l1_flag, l2_flag, l2_confidence). L2 is scored independently of L1."""
    l1_flags = [l1.scan_context(t).is_blocked for t in texts]
    if l2 is None:
        return [(f, False, 0.0) for f in l1_flags]
    l2_res = l2.scan_documents(texts, threshold=threshold)
    return [(f, r.is_injection, r.confidence) for f, r in zip(l1_flags, l2_res)]


def main() -> None:
    default_cfg, _, _ = load_all_configs()
    l2_cfg = default_cfg["defense"]["layers"]["layer2_injection_detector"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--context-threshold", type=float, default=None,
                        help="Override layer2 context_threshold (default: config).")
    parser.add_argument("--top-k", type=int, default=default_cfg["rag"]["top_k"],
                        help="Batch size for the latency probe (default: config rag.top_k).")
    parser.add_argument("--model-path", default=l2_cfg["model_path"],
                        help="L2 checkpoint to evaluate (default: config model_path).")
    parser.add_argument("--out", default="context_scan_eval.json",
                        help="Output file name under experiments/results/.")
    args = parser.parse_args()

    threshold = args.context_threshold
    if threshold is None:
        threshold = l2_cfg.get("context_threshold") or l2_cfg.get("threshold", 0.7)

    l1 = InputSanitizer()
    model_path = Path(args.model_path)
    l2 = InjectionDetector(model_path, threshold=threshold) if model_path.exists() else None
    if l2 is None:
        print(f"L2 checkpoint not found at {model_path} — reporting L1 only.")

    chunks = load_and_chunk()
    sources = [Path(c.metadata.get("source", "")).name for c in chunks]
    texts = [c.page_content for c in chunks]
    print(f"{len(chunks)} chunks from {len(set(sources))} documents | L2 threshold = {threshold}")

    # ── Clean chunks: false drops ──────────────────────────────────────────
    clean = scan(texts, l1, l2, threshold)
    by_split: dict[str, list[bool]] = defaultdict(list)
    false_drops = []
    for src, text, (f1, f2, conf) in zip(sources, texts, clean):
        split = "holdout" if src in DOC_HOLDOUT else "trained"
        by_split[split].append(f1 or f2)
        if f1 or f2:
            false_drops.append({"source": src, "split": split, "l1": f1, "l2": f2,
                                "l2_confidence": round(conf, 4), "preview": text[:80]})
    all_flags = [f1 or f2 for f1, f2, _ in clean]

    # ── Poisoned chunks: detection ─────────────────────────────────────────
    detection: dict[str, dict] = {}
    poisoned_conf: list[tuple[str, float]] = []   # (source, L2 confidence) over all positions
    for position in POSITIONS:
        poisoned = [poison(t, MARKER_PAYLOADS[i % len(MARKER_PAYLOADS)], position)
                    for i, t in enumerate(texts)]
        res = scan(poisoned, l1, l2, threshold)
        n = len(res)
        poisoned_conf += [(s, conf) for s, (_, _, conf) in zip(sources, res)]
        detection[position] = {
            "n": n,
            "l1_rate": sum(f1 for f1, _, _ in res) / n,
            "l2_rate": sum(f2 for _, f2, _ in res) / n,
            "combined_rate": sum(f1 or f2 for f1, f2, _ in res) / n,
            "holdout_combined_rate": _rate([f1 or f2 for s, (f1, f2, _) in zip(sources, res) if s in DOC_HOLDOUT]),
        }

    # ── L2 threshold sweep (same scores, no re-inference) ─────────────────
    sweep = []
    if l2 is not None:
        clean_conf = [(s, conf) for s, (_, _, conf) in zip(sources, clean)]
        for th in SWEEP_THRESHOLDS:
            row = {"threshold": th}
            for split, keep in (("trained", lambda s: s not in DOC_HOLDOUT), ("holdout", lambda s: s in DOC_HOLDOUT)):
                row[f"false_drop_{split}"] = _rate([c >= th for s, c in clean_conf if keep(s)])
                row[f"detection_{split}"] = _rate([c >= th for s, c in poisoned_conf if keep(s)])
            sweep.append(row)

    # ── Latency of one top-k batch ─────────────────────────────────────────
    latencies = []
    if l2 is not None:
        for i in range(0, len(texts) - args.top_k + 1, args.top_k):
            t0 = time.perf_counter()
            l2.scan_documents(texts[i:i + args.top_k], threshold=threshold)
            latencies.append((time.perf_counter() - t0) * 1000)

    output = {
        "experiment": "context_scan_eval",
        "l2_active": l2 is not None,
        "model_path": str(model_path),
        "threshold": threshold,
        "n_chunks": len(chunks),
        "doc_holdout": list(DOC_HOLDOUT),
        "false_drop_rate": _rate(all_flags),
        "false_drop_rate_trained_docs": _rate(by_split["trained"]),
        "false_drop_rate_holdout_docs": _rate(by_split["holdout"]),
        "false_drop_l1": sum(f1 for f1, _, _ in clean),
        "false_drop_l2": sum(f2 for _, f2, _ in clean),
        "detection_by_position": detection,
        "l2_scan_latency_ms_topk": {
            "top_k": args.top_k,
            "mean": statistics.mean(latencies) if latencies else None,
            "p95": sorted(latencies)[int(0.95 * (len(latencies) - 1))] if latencies else None,
        },
        "l2_threshold_sweep": sweep,
        "false_drops": false_drops,
    }

    print("\nClean chunks wrongly dropped")
    print(f"  overall        : {output['false_drop_rate']:.1%}  ({sum(all_flags)}/{len(all_flags)})")
    print(f"  trained docs   : {output['false_drop_rate_trained_docs']:.1%}")
    print(f"  held-out docs  : {output['false_drop_rate_holdout_docs']:.1%}")
    print("\nPoisoned chunks detected (Experience payloads, unseen in training)")
    for pos, d in detection.items():
        print(f"  {pos:<8} L1 {d['l1_rate']:.0%}  L2 {d['l2_rate']:.0%}  "
              f"combined {d['combined_rate']:.0%}  (held-out docs {d['holdout_combined_rate']:.0%})")
    if latencies:
        lat = output["l2_scan_latency_ms_topk"]
        print(f"\nL2 scan latency (top-{args.top_k}): mean {lat['mean']:.0f} ms, p95 {lat['p95']:.0f} ms")
    if sweep:
        print("\nL2 threshold sweep        false-drop (trained / held-out)   detection (trained / held-out)")
        for r in sweep:
            print(f"  th={r['threshold']:<5}               {r['false_drop_trained']:>6.1%} / {r['false_drop_holdout']:>6.1%}"
                  f"                  {r['detection_trained']:>6.1%} / {r['detection_holdout']:>6.1%}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / args.out
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved → {out_path}")


def _rate(flags: list[bool]) -> float | None:
    return sum(flags) / len(flags) if flags else None


if __name__ == "__main__":
    main()
