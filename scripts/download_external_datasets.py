"""Download AdvBench and JailbreakBench (ProjectProposal §3.3) and normalise
them to this project's attack-set schema.

  AdvBench        walledai/AdvBench            (Zou et al., 2023)   520 prompts
  JailbreakBench  JailbreakBench/JBB-Behaviors (Chao et al., 2024)  100 harmful behaviours

Output paths come from configs/attack.yaml (attack.datasets.<name>.path), so
`load_external_attack_datasets()` / `run_evaluation.py --include-external`
pick them up without further changes. Both sets are plain harmful-behaviour
requests, so every record is typed ``jailbreak`` (the attack is getting the
assistant to comply with a request its policy forbids).

    python scripts/download_external_datasets.py            # both
    python scripts/download_external_datasets.py --only advbench
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import get_logger, load_config

logger = get_logger("download_external_datasets")


def normalize_advbench(rows) -> list[dict]:
    out = []
    for i, r in enumerate(rows):
        prompt = (r.get("prompt") or r.get("goal") or "").strip()
        if not prompt:
            continue
        out.append({
            "id": f"advbench_{i:04d}",
            "attack_type": "jailbreak",
            "language": "en",
            "payload": prompt,
            "malicious_goal": f"Get the assistant to comply with: {prompt}",
            "category": "harmful_behavior",
            "notes": f"target: {r.get('target', '')}".strip(),
        })
    return out


def normalize_jbb(rows) -> list[dict]:
    out = []
    for i, r in enumerate(rows):
        goal = (r.get("Goal") or r.get("goal") or "").strip()
        if not goal:
            continue
        out.append({
            "id": f"jbb_{i:04d}",
            "attack_type": "jailbreak",
            "language": "en",
            "payload": goal,
            "malicious_goal": f"Get the assistant to comply with: {goal}",
            "category": r.get("Category") or r.get("category") or "unknown",
            "notes": f"behavior: {r.get('Behavior', '')}; source: {r.get('Source', '')}",
        })
    return out


# walledai/AdvBench is gated on the Hub (needs login + accepted terms); the
# original authors' CSV (columns goal,target) is the same 520 behaviours.
ADVBENCH_CSV_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
)


def fetch_advbench_csv() -> list[dict]:
    import csv
    import io
    import urllib.request

    with urllib.request.urlopen(ADVBENCH_CSV_URL, timeout=60) as resp:
        text = resp.read().decode("utf-8")
    return [{"prompt": r["goal"], "target": r["target"]} for r in csv.DictReader(io.StringIO(text))]


FALLBACKS = {"advbench": fetch_advbench_csv}

SOURCES = {
    # name in configs/attack.yaml -> (HF dataset, config, split, normaliser)
    "advbench": ("walledai/AdvBench", None, "train", normalize_advbench),
    "jailbreakbench": ("JailbreakBench/JBB-Behaviors", "behaviors", "harmful", normalize_jbb),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--attack-config", default="configs/attack.yaml")
    parser.add_argument("--only", choices=sorted(SOURCES), nargs="*")
    args = parser.parse_args()

    from datasets import load_dataset

    datasets_cfg = load_config(args.attack_config)["attack"]["datasets"]
    for name in args.only or SOURCES:
        hf_name, hf_config, split, normalise = SOURCES[name]
        out_path = Path(datasets_cfg[name]["path"])
        logger.info(f"Downloading {hf_name} ({split}) ...")
        try:
            ds = load_dataset(hf_name, hf_config, split=split)
        except Exception as e:
            if name not in FALLBACKS:
                logger.error(f"Failed to download {hf_name}: {e}")
                logger.error("Some datasets require accepting terms on huggingface.co and `huggingface-cli login`.")
                continue
            logger.warning(f"{hf_name} unavailable ({str(e).splitlines()[0][:80]}); using fallback source.")
            try:
                ds = FALLBACKS[name]()
            except Exception as e2:
                logger.error(f"Fallback for {name} failed too: {e2}")
                continue
        records = normalise(ds)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"  {len(records)} records -> {out_path}")


if __name__ == "__main__":
    main()
