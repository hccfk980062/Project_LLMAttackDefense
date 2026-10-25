"""Regression test: Experience_4's --seed sampling must be reproducible.

A plain `{set}` comprehension dedupes but its iteration order depends on
Python's per-process hash seed, which silently defeated the seeded
`random.Random(seed).shuffle()` downstream of it — two `--seed 42` runs, in
separate processes, sampled different queries (caught by comparing two real
Experience 4 runs whose "random" 60-query sample differed).
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO / "scripts" / "Experience_4"


def _sample_in_fresh_process(benign_path: Path, n: int, seed: int) -> list[str]:
    code = (
        f"import sys; sys.path.insert(0, {str(SCRIPT_DIR)!r}); "
        f"from pathlib import Path; import run_fpr_test as m; "
        f"m.BENIGN_DATASET_PATH = Path({str(benign_path)!r}); "
        f"import json; print(json.dumps([t for t, _ in m.load_benign_sample({n}, {seed})]))"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_seeded_sample_is_reproducible_across_processes(tmp_path):
    benign_path = tmp_path / "benign.json"
    rows = [{"payload": f"query {i}", "text": f"query {i}"} for i in range(30)]
    benign_path.write_text(json.dumps(rows), encoding="utf-8")

    # PYTHONHASHSEED is randomized per process by default, so this genuinely
    # exercises cross-process determinism (not just re-shuffling in-process).
    first = _sample_in_fresh_process(benign_path, n=15, seed=42)
    second = _sample_in_fresh_process(benign_path, n=15, seed=42)
    assert first == second


def test_seeded_sample_dedupes_and_respects_n(tmp_path):
    benign_path = tmp_path / "benign.json"
    rows = [{"payload": "dup"}, {"payload": "dup"}, {"payload": "unique-a"}, {"payload": "unique-b"}]
    benign_path.write_text(json.dumps(rows), encoding="utf-8")

    sample = _sample_in_fresh_process(benign_path, n=10, seed=1)
    assert len(sample) == len(set(sample)) == 3  # deduped: "dup" appears once
