"""Integration test for scripts/build_detector_dataset.py (offline, no API)."""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "build_detector_dataset.py"

REQUIRED_KEYS = {"id", "group_id", "payload", "text", "label", "attack_type", "language", "source"}


def test_build_dataset_end_to_end(tmp_path):
    out = tmp_path / "sets"
    result = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--out-dir", str(out), "--target-per-class", "120", "--seed", "1"],
        cwd=REPO, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    for name in ("benign.json", "all_attacks.json", "detector_dataset.json",
                 "ood_eval.json", "stats.json"):
        assert (out / name).exists(), f"{name} not written"

    benign = json.loads((out / "benign.json").read_text(encoding="utf-8"))
    attacks = json.loads((out / "all_attacks.json").read_text(encoding="utf-8"))
    merged = json.loads((out / "detector_dataset.json").read_text(encoding="utf-8"))
    ood = json.loads((out / "ood_eval.json").read_text(encoding="utf-8"))
    stats = json.loads((out / "stats.json").read_text(encoding="utf-8"))

    # Balanced classes.
    assert len(benign) == len(attacks)
    assert len(merged) == len(benign) + len(attacks)
    assert all(r["label"] == 0 for r in benign)
    assert all(r["label"] == 1 for r in attacks)

    # Schema.
    for r in merged[:50]:
        assert REQUIRED_KEYS <= set(r)
    for r in ood:
        assert REQUIRED_KEYS <= set(r)

    # No duplicate texts within a class.
    assert len({r["text"] for r in benign}) == len(benign)
    assert len({r["text"] for r in attacks}) == len(attacks)

    # Every attack type represented, chinese set folded in.
    assert set(stats["attacks_by_type"]) >= {"direct_injection", "jailbreak", "prompt_leaking"}
    assert any("chinese" in s for s in stats["attacks_by_source"])

    # group_id is set on every record, and augmentation actually produces
    # multiple rows per group (otherwise the leakage fix has nothing to do).
    assert all(r.get("group_id") for r in merged)
    assert stats["n_groups"] < stats["n_total"]

    # OOD set has both classes, no overlap with the augmented/generated pool,
    # and is untouched by augmentation (its own group_id == its own id).
    assert {r["label"] for r in ood} == {0, 1}
    assert all(r["source"] == "ood_handwritten" for r in ood)
    assert all(r["group_id"] == r["id"] for r in ood)
    assert not ({r["text"] for r in ood} & {r["text"] for r in merged})


def test_document_windows_and_payload_holdout(tmp_path):
    sys.path.insert(0, str(REPO))
    from src.attack.indirect_injection import MARKER_PAYLOADS, SUCCESS_MARKERS

    out = tmp_path / "sets"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--out-dir", str(out), "--target-per-class", "120", "--seed", "3"],
        cwd=REPO, capture_output=True, text=True, check=True,
    )
    merged = json.loads((out / "detector_dataset.json").read_text(encoding="utf-8"))
    ood_doc = json.loads((out / "ood_doc_eval.json").read_text(encoding="utf-8"))
    stats = json.loads((out / "stats.json").read_text(encoding="utf-8"))
    holdout = set(stats["doc_holdout"])

    # Knowledge-base windows are present in both classes, one group per document.
    doc_rows = [r for r in merged if r["source"].startswith("doc:")]
    assert {r["label"] for r in doc_rows} == {0, 1}
    for r in doc_rows:
        assert r["group_id"] == "doc::" + r["source"][len("doc:"):]

    # Held-out documents never reach training.
    assert not {r["source"][len("doc:"):] for r in doc_rows} & holdout

    # Experience payloads / success markers never reach training — otherwise
    # indirect-injection ASR would measure memorisation.
    for r in merged:
        for p in MARKER_PAYLOADS:
            assert p.strip() not in r["text"]
        for m in SUCCESS_MARKERS:
            assert m not in r["text"], (m, r["text"][:80])

    # The document OOD set is built only from held-out docs, both classes.
    assert ood_doc and {r["label"] for r in ood_doc} == {0, 1}
    assert {r["source"][len("ood_doc:"):] for r in ood_doc} <= holdout
    assert not ({r["text"] for r in ood_doc} & {r["text"] for r in merged})


def _norm(text: str) -> str:
    """Drop whitespace/punctuation so augmentation variants compare equal."""
    return "".join(ch for ch in text.lower() if ch.isalnum())


def test_heldout_benign_queries_never_reach_training():
    # FPR (Experience 4 / run_evaluation) and the extended trigger queries are
    # only meaningful if L2 has never seen these exact questions.
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "scripts"))
    from build_detector_dataset import OOD_BENIGN, build_benign_bank
    from src.evaluation.benign import heldout_benign_queries

    heldout = heldout_benign_queries()
    texts = [q for q, _ in heldout]
    assert len(texts) == len(set(texts)) >= 80
    assert {c for _, c in heldout} == {"kb", "general", "edge", "multilingual"}

    training = {_norm(q) for q, _ in build_benign_bank()} | {_norm(q) for q, _ in OOD_BENIGN}
    leaked = [q for q in texts if _norm(q) in training]
    assert not leaked, leaked


def test_group_ids_link_augmented_variants_to_their_base_sample(tmp_path):
    out = tmp_path / "sets"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--out-dir", str(out), "--target-per-class", "150", "--seed", "2"],
        cwd=REPO, capture_output=True, text=True, check=True,
    )
    attacks = json.loads((out / "all_attacks.json").read_text(encoding="utf-8"))
    by_group: dict[str, list[dict]] = {}
    for r in attacks:
        by_group.setdefault(r["group_id"], []).append(r)

    multi = [g for g in by_group.values() if len(g) > 1]
    assert multi, "expected at least one group with augmented variants"
    for group in multi:
        # All variants of one base sample must agree on label/attack_type — the
        # thing group-aware splitting relies on.
        assert len({r["label"] for r in group}) == 1
        assert len({r["attack_type"] for r in group}) == 1
