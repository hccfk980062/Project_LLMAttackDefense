"""Tests for Evaluator's detector_f1 wiring (src.evaluation.evaluator)."""
import json
from dataclasses import dataclass, field

import pytest

from src.attack.attack_types import AttackSample, AttackType
from src.evaluation.evaluator import Evaluator


@dataclass
class _FakeDetectionResult:
    is_injection: bool


class _FakeDetector:
    """Duck-typed stand-in for src.defense.layer2_injection_detector.InjectionDetector."""

    def __init__(self, predict_fn):
        self.predict_fn = predict_fn
        self.calls: list[list[str]] = []

    def detect_batch(self, texts):
        self.calls.append(list(texts))
        return [_FakeDetectionResult(self.predict_fn(t)) for t in texts]


@dataclass
class _FakeDefenseResult:
    final_answer: str
    is_blocked: bool
    block_layer: str = ""
    traces: list = field(default_factory=list)


def _always_blocked_defended_fn(q: str) -> _FakeDefenseResult:
    # Always blocked -> AttackRunner short-circuits before calling the (real,
    # network-hitting) LLM judge, keeping this test fully offline.
    return _FakeDefenseResult(final_answer="", is_blocked=True, block_layer="L1", traces=[])


def _always_blocked_baseline_fn(q: str) -> str:
    return ""  # AttackRunner's plain-string blocked heuristic: response == ""


@pytest.fixture
def samples():
    attack_samples = [
        AttackSample(id="a1", attack_type=AttackType.DIRECT_INJECTION, payload="malicious one", malicious_goal="g"),
        AttackSample(id="a2", attack_type=AttackType.JAILBREAK, payload="malicious two", malicious_goal="g"),
    ]
    benign_samples = [
        AttackSample(id="b1", attack_type=AttackType.DIRECT_INJECTION, payload="malicious-looking benign", malicious_goal="N/A"),
        AttackSample(id="b2", attack_type=AttackType.DIRECT_INJECTION, payload="totally normal question", malicious_goal="N/A"),
    ]
    return attack_samples, benign_samples


def test_detector_f1_is_zero_without_a_detector(tmp_path, samples):
    attack_samples, benign_samples = samples
    evaluator = Evaluator(
        defended_fn=_always_blocked_defended_fn,
        undefended_fn=_always_blocked_baseline_fn,
        output_dir=str(tmp_path),
    )
    reports = evaluator.evaluate(attack_samples, benign_samples, experiment_name="no_detector")
    assert reports["defended"].detector_f1 == 0.0
    assert reports["baseline"].detector_f1 == 0.0


def test_detector_f1_computed_and_shared_across_reports(tmp_path, samples):
    attack_samples, benign_samples = samples
    detector = _FakeDetector(predict_fn=lambda text: "malicious" in text)

    evaluator = Evaluator(
        defended_fn=_always_blocked_defended_fn,
        undefended_fn=_always_blocked_baseline_fn,
        output_dir=str(tmp_path),
        detector=detector,
    )
    reports = evaluator.evaluate(attack_samples, benign_samples, experiment_name="with_detector")

    # labels: [1, 1, 0, 0] (2 attacks then 2 benign); preds via predict_fn:
    # both attacks match "malicious" -> 1, 1; b1 also contains "malicious" (FP) -> 1;
    # b2 doesn't -> 0. TP=2 FP=1 FN=0 -> F1 = 0.8.
    assert reports["defended"].detector_f1 == pytest.approx(0.8)
    # Same detector, same samples -> identical F1 regardless of which pipeline
    # variant is being scored (detector_f1 is a property of L2 alone).
    assert reports["baseline"].detector_f1 == pytest.approx(0.8)

    # detect_batch was called once with attack payloads followed by benign ones.
    assert detector.calls == [["malicious one", "malicious two",
                               "malicious-looking benign", "totally normal question"]]


def test_evaluate_writes_summary_with_blocked_by_layer(tmp_path, samples):
    attack_samples, benign_samples = samples
    evaluator = Evaluator(
        defended_fn=_always_blocked_defended_fn,
        undefended_fn=_always_blocked_baseline_fn,
        output_dir=str(tmp_path),
    )
    evaluator.evaluate(attack_samples, benign_samples, experiment_name="summary_check")

    summary = json.loads((tmp_path / "summary_check_summary.json").read_text(encoding="utf-8"))
    assert summary["defended"]["blocked_by_layer"] == {"L1": 2}
    assert summary["defended"]["benign_blocked_by_layer"] == {"L1": 2}
