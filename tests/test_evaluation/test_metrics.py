"""Tests for evaluation metrics."""
import pytest
from src.attack.attack_types import AttackResult, AttackSample, AttackType
from src.evaluation.metrics import compute_metrics


def _make_result(
    is_success: bool, was_blocked: bool, attack_type: AttackType,
    latency: float = 100.0, block_layer: str = "",
) -> AttackResult:
    sample = AttackSample(
        id="test",
        attack_type=attack_type,
        payload="test",
        malicious_goal="test",
    )
    return AttackResult(
        sample=sample,
        raw_response="test response",
        is_success=is_success,
        was_blocked=was_blocked,
        latency_ms=latency,
        block_layer=block_layer,
    )


def test_asr_all_success():
    results = [_make_result(True, False, AttackType.DIRECT_INJECTION) for _ in range(10)]
    report = compute_metrics(results, [])
    assert report.asr == pytest.approx(1.0)


def test_asr_none_success():
    results = [_make_result(False, True, AttackType.DIRECT_INJECTION) for _ in range(10)]
    report = compute_metrics(results, [])
    assert report.asr == pytest.approx(0.0)


def test_fpr_calculation():
    attack_results = [_make_result(False, True, AttackType.DIRECT_INJECTION)]
    benign_results = [
        _make_result(False, True, AttackType.DIRECT_INJECTION),   # falsely blocked
        _make_result(False, False, AttackType.DIRECT_INJECTION),  # correctly passed
    ]
    report = compute_metrics(attack_results, benign_results)
    assert report.fpr == pytest.approx(0.5)


def test_asr_by_type():
    results = [
        _make_result(True, False, AttackType.DIRECT_INJECTION),
        _make_result(False, True, AttackType.INDIRECT_INJECTION),
    ]
    report = compute_metrics(results, [])
    assert "direct_injection" in report.asr_by_type
    assert report.asr_by_type["direct_injection"] == pytest.approx(1.0)
    assert report.asr_by_type["indirect_injection"] == pytest.approx(0.0)


def test_latency_stats():
    results = [_make_result(False, False, AttackType.DIRECT_INJECTION, latency=l) for l in [100, 200, 300]]
    report = compute_metrics(results, [])
    assert report.mean_latency_ms == pytest.approx(200.0)


def test_blocked_by_layer_breakdown():
    attack_results = [
        _make_result(False, True, AttackType.DIRECT_INJECTION, block_layer="L1"),
        _make_result(False, True, AttackType.DIRECT_INJECTION, block_layer="L1"),
        _make_result(False, True, AttackType.JAILBREAK, block_layer="L2"),
        _make_result(True, False, AttackType.DIRECT_INJECTION),  # not blocked
    ]
    benign_results = [
        _make_result(False, True, AttackType.DIRECT_INJECTION, block_layer="L5"),
    ]
    report = compute_metrics(attack_results, benign_results)
    assert report.blocked_by_layer == {"L1": 2, "L2": 1}
    assert report.benign_blocked_by_layer == {"L5": 1}


def test_blocked_by_layer_empty_when_no_layer_info():
    # Plain-string target_fn's never populate block_layer, even when blocked.
    results = [_make_result(False, True, AttackType.DIRECT_INJECTION)]
    report = compute_metrics(results, [])
    assert report.blocked_by_layer == {}


def test_detector_f1_stays_zero_without_labels():
    report = compute_metrics([], [])
    assert report.detector_f1 == 0.0


def test_detector_f1_computed_when_labels_and_preds_given():
    report = compute_metrics([], [], detector_labels=[1, 1, 0, 0], detector_preds=[1, 0, 0, 0])
    assert report.detector_f1 == pytest.approx(2 / 3)
