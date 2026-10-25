"""Tests for the attack-sample generators (jailbreak / prompt_leaking / multilingual)."""
import pytest

from src.attack import (
    AttackType,
    generate_jailbreak_samples,
    generate_multilingual_obfuscation_samples,
    generate_prompt_leaking_samples,
    generate_samples,
)

GEN_CASES = [
    (generate_jailbreak_samples, AttackType.JAILBREAK),
    (generate_prompt_leaking_samples, AttackType.PROMPT_LEAKING),
    (generate_multilingual_obfuscation_samples, AttackType.MULTILINGUAL_OBFUSCATION),
]


@pytest.mark.parametrize("gen_fn,expected_type", GEN_CASES)
def test_generator_produces_valid_samples(gen_fn, expected_type):
    samples = gen_fn()
    assert len(samples) > 0
    assert all(s.attack_type is expected_type for s in samples)
    assert all(s.payload.strip() for s in samples)
    assert all(s.malicious_goal for s in samples)
    ids = [s.id for s in samples]
    assert len(ids) == len(set(ids)), "sample ids must be unique"


def test_generate_samples_all_types_are_covered():
    samples = generate_samples()
    seen = {s.attack_type for s in samples}
    assert seen == set(AttackType)


def test_generate_samples_subset_and_dedupe():
    samples = generate_samples(["direct_injection", "jailbreak"])
    assert {s.attack_type for s in samples} == {
        AttackType.DIRECT_INJECTION,
        AttackType.JAILBREAK,
    }
    payloads = [s.payload for s in samples]
    assert len(payloads) == len(set(payloads)), "generate_samples should dedupe payloads"


def test_multilingual_respects_language_config():
    samples = generate_multilingual_obfuscation_samples(languages=["ja"])
    # No en/zh-cn language-switch payloads when only ja is requested.
    assert not any(s.language == "mixed:en" and s.metadata.get("technique") == "language_switch"
                   for s in samples)
    assert any("ja" in s.language for s in samples)
