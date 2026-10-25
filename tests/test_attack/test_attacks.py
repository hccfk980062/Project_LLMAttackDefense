"""Tests for attack generation modules."""
import pytest
from src.attack.attack_types import AttackType
from src.attack.direct_injection import generate_direct_samples
from src.attack.indirect_injection import poison_documents
from langchain_core.documents import Document


def test_direct_injection_generates_samples():
    samples = generate_direct_samples()
    assert len(samples) > 0
    for s in samples:
        assert s.attack_type == AttackType.DIRECT_INJECTION
        assert s.payload
        assert s.malicious_goal


def test_direct_injection_includes_chinese():
    samples = generate_direct_samples()
    zh_samples = [s for s in samples if s.language == "zh-tw"]
    assert len(zh_samples) > 0


def test_indirect_injection_poisons_correct_fraction():
    docs = [Document(page_content=f"doc {i}") for i in range(10)]
    poisoned, attack_samples = poison_documents(docs, poison_ratio=0.2)

    n_poisoned = sum(1 for d in poisoned if d.metadata.get("poisoned"))
    assert n_poisoned == 2
    assert len(attack_samples) == 2


def test_indirect_injection_preserves_original_docs():
    docs = [Document(page_content=f"doc {i}") for i in range(10)]
    poisoned, _ = poison_documents(docs, poison_ratio=0.1)

    assert len(poisoned) == len(docs)


def test_attack_sample_to_dict():
    samples = generate_direct_samples()
    d = samples[0].to_dict()
    assert "id" in d
    assert "attack_type" in d
    assert "payload" in d
    assert "malicious_goal" in d
