"""Tests for Layer 3: Prompt Segmentation."""
import pytest
from src.defense.layer3_prompt_segmentation import PromptSegmenter, SYSTEM_TAG, USER_TAG, CONTEXT_TAG


@pytest.fixture
def segmenter():
    return PromptSegmenter(system_prompt="你是一個問答助理。")


def test_segment_produces_correct_structure(segmenter):
    result = segmenter.segment("什麼是 AI？", "AI 是人工智慧的縮寫。")
    assert result.system_block == "你是一個問答助理。"
    assert result.user_block == "什麼是 AI？"
    assert result.context_block == "AI 是人工智慧的縮寫。"


def test_to_messages_returns_two_messages(segmenter):
    result = segmenter.segment("Q", "ctx")
    messages = result.to_messages()
    assert len(messages) == 2
    assert messages[0][0] == "system"
    assert messages[1][0] == "human"


def test_to_string_contains_all_tags(segmenter):
    result = segmenter.segment("Q", "ctx")
    s = result.to_string()
    assert SYSTEM_TAG in s
    assert USER_TAG in s
    assert CONTEXT_TAG in s


def test_guarded_system_prompt_contains_boundary_rules(segmenter):
    guarded = segmenter.build_guarded_system_prompt()
    assert "CONTEXT" in guarded
    assert "SYSTEM" in guarded
