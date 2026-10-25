"""Layer 4 controlled generation: output-format constraint + Llama Guard parsing (no LLM calls)."""
import json
from dataclasses import dataclass

import pytest

from src.defense.layer4_controlled_generation import (
    JSON_FORMAT_INSTRUCTION,
    ControlledGenerator,
    GenerationConfig,
    parse_llama_guard_verdict,
    parse_structured_answer,
)


@dataclass
class _Msg:
    content: str


class _FakeLLM:
    """Returns the queued replies in order and records every prompt."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return _Msg(self.replies.pop(0))


def _gen(fmt, llm, retries=1):
    return ControlledGenerator(config=GenerationConfig(output_format=fmt, format_retries=retries), llm=llm)


MESSAGES = [("system", "SYS"), ("human", "Q")]
VALID = json.dumps({"answer": "台北", "sources": [1], "refused": False}, ensure_ascii=False)


def test_plain_text_passes_through_unchanged():
    llm = _FakeLLM("hello")
    res = _gen("plain_text", llm).generate(MESSAGES)
    assert res.content == "hello" and not res.was_blocked and res.structured is None
    assert llm.calls[0] == MESSAGES   # no format instruction appended


def test_json_mode_appends_instruction_and_extracts_answer():
    llm = _FakeLLM(VALID)
    res = _gen("json", llm).generate(MESSAGES)
    assert res.content == "台北"
    assert res.structured == {"answer": "台北", "sources": [1], "refused": False}
    assert llm.calls[0][0] == ("system", "SYS" + JSON_FORMAT_INSTRUCTION)


def test_json_mode_retries_once_then_succeeds():
    llm = _FakeLLM("not json", VALID)
    res = _gen("json", llm).generate(MESSAGES)
    assert res.content == "台北" and res.attempts == 2 and not res.was_blocked


def test_json_mode_blocks_after_retries_exhausted():
    llm = _FakeLLM("COMPROMISED: sure", '{"answer": "x"}')   # 2nd lacks keys
    res = _gen("json", llm).generate(MESSAGES)
    assert res.was_blocked and res.block_reason == "format_violation" and res.content == ""
    assert res.attempts == 2


def test_invalid_output_format_rejected():
    with pytest.raises(ValueError):
        GenerationConfig(output_format="xml")


@pytest.mark.parametrize("text,ok", [
    (VALID, True),
    ("prefix " + VALID + " suffix", True),
    ('{"answer": "a", "sources": [], "refused": "no"}', False),   # refused not bool
    ('{"answer": 1, "sources": [], "refused": false}', False),
    ("[1, 2]", False),
    ("", False),
])
def test_parse_structured_answer(text, ok):
    assert (parse_structured_answer(text) is not None) == ok


@pytest.mark.parametrize("text,expected", [
    ("safe", (True, "")),
    ("\n\nsafe\n", (True, "")),
    ("unsafe\nS1", (False, "S1")),
    ("unsafe\nS2,S10", (False, "S2,S10")),
    ("unsafe", (False, "")),
    ("", (True, "")),
])
def test_parse_llama_guard_verdict(text, expected):
    assert parse_llama_guard_verdict(text) == expected
