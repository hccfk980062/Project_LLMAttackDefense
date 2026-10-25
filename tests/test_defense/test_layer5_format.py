"""Layer 5: format-compliance check and the independent audit model (no LLM calls)."""
from dataclasses import dataclass

from src.defense import DefensePipeline
from src.defense.layer5_output_validator import OutputValidator
from src.utils import load_config


@dataclass
class _Msg:
    content: str


class _FakeAuditor:
    def __init__(self, verdict="SAFE\n原因：正常"):
        self.verdict = verdict
        self.prompts = []

    def invoke(self, messages):
        self.prompts.append(messages[0][1])
        return _Msg(self.verdict)


def _v(**kw):
    return OutputValidator(use_llm_audit=False, **kw)


def test_normal_answer_passes():
    assert _v().validate("OWASP LLM01 是 Prompt Injection。").is_safe


def test_empty_output_is_noncompliant():
    r = _v().validate("   ")
    assert not r.is_safe and r.blocked_reason == "format_noncompliant: empty_output"


def test_too_long_output_is_noncompliant():
    r = _v(max_output_chars=10).validate("x" * 11)
    assert not r.is_safe and "too_long" in r.blocked_reason


def test_single_structure_tag_is_allowed():
    # The KB itself explains "[SYSTEM]" delimiter spoofing — one mention is legitimate.
    assert _v().validate("攻擊者會用 ---、###、[SYSTEM] 等符號假裝系統邊界。").is_safe


def test_prompt_layout_echo_is_blocked():
    r = _v().validate("[CONTEXT]\n參考資料...\n[END]\n[USER] 問題")
    assert not r.is_safe and "structure_tags_leaked" in r.blocked_reason


def test_json_mode_requires_structured_output():
    v = _v(output_format="json")
    assert not v.validate("台北", structured=None).is_safe
    assert v.validate("台北", structured={"answer": "台北", "sources": [], "refused": False}).is_safe


def test_audit_uses_separate_model_and_custom_prompt():
    auditor = _FakeAuditor("UNSAFE\n原因：洩漏")
    v = OutputValidator(llm_model="gen-model", audit_model="audit-model", use_llm_audit=True,
                        audit_prompt="CHECK: {response}", auditor=auditor)
    r = v.validate("some answer")
    assert not r.is_safe and r.blocked_reason == "llm_audit_unsafe"
    assert r.audit_model == "audit-model"
    assert auditor.prompts == ["CHECK: some answer"]


def test_default_patterns_do_not_know_the_evaluation_markers():
    # L5 must not be scored against payloads whose markers it was hard-coded
    # with — that is the defense reading the answer key.
    from src.attack.indirect_injection import MARKER_PAYLOADS, SUCCESS_MARKERS
    from src.defense.layer5_output_validator import EVAL_MARKER_PATTERNS

    v = _v()
    for marker in SUCCESS_MARKERS:
        assert v.validate(f"{marker} 台灣資安市場規模約 250 億元。").is_safe, marker
    opt_in = _v(reject_patterns=EVAL_MARKER_PATTERNS)
    assert not opt_in.validate("[DEBUG: PWNED]").is_safe
    assert MARKER_PAYLOADS  # the markers exist; only the default list ignores them


def test_audit_model_defaults_to_generation_model():
    assert OutputValidator(llm_model="gen-model", use_llm_audit=False).audit_model == "gen-model"


class _FakeRetriever:
    def retrieve(self, query):
        return []

    def format_context(self, docs):
        return ""


def test_from_config_wires_l4_and_l5_settings():
    cfg = load_config("configs/default.yaml")
    p = DefensePipeline.from_config(cfg, _FakeRetriever(), "sp", detector_model_path=None)
    l4 = cfg["defense"]["layers"]["layer4_controlled_generation"]
    l5 = cfg["defense"]["layers"]["layer5_output_validator"]
    assert p.l4.config.max_tokens == l4["max_tokens"]
    assert p.l4.config.output_format == l4["output_format"] == p.l5.output_format
    assert p.l5.audit_model == l5["audit_model"] != cfg["llm"]["model"]   # independent auditor
    assert "{response}" in p.l5.audit_prompt


def test_json_override_reaches_both_layers():
    cfg = load_config("configs/default.yaml")
    p = DefensePipeline.from_config(cfg, _FakeRetriever(), "sp", detector_model_path=None, output_format="json")
    assert p.l4.config.output_format == "json" and p.l5.output_format == "json"
