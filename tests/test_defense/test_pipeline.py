"""Construction tests for DefensePipeline.from_config (no LLM calls)."""
from pathlib import Path

from src.defense import DefensePipeline
from src.utils import load_config


class _FakeRetriever:
    def retrieve(self, query):
        return []

    def format_context(self, docs):
        return ""


def _cfg():
    return load_config("configs/default.yaml")


def test_from_config_builds_regardless_of_checkpoint_state():
    cfg = _cfg()
    p = DefensePipeline.from_config(cfg, _FakeRetriever(), "You are a QA assistant.")
    # Before scripts/train_detector.py has run there's no checkpoint -> L2 stays
    # None; once training produced models/detector/checkpoints/best_model/, L2
    # should load automatically. Either way the pipeline must be constructible.
    model_path = cfg["defense"]["layers"]["layer2_injection_detector"]["model_path"]
    checkpoint_exists = Path(model_path).exists()
    assert (p.l2 is not None) == checkpoint_exists
    assert p.enabled_layers == set(DefensePipeline.ALL_LAYERS)


def test_from_config_can_force_detector_off():
    p = DefensePipeline.from_config(_cfg(), _FakeRetriever(), "sp", detector_model_path=None)
    assert p.l2 is None


def test_from_config_applies_guarded_system_prompt():
    p = DefensePipeline.from_config(_cfg(), _FakeRetriever(), "You are a QA assistant.")
    assert "重要規則" in p.l3.system_prompt
    assert p._raw_system_prompt == "You are a QA assistant."


def test_from_config_loads_blacklist_file():
    p = DefensePipeline.from_config(_cfg(), _FakeRetriever(), "sp")
    # DEFAULT_BLACKLIST + configs/blacklist.txt entries.
    assert len(p.l1.compiled_patterns) > 10


def test_enabled_layers_subset_and_overrides():
    p = DefensePipeline.from_config(
        _cfg(), _FakeRetriever(), "sp",
        enabled_layers={"L3", "L5"}, detector_model_path=None, max_input_length=123,
    )
    assert p.enabled_layers == {"L3", "L5"}
    assert p.l1.max_length == 123
