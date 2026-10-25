"""Unified attack-sample generation: one entry point for every attack type.

``generate_samples`` dispatches to the per-type generators and, optionally,
folds in the on-disk datasets (self-built zh-TW set + external datasets).
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, List

from .attack_types import AttackSample, AttackType
from .dataset_loader import (
    dedupe_samples,
    load_chinese_attack_set,
    load_external_attack_datasets,
)
from .direct_injection import generate_direct_samples
from .indirect_injection import POISON_PAYLOADS
from .jailbreak import generate_jailbreak_samples
from .multilingual_obfuscation import generate_multilingual_obfuscation_samples
from .prompt_leaking import generate_prompt_leaking_samples


def _indirect_payload_samples() -> List[AttackSample]:
    """Indirect injection has no user-side payload; expose the poison strings
    themselves as samples so the L2 detector can learn them."""
    return [
        AttackSample(
            id=f"indirect_payload_{i:04d}",
            attack_type=AttackType.INDIRECT_INJECTION,
            payload=p.strip(),
            malicious_goal="Hijack model output via poisoned knowledge base",
            language="mixed",
            metadata={"technique": "kb_poison_payload"},
        )
        for i, p in enumerate(POISON_PAYLOADS)
    ]


SAMPLE_GENERATORS: dict[AttackType, Callable[[], List[AttackSample]]] = {
    AttackType.DIRECT_INJECTION: generate_direct_samples,
    AttackType.INDIRECT_INJECTION: _indirect_payload_samples,
    AttackType.JAILBREAK: generate_jailbreak_samples,
    AttackType.PROMPT_LEAKING: generate_prompt_leaking_samples,
    AttackType.MULTILINGUAL_OBFUSCATION: generate_multilingual_obfuscation_samples,
}


def _normalize_types(types: Iterable[str | AttackType] | None) -> list[AttackType]:
    if types is None:
        return list(SAMPLE_GENERATORS)
    out: list[AttackType] = []
    for t in types:
        out.append(t if isinstance(t, AttackType) else AttackType(str(t)))
    return out


def generate_samples(
    types: Iterable[str | AttackType] | None = None,
    *,
    attack_config: dict[str, Any] | None = None,
    include_chinese_set: bool = False,
    include_external: bool = False,
    dedupe: bool = True,
) -> List[AttackSample]:
    """
    Build attack samples for the requested ``types`` (default: all).

    Args:
        types: subset of AttackType values / names, or None for every type.
        attack_config: parsed configs/attack.yaml, used for per-type templates
            (multilingual languages) and the external dataset list.
        include_chinese_set: also load data/chinese_attack_set/*.json.
        include_external: also load datasets listed in attack_config.
        dedupe: drop duplicate payloads.
    """
    wanted = _normalize_types(types)
    atk = ((attack_config or {}).get("attack") or attack_config or {})
    ml_cfg = atk.get("multilingual_obfuscation", {}) or {}
    jb_cfg = atk.get("jailbreak", {}) or {}
    pl_cfg = atk.get("prompt_leaking", {}) or {}

    per_type_kwargs: dict[AttackType, dict[str, Any]] = {
        AttackType.MULTILINGUAL_OBFUSCATION: (
            {"languages": ml_cfg["languages"]} if ml_cfg.get("languages") else {}
        ),
        AttackType.JAILBREAK: {
            k: jb_cfg[k]
            for k in ("templates_zh", "templates_en", "goals", "include_english")
            if k in jb_cfg
        },
        AttackType.PROMPT_LEAKING: {
            k: pl_cfg[k]
            for k in ("payloads_zh", "payloads_en", "include_english")
            if k in pl_cfg
        },
    }

    samples: List[AttackSample] = []
    for atype in wanted:
        gen = SAMPLE_GENERATORS[atype]
        samples.extend(gen(**per_type_kwargs.get(atype, {})))  # type: ignore[call-arg]

    if include_chinese_set:
        cn = load_chinese_attack_set()
        samples.extend(s for s in cn if s.attack_type in wanted)
    if include_external:
        ext = load_external_attack_datasets(attack_config)
        samples.extend(s for s in ext if s.attack_type in wanted)

    return dedupe_samples(samples) if dedupe else samples
