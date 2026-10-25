"""Attack type definitions and sample schema."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AttackType(str, Enum):
    DIRECT_INJECTION = "direct_injection"
    INDIRECT_INJECTION = "indirect_injection"
    MULTILINGUAL_OBFUSCATION = "multilingual_obfuscation"
    JAILBREAK = "jailbreak"
    PROMPT_LEAKING = "prompt_leaking"


@dataclass
class AttackSample:
    id: str
    attack_type: AttackType
    payload: str                    # malicious user input
    malicious_goal: str             # what the attack aims to achieve
    language: str = "zh-tw"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "attack_type": self.attack_type.value,
            "payload": self.payload,
            "malicious_goal": self.malicious_goal,
            "language": self.language,
            "metadata": self.metadata,
        }


@dataclass
class AttackResult:
    sample: AttackSample
    raw_response: str
    is_success: bool                # did the attack bypass the defense?
    was_blocked: bool               # was the request blocked by any layer?
    block_layer: str = ""
    latency_ms: float = 0.0
    judge_reason: str = ""
    # Per-layer latency (ms), populated when target_fn returns a DefensePipeline
    # DefenseResult (has a `.traces` list) instead of a plain string. Empty for
    # undefended target_fn's, which have no layers to report on.
    layer_latencies: dict[str, float] = field(default_factory=dict)
