from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User question")
    defense_enabled: bool = Field(True, description="Enable defense pipeline")
    experiment_id: Optional[str] = Field(None, description="Tag for experiment tracking")


class AttackRequest(BaseModel):
    attack_type: str = Field(..., description="Attack type (see AttackType enum)")
    payload: str = Field(..., description="Malicious payload to inject")
    malicious_goal: str = Field(..., description="Intended goal of the attack")
    language: str = Field("zh-tw", description="Language of the attack")


class ABVariant(BaseModel):
    name: str = Field(..., description="Label for this arm of the comparison")
    # None -> undefended baseline RAG (no DefensePipeline at all).
    enabled_layers: Optional[List[str]] = Field(
        None, description="Subset of L1..L5; null = undefended baseline RAG",
    )
    overrides: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra DefensePipeline kwargs, e.g. {\"output_format\": \"json\", \"use_llm_audit\": false}",
    )


def _default_variants() -> List[ABVariant]:
    return [
        ABVariant(name="baseline", enabled_layers=None),
        ABVariant(name="defended", enabled_layers=["L1", "L2", "L3", "L4", "L5"]),
    ]


class ABRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    variants: List[ABVariant] = Field(default_factory=_default_variants, min_length=1, max_length=8)
    experiment_id: Optional[str] = None
