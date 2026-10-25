from pydantic import BaseModel
from typing import List, Optional


class LayerTraceSchema(BaseModel):
    layer: str
    passed: bool
    detail: dict
    latency_ms: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    is_blocked: bool
    block_layer: Optional[str] = None
    block_reason: Optional[str] = None
    traces: List[LayerTraceSchema] = []
    total_latency_ms: float
    # Sources of the retrieved documents (undefended path) / of documents the
    # L1/L2 context scan dropped (defended path) — makes indirect injection visible.
    sources: List[str] = []
    dropped_docs: List[dict] = []


class AttackResponse(BaseModel):
    payload: str
    raw_response: str
    is_success: bool
    was_blocked: bool
    block_layer: Optional[str] = None
    latency_ms: float
    judge_reason: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    defense_enabled: bool
    l2_detector_active: bool = False


class ABVariantResult(BaseModel):
    name: str
    enabled_layers: Optional[List[str]] = None
    answer: str
    is_blocked: bool
    block_layer: Optional[str] = None
    block_reason: Optional[str] = None
    traces: List[LayerTraceSchema] = []
    total_latency_ms: float
    # Relative to the first variant without enabled_layers (the baseline), or
    # to the first variant when there is no baseline arm.
    latency_delta_ms: float = 0.0


class ABResponse(BaseModel):
    query: str
    experiment_id: Optional[str] = None
    reference_variant: str
    results: List[ABVariantResult]


class ChunkVerdict(BaseModel):
    index: int
    preview: str
    flagged: bool
    flagged_by: List[str] = []
    reasons: List[str] = []


class UploadResponse(BaseModel):
    filename: str
    saved_path: str
    n_chunks: int
    n_added: int
    n_rejected: int
    scanned: bool
    chunks: List[ChunkVerdict] = []


class DocumentSource(BaseModel):
    source: str
    n_chunks: int


class DocumentListResponse(BaseModel):
    total_chunks: int
    documents: List[DocumentSource]
