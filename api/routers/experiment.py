"""A/B experiment endpoint: run one query through several system variants
(undefended baseline, full pipeline, layer subsets, L4/L5 overrides) side by side."""
import json

from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from api.schemas.request import ABRequest, ABVariant
from api.schemas.response import ABResponse, ABVariantResult, LayerTraceSchema

router = APIRouter(prefix="/experiment", tags=["Experiment"])

ALL_LAYERS = {"L1", "L2", "L3", "L4", "L5"}


def _variant_key(v: ABVariant) -> str:
    return json.dumps([sorted(v.enabled_layers or []), v.overrides], sort_keys=True, default=str)


def get_variant_pipeline(app_state, v: ABVariant):
    """Build (once, then cache) the DefensePipeline for a variant."""
    from src.defense import DefensePipeline
    from src.rag.rag_pipeline import SYSTEM_PROMPT

    layers = set(v.enabled_layers or [])
    unknown = layers - ALL_LAYERS
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown layers {sorted(unknown)}; use {sorted(ALL_LAYERS)}")

    default = app_state.defense_pipeline
    if default is not None and layers == ALL_LAYERS and not v.overrides:
        return default

    cache = getattr(app_state, "variant_pipelines", None)
    if cache is None:
        cache = app_state.variant_pipelines = {}
    key = _variant_key(v)
    if key not in cache:
        try:
            # Never load a second copy of the ~700 MB detector: build without it
            # and share the default pipeline's instance.
            p = DefensePipeline.from_config(
                app_state.config, app_state.rag_pipeline.retriever, SYSTEM_PROMPT,
                enabled_layers=layers, **{**v.overrides, "detector_model_path": None},
            )
        except TypeError as e:
            raise HTTPException(status_code=422, detail=f"Invalid overrides for variant {v.name!r}: {e}")
        p.l2 = default.l2 if default is not None else None
        cache[key] = p
    return cache[key]


@router.post("/ab", response_model=ABResponse)
async def ab_compare(body: ABRequest, request: Request):
    """
    Run ``query`` through each variant in order. A variant with
    ``enabled_layers: null`` is the undefended RAG baseline; every other
    variant is a DefensePipeline built from configs/default.yaml with the given
    layer subset and overrides (e.g. ``{"output_format": "json"}``).
    ``latency_delta_ms`` is relative to the first baseline variant.
    """
    app_state = request.app.state
    if app_state.rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    names = [v.name for v in body.variants]
    if len(set(names)) != len(names):
        raise HTTPException(status_code=422, detail="Variant names must be unique")

    results = []
    for v in body.variants:
        results.append(await run_in_threadpool(_run_one, app_state, v, body.query))

    ref = next((r for r in results if r.enabled_layers is None), results[0])
    for r in results:
        r.latency_delta_ms = r.total_latency_ms - ref.total_latency_ms
    return ABResponse(query=body.query, experiment_id=body.experiment_id,
                      reference_variant=ref.name, results=results)


def _run_one(app_state, v: ABVariant, query: str) -> ABVariantResult:
    if v.enabled_layers is None:
        r = app_state.rag_pipeline.query(query)
        return ABVariantResult(
            name=v.name, enabled_layers=None, answer=r.answer, is_blocked=False,
            total_latency_ms=r.latency_ms,
        )
    pipeline = get_variant_pipeline(app_state, v)
    res = pipeline.run(query)
    return ABVariantResult(
        name=v.name,
        enabled_layers=sorted(pipeline.enabled_layers),
        answer=res.final_answer,
        is_blocked=res.is_blocked,
        block_layer=res.block_layer or None,
        block_reason=res.block_reason or None,
        traces=[LayerTraceSchema(layer=t.layer, passed=t.passed, detail=t.detail, latency_ms=t.latency_ms)
                for t in res.traces],
        total_latency_ms=res.total_latency_ms,
    )
