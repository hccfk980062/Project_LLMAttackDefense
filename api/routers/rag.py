"""RAG query endpoints."""
from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from api.schemas.request import QueryRequest
from api.schemas.response import LayerTraceSchema, QueryResponse

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.post("/query", response_model=QueryResponse)
async def query_rag(body: QueryRequest, request: Request):
    """Submit a query to the RAG system (with or without defense)."""
    app_state = request.app.state

    if body.defense_enabled and app_state.defense_pipeline:
        result = await run_in_threadpool(app_state.defense_pipeline.run, body.query)
        return QueryResponse(
            query=result.query,
            answer=result.final_answer,
            is_blocked=result.is_blocked,
            block_layer=result.block_layer or None,
            block_reason=result.block_reason or None,
            traces=[
                LayerTraceSchema(
                    layer=t.layer,
                    passed=t.passed,
                    detail=t.detail,
                    latency_ms=t.latency_ms,
                )
                for t in result.traces
            ],
            total_latency_ms=result.total_latency_ms,
            dropped_docs=result.dropped_docs,
        )

    # Undefended baseline
    if not app_state.rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    result = await run_in_threadpool(app_state.rag_pipeline.query, body.query)
    return QueryResponse(
        query=result.query,
        answer=result.answer,
        is_blocked=False,
        total_latency_ms=result.latency_ms,
        sources=result.sources,
    )
