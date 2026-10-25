"""FastAPI application entry point."""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import documents, evaluation, experiment, rag
from api.schemas.response import HealthResponse
from src.utils import get_logger, load_config

logger = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG and defense pipelines on startup."""
    config = load_config("configs/default.yaml")

    store_path = config["rag"]["vector_store_path"]
    llm_provider = config["llm"]["provider"]
    llm_model = config["llm"]["model"]

    app.state.config = config
    app.state.rag_pipeline = None
    app.state.defense_pipeline = None
    app.state.variant_pipelines = {}   # /experiment/ab cache

    if os.path.exists(store_path):
        from src.rag import RAGPipeline
        app.state.rag_pipeline = RAGPipeline(
            store_path=store_path,
            llm_provider=llm_provider,
            llm_model=llm_model,
            top_k=config["rag"]["top_k"],
        )
        logger.info("RAG pipeline initialized.")

        if config["defense"]["enabled"]:
            from src.defense import DefensePipeline
            from src.rag.rag_pipeline import SYSTEM_PROMPT

            # Build straight from configs/default.yaml — same helper every
            # Experience_* script and run_evaluation.py use, so the API demo
            # never drifts from the layer parameters (blacklist path, L2
            # threshold, use_llm_audit, use_llama_guard, ...) the rest of the
            # framework is measured with.
            app.state.defense_pipeline = DefensePipeline.from_config(
                config, app.state.rag_pipeline.retriever, SYSTEM_PROMPT,
            )
            l2_active = app.state.defense_pipeline.l2 is not None
            logger.info(f"Defense pipeline initialized (L2 detector: {'on' if l2_active else 'off — no checkpoint'}).")
    else:
        logger.warning(
            f"Vector store not found at {store_path}. "
            "Run scripts/setup_vector_store.py first."
        )

    yield

    logger.info("Shutting down.")


app = FastAPI(
    title="Prompt Injection Attack & Defense Framework",
    description="RAG-based Prompt Injection research platform",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rag.router)
app.include_router(evaluation.router)
app.include_router(documents.router)
app.include_router(experiment.router)


@app.get("/health", response_model=HealthResponse)
async def health():
    pipeline = app.state.defense_pipeline
    return HealthResponse(
        status="ok",
        version="0.1.0",
        defense_enabled=pipeline is not None,
        l2_detector_active=bool(pipeline and pipeline.l2 is not None),
    )
