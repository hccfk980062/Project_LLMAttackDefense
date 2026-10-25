"""Evaluation endpoints for running experiments via API."""
import copy
import random
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

router = APIRouter(prefix="/evaluation", tags=["Evaluation"])


class EvalRequest(BaseModel):
    experiment_name: str = "api_experiment"
    attack_types: list[str] = ["direct_injection"]
    n_samples: int = 50
    include_baseline: bool = True
    # Benign queries (held-out set, see src/evaluation/benign.py) — without
    # them FPR and the latency overhead can't be measured.
    n_benign: int = 20
    include_external: bool = False
    # End-to-end indirect scenarios (trigger query + poisoned retrieved doc).
    include_indirect: bool = False
    max_indirect: int = 10
    seed: int = 42


class EvalStatusResponse(BaseModel):
    experiment_name: str
    status: str
    result_path: Optional[str] = None


_running_experiments: dict[str, dict] = {}


@router.post("/run", response_model=EvalStatusResponse)
async def run_evaluation(body: EvalRequest, background_tasks: BackgroundTasks, request: Request):
    """Kick off a background evaluation experiment."""
    if request.app.state.defense_pipeline is None:
        raise HTTPException(status_code=503, detail="Defense pipeline not initialized")
    if _running_experiments.get(body.experiment_name, {}).get("status") == "running":
        raise HTTPException(status_code=409, detail="An experiment with this name is already running")
    _running_experiments[body.experiment_name] = {"status": "running"}

    background_tasks.add_task(_run_eval_task, request.app.state, body)

    return EvalStatusResponse(experiment_name=body.experiment_name, status="running")


@router.get("/status/{experiment_name}", response_model=EvalStatusResponse)
async def get_status(experiment_name: str):
    entry = _running_experiments.get(experiment_name, {"status": "not_found"})
    return EvalStatusResponse(experiment_name=experiment_name, **entry)


def build_eval_inputs(body: EvalRequest):
    """(attack_samples, benign_samples, indirect_scenarios) for a request."""
    from src.attack import default_indirect_scenarios, generate_samples, load_external_attack_datasets
    from src.attack.dataset_loader import EXTERNAL_BENCHMARKS
    from src.evaluation.benign import heldout_benign_queries, to_benign_samples
    from src.utils import load_config

    attack_cfg = load_config("configs/attack.yaml")
    # indirect_injection as a *user query* is really a direct injection; it is
    # measured through the scenarios (include_indirect) instead.
    types = [t for t in (body.attack_types or []) if t != "indirect_injection"]
    samples = generate_samples(types, attack_config=attack_cfg)[: body.n_samples] if types else []
    if body.include_external:
        ext = load_external_attack_datasets(attack_cfg, names=EXTERNAL_BENCHMARKS)
        samples += random.Random(body.seed).sample(ext, min(body.n_samples, len(ext)))
    pool = [q for q, _ in heldout_benign_queries()]
    benign = to_benign_samples(random.Random(body.seed).sample(pool, min(body.n_benign, len(pool))))
    scenarios = default_indirect_scenarios(max_scenarios=body.max_indirect, seed=body.seed) \
        if body.include_indirect else []
    return samples, benign, scenarios


def _evaluate_sync(app_state, body: EvalRequest) -> str:
    from src.attack import PoisoningRetriever
    from src.evaluation.evaluator import Evaluator

    samples, benign, scenarios = build_eval_inputs(body)
    defense = app_state.defense_pipeline
    rag = app_state.rag_pipeline

    poisoner = None
    if scenarios:
        # Shallow copies that retrieve through one PoisoningRetriever, so both
        # systems see identical poisoning while the live app.state pipelines
        # (serving /rag/query concurrently) are never exposed to it. Layers,
        # LLM clients and the L2 model are shared, not duplicated.
        poisoner = PoisoningRetriever(rag.retriever)
        rag, defense = copy.copy(rag), copy.copy(defense)
        rag.retriever = defense.retriever = poisoner

    evaluator = Evaluator(
        # Pass .run directly (not a lambda collapsing it to final_answer) so
        # AttackRunner/Evaluator capture block_layer + per-layer latency.
        defended_fn=defense.run,
        undefended_fn=(lambda q: rag.query(q).answer) if body.include_baseline else None,
        detector=defense.l2,
    )
    evaluator.evaluate(samples, benign, experiment_name=body.experiment_name,
                       indirect_scenarios=scenarios, poisoner=poisoner)
    return str(Path(evaluator.output_dir) / f"{body.experiment_name}_summary.json")


async def _run_eval_task(app_state, body: EvalRequest):
    try:
        # Evaluation is long and synchronous (LLM calls) — keep it off the event loop.
        path = await run_in_threadpool(_evaluate_sync, app_state, body)
        _running_experiments[body.experiment_name] = {"status": "completed", "result_path": path}
    except Exception as e:
        _running_experiments[body.experiment_name] = {"status": f"failed: {e}"}
