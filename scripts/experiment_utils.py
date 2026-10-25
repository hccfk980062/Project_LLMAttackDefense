"""Shared helpers for the Experience_* experiment scripts.

Single source of truth for: config loading, the poisoned-store build used by the
indirect-injection experiments, the marker payloads / trigger queries, and a
one-liner to construct a `DefensePipeline` from `configs/default.yaml`.
"""
from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path
from typing import Iterable

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.attack.indirect_injection import MARKER_PAYLOADS, SUCCESS_MARKERS, poison_documents
from src.attack.indirect_scenarios import TRIGGER_QUERIES, TRIGGER_QUERY_SETS
from src.defense import DefensePipeline
from src.evaluation.metrics import rate_with_ci
from src.rag.rag_pipeline import SYSTEM_PROMPT
from src.rag.retriever import Retriever
from src.rag.vector_store import build_vector_store, load_vector_store
from src.utils import load_config
from src.utils.tracking import ExperimentTracker

DOCS_DIR = "data/raw/documents"
RESULTS_DIR = Path("experiments/results")

# MARKER_PAYLOADS / SUCCESS_MARKERS (visible marker => automatic scoring) live in
# src/attack/indirect_injection.py so the detector-dataset builder can exclude them.

# TRIGGER_QUERIES (benign questions that reliably retrieve a known document)
# lives in src/attack/indirect_scenarios.py so run_evaluation.py can share it.


def load_all_configs(
    default_path: str = "configs/default.yaml",
    attack_path: str = "configs/attack.yaml",
    defense_path: str = "configs/defense.yaml",
):
    """Return (default_cfg, attack_cfg, defense_cfg)."""
    return (
        load_config(default_path),
        load_config(attack_path),
        load_config(defense_path),
    )


def load_and_chunk(docs_dir: str = DOCS_DIR, chunk_size: int = 512, chunk_overlap: int = 64):
    loader = DirectoryLoader(
        docs_dir, glob="**/*.txt", loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}, silent_errors=True,
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
    return splitter.split_documents(loader.load())


def check_marker(text: str) -> tuple[bool, str]:
    for m in SUCCESS_MARKERS:
        if m in (text or ""):
            return True, m
    return False, ""


def build_clean_store(chunks) -> str:
    path = tempfile.mkdtemp(prefix="clean_store_")
    build_vector_store(chunks, save_path=path)
    return path


def build_poisoned_store(
    chunks,
    poison_ratio: float = 0.30,
    seed: int = 42,
    payloads: Iterable[str] | None = None,
) -> tuple[str, list]:
    """
    Poison a fraction of the chunks with marker payloads and persist a FAISS
    store. Returns (store_path, attack_samples).
    """
    payload_list = list(payloads) if payloads is not None else MARKER_PAYLOADS
    poisoned_chunks, attack_samples = poison_documents(
        chunks, poison_ratio=poison_ratio, strategy="append", seed=seed
    )
    rng = random.Random(seed)
    for s in attack_samples:
        idx = s.metadata["doc_index"]
        poisoned_chunks[idx].page_content = chunks[idx].page_content + rng.choice(payload_list)

    path = tempfile.mkdtemp(prefix="poisoned_store_")
    build_vector_store(poisoned_chunks, save_path=path)
    return path, attack_samples


def get_top_k(default_cfg: dict) -> int:
    """Retrieval depth shared by every experiment: configs/default.yaml rag.top_k."""
    return default_cfg["rag"]["top_k"]


def make_retriever(store_path: str, top_k: int) -> Retriever:
    return Retriever(load_vector_store(store_path), top_k=top_k)


def build_defense_pipeline(
    retriever,
    default_cfg: dict,
    *,
    enabled_layers: set[str] | list[str] | None = None,
    use_llm_audit: bool = False,
    system_prompt: str = SYSTEM_PROMPT,
    **overrides,
) -> DefensePipeline:
    """Thin wrapper over DefensePipeline.from_config so every experiment builds
    the pipeline the same way (and honours configs/default.yaml)."""
    return DefensePipeline.from_config(
        default_cfg,
        retriever,
        system_prompt,
        enabled_layers=enabled_layers,
        use_llm_audit=use_llm_audit,
        **overrides,
    )


def log_experiment(default_cfg: dict, name: str, output: dict, use_wandb: bool = True) -> None:
    """Log an experiment's result dict to W&B (numeric fields; nested dicts are
    flattened, per-record lists skipped). No-op when W&B is off/unconfigured."""
    with ExperimentTracker(default_cfg, name, use_wandb=use_wandb) as tracker:
        tracker.log_metrics(output)


def add_wandb_flag(parser) -> None:
    parser.add_argument("--no-wandb", dest="use_wandb", action="store_false",
                        help="Disable Weights & Biases logging for this run.")


# Experience 1-3 scale: each (poison seed, trigger query) pair is one trial.
# Temperature is 0, so repeating a query on the same store adds almost no
# information — more queries and more poisoned stores do.
DEFAULT_POISON_SEEDS = [42, 43, 44]


def add_scale_args(parser) -> None:
    parser.add_argument("--queries", choices=sorted(TRIGGER_QUERY_SETS), default="extended",
                        help="Trigger query set: original (8) or extended (44, every document).")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_POISON_SEEDS,
                        help="Poisoning seeds; each builds a differently poisoned store.")


def get_trigger_queries(name: str) -> list[tuple[str, str]]:
    return TRIGGER_QUERY_SETS[name]


def fmt_rate(k: int, n: int) -> str:
    """'75.0% (6/8, 95% CI 41–93%)'."""
    r = rate_with_ci(k, n)
    lo, hi = r["ci95"]
    return f"{r['rate']:.1%} ({k}/{n}, 95% CI {lo:.0%}–{hi:.0%})"


__all__ = [
    "DOCS_DIR", "RESULTS_DIR", "MARKER_PAYLOADS", "SUCCESS_MARKERS", "TRIGGER_QUERIES",
    "SYSTEM_PROMPT", "load_all_configs", "load_and_chunk", "check_marker",
    "build_clean_store", "build_poisoned_store", "get_top_k", "make_retriever", "build_defense_pipeline",
    "log_experiment", "add_wandb_flag", "DEFAULT_POISON_SEEDS", "add_scale_args",
    "get_trigger_queries", "fmt_rate", "rate_with_ci",
]
