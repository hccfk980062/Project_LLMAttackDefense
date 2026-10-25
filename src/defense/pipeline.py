"""Defense pipeline orchestrator: runs all 5 layers in sequence."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .layer1_input_sanitizer import InputSanitizer
from .layer2_injection_detector import InjectionDetector
from .layer3_prompt_segmentation import PromptSegmenter
from .layer4_controlled_generation import ControlledGenerator, GenerationConfig
from .layer5_output_validator import OutputValidator


# Trace names for steps the undefended RAG performs as well. Everything else in
# DefenseResult.traces is pure defense overhead (see evaluation.metrics).
RETRIEVAL_TRACE = "Retrieval"
GENERATION_TRACE = "L4_ControlledGeneration"
NON_DEFENSE_TRACES = (RETRIEVAL_TRACE, GENERATION_TRACE)


@dataclass
class LayerTrace:
    layer: str
    passed: bool
    detail: dict = field(default_factory=dict)
    latency_ms: float = 0.0


@dataclass
class DefenseResult:
    query: str
    final_answer: str
    is_blocked: bool
    block_layer: str = ""
    block_reason: str = ""
    traces: list[LayerTrace] = field(default_factory=list)
    total_latency_ms: float = 0.0
    # Retrieved documents removed by the L1/L2 context scan before prompt
    # assembly: [{"layer", "source", "reason", "metadata"}]. Dropping a poisoned document
    # does not block the query — the user's question itself is legitimate.
    dropped_docs: list[dict] = field(default_factory=list)


class DefensePipeline:
    """
    Chains all 5 defense layers.
    Each layer can short-circuit the pipeline by blocking the request.
    """

    ALL_LAYERS = ("L1", "L2", "L3", "L4", "L5")

    def __init__(
        self,
        system_prompt: str,
        retriever: Any,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o-mini",
        detector_model_path: str | Path | None = None,
        detector_threshold: float = 0.7,
        max_input_length: int = 2000,
        use_llm_audit: bool = True,
        use_llama_guard: bool = False,
        blacklist_path: str | Path | None = None,
        keyword_blacklist: list[str] | None = None,
        guarded_system_prompt: bool = True,
        enabled_layers: set[str] | list[str] | None = None,
        scan_context_l1: bool = True,
        scan_context_l2: bool = True,
        context_threshold: float | None = None,
        gen_temperature: float = 0.0,
        max_tokens: int = 1024,
        output_format: str = "plain_text",
        llama_guard_model: str = "meta-llama/Meta-Llama-Guard-2-8B",
        audit_model: str | None = None,
        audit_prompt: str | None = None,
        max_output_chars: int = 4000,
    ):
        self.retriever = retriever
        self.enabled_layers = set(enabled_layers) if enabled_layers is not None else set(self.ALL_LAYERS)
        self._raw_system_prompt = system_prompt
        # L1/L2 also screen retrieved documents (indirect injection), not just
        # the user query. context_threshold=None reuses detector_threshold.
        self.scan_context_l1 = scan_context_l1
        self.scan_context_l2 = scan_context_l2
        self.context_threshold = context_threshold

        self.l1 = InputSanitizer(
            max_length=max_input_length,
            keyword_blacklist=keyword_blacklist,
            blacklist_path=blacklist_path,
        )

        self.l2 = (
            InjectionDetector(
                model_path=detector_model_path,
                threshold=detector_threshold,
            )
            if detector_model_path and Path(str(detector_model_path)).exists()
            else None
        )

        # L3 augments the system prompt with explicit trust-boundary rules so the
        # model treats [CONTEXT]/[USER] blocks as untrusted. The old pipeline
        # skipped this step even though every Experience script applied it.
        if guarded_system_prompt:
            effective_system_prompt = PromptSegmenter(
                system_prompt=system_prompt
            ).build_guarded_system_prompt()
        else:
            effective_system_prompt = system_prompt
        self.l3 = PromptSegmenter(system_prompt=effective_system_prompt)

        # L4 enforces the output format; L5 has to check against the same one.
        gen_config = GenerationConfig(
            temperature=gen_temperature,
            max_tokens=max_tokens,
            output_format=output_format,
            use_llama_guard=use_llama_guard,
            llama_guard_model=llama_guard_model,
        )
        self.l4 = ControlledGenerator(
            llm_provider=llm_provider,
            llm_model=llm_model,
            config=gen_config,
        )

        self.l5 = OutputValidator(
            llm_provider=llm_provider,
            llm_model=llm_model,
            use_llm_audit=use_llm_audit,
            audit_model=audit_model,
            audit_prompt=audit_prompt,
            output_format=output_format,
            max_output_chars=max_output_chars,
        )

    @classmethod
    def from_config(
        cls,
        cfg: dict,
        retriever: Any,
        system_prompt: str,
        *,
        enabled_layers: set[str] | list[str] | None = None,
        **overrides: Any,
    ) -> "DefensePipeline":
        """
        Build a pipeline straight from a parsed ``configs/default.yaml`` dict so
        experiment scripts stop hard-coding layer parameters.

        Any keyword in ``overrides`` wins over the config values.
        """
        llm = cfg.get("llm", {})
        layers = cfg.get("defense", {}).get("layers", {})
        l1 = layers.get("layer1_input_sanitizer", {})
        l2 = layers.get("layer2_injection_detector", {})
        l4 = layers.get("layer4_controlled_generation", {})
        l5 = layers.get("layer5_output_validator", {})

        kwargs: dict[str, Any] = dict(
            system_prompt=system_prompt,
            retriever=retriever,
            llm_provider=llm.get("provider", "openai"),
            llm_model=llm.get("model", "gpt-4o-mini"),
            detector_model_path=(l2.get("model_path") if l2.get("enabled", True) else None),
            detector_threshold=l2.get("threshold", 0.7),
            max_input_length=l1.get("max_input_length", 2000),
            blacklist_path=l1.get("keyword_blacklist_path"),
            use_llm_audit=l5.get("use_llm_audit", True),
            use_llama_guard=l4.get("use_llama_guard", False),
            enabled_layers=enabled_layers,
            scan_context_l1=l1.get("scan_context", True),
            scan_context_l2=l2.get("scan_context", True),
            context_threshold=l2.get("context_threshold"),
            gen_temperature=l4.get("temperature", 0.0),
            max_tokens=l4.get("max_tokens", llm.get("max_tokens", 1024)),
            output_format=l4.get("output_format", "plain_text"),
            llama_guard_model=l4.get("llama_guard_model", "meta-llama/Meta-Llama-Guard-2-8B"),
            audit_model=l5.get("audit_model"),
            audit_prompt=l5.get("audit_prompt"),
            max_output_chars=l5.get("max_output_chars", 4000),
        )
        kwargs.update(overrides)
        return cls(**kwargs)

    def run(self, user_input: str) -> DefenseResult:
        traces: list[LayerTrace] = []
        start_total = time.perf_counter()

        # Layer 1 — Input Sanitizer
        if "L1" in self.enabled_layers:
            t0 = time.perf_counter()
            l1_result = self.l1.sanitize(user_input)
            traces.append(LayerTrace(
                layer="L1_InputSanitizer",
                passed=not l1_result.is_blocked,
                detail={"reason": l1_result.reason, "pattern": l1_result.triggered_pattern},
                latency_ms=(time.perf_counter() - t0) * 1000,
            ))
            if l1_result.is_blocked:
                return self._blocked(user_input, "L1", l1_result.reason, traces, start_total)
            clean_input = l1_result.cleaned_text
        else:
            clean_input = user_input

        # Layer 2 — Injection Detector
        if self.l2 and "L2" in self.enabled_layers:
            t0 = time.perf_counter()
            l2_result = self.l2.detect(clean_input)
            traces.append(LayerTrace(
                layer="L2_InjectionDetector",
                passed=not l2_result.is_injection,
                detail={"label": l2_result.label, "confidence": l2_result.confidence},
                latency_ms=(time.perf_counter() - t0) * 1000,
            ))
            if l2_result.is_injection:
                return self._blocked(user_input, "L2", "injection_detected", traces, start_total)

        # Retrieval always happens, whichever layers are enabled.
        t0 = time.perf_counter()
        retrieved_docs = self.retriever.retrieve(clean_input)
        # Own trace (not folded into L3) so the defense's added cost can be
        # separated from work the undefended RAG does too (retrieval, L4 call).
        traces.append(LayerTrace(
            layer=RETRIEVAL_TRACE,
            passed=True,
            detail={"num_docs": len(retrieved_docs)},
            latency_ms=(time.perf_counter() - t0) * 1000,
        ))

        # Layers 1/2 on the retrieved context — indirect injection arrives here,
        # not in the user query. Flagged documents are dropped, not the query.
        dropped_docs: list[dict] = []
        if self.scan_context_l1 and "L1" in self.enabled_layers:
            retrieved_docs = self._scan_context_l1(retrieved_docs, traces, dropped_docs)
        if self.l2 and self.scan_context_l2 and "L2" in self.enabled_layers:
            retrieved_docs = self._scan_context_l2(retrieved_docs, traces, dropped_docs)

        # Layer 3 — Prompt Segmentation
        # "L3" only toggles the structured-tag isolation and the guarded system
        # prompt (used by the ablation study).
        t0 = time.perf_counter()
        context = self.retriever.format_context(retrieved_docs)
        if "L3" in self.enabled_layers:
            messages = self.l3.segment(clean_input, context).to_messages()
        else:
            messages = [
                ("system", self._raw_system_prompt),
                ("human", f"參考資料：\n\n{context}\n\n問題：{clean_input}"),
            ]
        traces.append(LayerTrace(
            layer="L3_PromptSegmentation",
            passed=True,
            detail={
                "num_docs_retrieved": len(retrieved_docs),
                "segmented": "L3" in self.enabled_layers,
            },
            latency_ms=(time.perf_counter() - t0) * 1000,
        ))

        # Layer 4 — Controlled Generation
        t0 = time.perf_counter()
        l4_result = self.l4.generate(messages)
        traces.append(LayerTrace(
            layer=GENERATION_TRACE,
            passed=not l4_result.was_blocked,
            detail={
                "block_reason": l4_result.block_reason,
                "output_format": self.l4.config.output_format,
                "attempts": l4_result.attempts,
            },
            latency_ms=(time.perf_counter() - t0) * 1000,
        ))
        if l4_result.was_blocked:
            return self._blocked(user_input, "L4", l4_result.block_reason, traces, start_total, dropped_docs)

        # Layer 5 — Output Validator
        final_answer = l4_result.content
        if "L5" in self.enabled_layers:
            t0 = time.perf_counter()
            l5_result = self.l5.validate(l4_result.content, structured=l4_result.structured)
            traces.append(LayerTrace(
                layer="L5_OutputValidator",
                passed=l5_result.is_safe,
                detail={
                    "verdict": l5_result.audit_verdict,
                    "reason": l5_result.blocked_reason,
                    "audit_model": l5_result.audit_model,
                },
                latency_ms=(time.perf_counter() - t0) * 1000,
            ))
            if not l5_result.is_safe:
                return self._blocked(user_input, "L5", l5_result.blocked_reason, traces, start_total, dropped_docs)
            final_answer = l5_result.final_output

        total_ms = (time.perf_counter() - start_total) * 1000
        return DefenseResult(
            query=user_input,
            final_answer=final_answer,
            is_blocked=False,
            traces=traces,
            total_latency_ms=total_ms,
            dropped_docs=dropped_docs,
        )

    @staticmethod
    def _drop_record(layer: str, doc: Any, reason: str) -> dict:
        # metadata is copied so experiments can score the scan against ground
        # truth (poisoned stores mark poisoned chunks with metadata["poisoned"]).
        metadata = dict(getattr(doc, "metadata", {}) or {})
        return {"layer": layer, "source": str(metadata.get("source", "")),
                "reason": reason, "metadata": metadata}

    def _scan_context_l1(self, docs: list, traces: list[LayerTrace], dropped: list[dict]) -> list:
        t0 = time.perf_counter()
        kept = []
        for doc in docs:
            res = self.l1.scan_context(doc.page_content)
            if res.is_blocked:
                dropped.append(self._drop_record(
                    "L1", doc, f"{res.reason}: {res.triggered_pattern[:60]}"))
            else:
                kept.append(doc)
        traces.append(LayerTrace(
            layer="L1_ContextScan",
            passed=True,
            detail={"n_scanned": len(docs), "n_dropped": len(docs) - len(kept)},
            latency_ms=(time.perf_counter() - t0) * 1000,
        ))
        return kept

    def _scan_context_l2(self, docs: list, traces: list[LayerTrace], dropped: list[dict]) -> list:
        t0 = time.perf_counter()
        results = self.l2.scan_documents(
            [d.page_content for d in docs], threshold=self.context_threshold,
        ) if docs else []
        kept = []
        for doc, res in zip(docs, results):
            if res.is_injection:
                dropped.append(self._drop_record(
                    "L2", doc, f"injection_detected: p={res.confidence:.3f}"))
            else:
                kept.append(doc)
        traces.append(LayerTrace(
            layer="L2_ContextScan",
            passed=True,
            detail={"n_scanned": len(docs), "n_dropped": len(docs) - len(kept),
                    "max_confidence": max((r.confidence for r in results), default=0.0)},
            latency_ms=(time.perf_counter() - t0) * 1000,
        ))
        return kept

    def _blocked(
        self,
        query: str,
        layer: str,
        reason: str,
        traces: list[LayerTrace],
        start: float,
        dropped_docs: list[dict] | None = None,
    ) -> DefenseResult:
        return DefenseResult(
            query=query,
            final_answer="",
            is_blocked=True,
            block_layer=layer,
            block_reason=reason,
            traces=traces,
            total_latency_ms=(time.perf_counter() - start) * 1000,
            dropped_docs=dropped_docs or [],
        )
