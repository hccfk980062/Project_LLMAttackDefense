"""End-to-end evaluation orchestrator."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, List

from src.attack.attack_runner import AttackRunner
from src.attack.attack_types import AttackSample
from .metrics import MetricReport, apply_latency_overhead, compute_metrics
from .report_generator import generate_report


class Evaluator:
    def __init__(
        self,
        defended_fn: Callable[[str], Any],
        undefended_fn: Callable[[str], Any] | None = None,
        judge_model: str = "gpt-4o-mini",
        output_dir: str | Path = "experiments/results",
        detector: Any | None = None,
    ):
        """
        Args:
            defended_fn: System with defense enabled. Pass the pipeline's `.run`
                method directly (e.g. `defense.run`) rather than
                `lambda q: defense.run(q).final_answer` — AttackRunner duck-types
                the richer DefenseResult to capture block_layer / per-layer
                latency, which a plain string throws away.
            undefended_fn: Baseline system without defense (for ASR baseline).
                A plain `Callable[[str], str]` is fine here — there's no layer
                to attribute a block to.
            judge_model: LLM to judge attack success.
            output_dir: Directory to save results.
            detector: Optional InjectionDetector (src.defense.layer2_injection_detector).
                When given, Evaluator runs it independently over every attack
                (label=1) and benign (label=0) sample to compute a real
                `detector_f1` — without this, detector_f1 stays 0.0 in every
                report (it's simply never measured, not evidence L2 is bad).
        """
        self.defended_fn = defended_fn
        self.undefended_fn = undefended_fn
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.judge_model = judge_model
        self.detector = detector

    def _detector_predictions(
        self,
        attack_samples: List[AttackSample],
        benign_samples: List[AttackSample],
    ) -> tuple[list[int] | None, list[int] | None]:
        """Run self.detector standalone over the sample payloads to get an
        honest detector_f1 — independent of which pipeline variant is being
        scored, since L2's own classification quality doesn't change with it."""
        if self.detector is None:
            return None, None
        texts = [s.payload for s in attack_samples] + [s.payload for s in benign_samples]
        if not texts:
            return None, None
        labels = [1] * len(attack_samples) + [0] * len(benign_samples)
        preds = [int(r.is_injection) for r in self.detector.detect_batch(texts)]
        return labels, preds

    def evaluate(
        self,
        attack_samples: List[AttackSample],
        benign_samples: List[AttackSample],
        experiment_name: str = "experiment",
        indirect_scenarios: List[Any] | None = None,
        poisoner: Any | None = None,
        tracker: Any | None = None,
    ) -> dict[str, MetricReport]:
        """
        Args:
            indirect_scenarios: IndirectScenario objects (src.attack.indirect_scenarios).
                Each sends its benign trigger query while ``poisoner`` injects the
                payload into the retrieved context, and the results are scored
                together with ``attack_samples`` (as attack_type indirect_injection).
            poisoner: the PoisoningRetriever that BOTH defended_fn and
                undefended_fn retrieve through. Required with indirect_scenarios.
            tracker: optional src.utils.tracking.ExperimentTracker (W&B).
        """
        if indirect_scenarios and poisoner is None:
            raise ValueError("indirect_scenarios need the PoisoningRetriever both systems retrieve through")
        indirect_scenarios = indirect_scenarios or []
        detector_labels, detector_preds = self._detector_predictions(attack_samples, benign_samples)

        runner_defended = AttackRunner(self.defended_fn, judge_model=self.judge_model)

        defended_attack_results = runner_defended.run(attack_samples, desc="[Defended] Attacks")
        if indirect_scenarios:
            defended_attack_results += runner_defended.run_indirect(
                indirect_scenarios, poisoner, desc="[Defended] Indirect")
        defended_benign_results = runner_defended.run(benign_samples, desc="[Defended] Benign")

        defended_report = compute_metrics(
            defended_attack_results, defended_benign_results,
            detector_labels=detector_labels, detector_preds=detector_preds,
        )

        reports = {"defended": defended_report}

        if self.undefended_fn:
            runner_baseline = AttackRunner(self.undefended_fn, judge_model=self.judge_model)
            baseline_attack_results = runner_baseline.run(attack_samples, desc="[Baseline] Attacks")
            if indirect_scenarios:
                baseline_attack_results += runner_baseline.run_indirect(
                    indirect_scenarios, poisoner, desc="[Baseline] Indirect")
            baseline_benign_results = runner_baseline.run(benign_samples, desc="[Baseline] Benign")
            baseline_report = compute_metrics(
                baseline_attack_results, baseline_benign_results,
                detector_labels=detector_labels, detector_preds=detector_preds,
            )
            reports["baseline"] = baseline_report
            apply_latency_overhead(defended_report, baseline_report)

            runner_baseline.save_results(
                baseline_attack_results,
                self.output_dir / f"{experiment_name}_baseline_attacks.json",
            )

        runner_defended.save_results(
            defended_attack_results,
            self.output_dir / f"{experiment_name}_defended_attacks.json",
        )

        generate_report(reports, self.output_dir / f"{experiment_name}_report.md")
        self._save_json_summary(reports, self.output_dir / f"{experiment_name}_summary.json")

        if tracker is not None:
            for name, r in reports.items():
                tracker.log_metrics(self._summary_dict(r), prefix=name)
            tracker.log_table(
                "defended_attacks",
                [{"id": x.sample.id, "attack_type": x.sample.attack_type.value,
                  "is_success": x.is_success, "was_blocked": x.was_blocked,
                  "block_layer": x.block_layer, "latency_ms": x.latency_ms}
                 for x in defended_attack_results],
            )

        print(defended_report.summary())
        return reports

    @staticmethod
    def _summary_dict(r: MetricReport) -> dict:
        return {
            "asr": r.asr,
            "fpr": r.fpr,
            "detector_f1": r.detector_f1,
            "mean_latency_ms": r.mean_latency_ms,
            "p95_latency_ms": r.p95_latency_ms,
            "asr_by_type": r.asr_by_type,
            "asr_user_input": r.asr_user_input,
            "asr_knowledge_base": r.asr_knowledge_base,
            "benign_mean_latency_ms": r.benign_mean_latency_ms,
            "latency_overhead_ms": r.latency_overhead_ms,
            "defense_overhead_ms": r.defense_overhead_ms,
            "n_attacks": r.n_attacks,
            "n_benign": r.n_benign,
            "blocked_by_layer": r.blocked_by_layer,
            "benign_blocked_by_layer": r.benign_blocked_by_layer,
        }

    def _save_json_summary(self, reports: dict, path: Path) -> None:
        summary = {name: self._summary_dict(r) for name, r in reports.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
