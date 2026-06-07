"""Runtime MDN orchestration for certification, selection, logging, and updates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from generator.mdn import MotiveDecompositionNetwork
from generator.mdn_auxiliary_trainer import MDNAuxiliaryTrainer
from generator.mdn_runtime_selector import MDNRuntimeSelector
from generator.mdn_trainer import MDNTrainer
from utils.mdn_contracts import CandidateSkillRecord, MDNDecisionRecord
from utils.mdn_record_builder import build_candidate_skill_records
from utils.mdn_runtime_pipeline import RuntimeCertificationPipeline
from utils.weight_set_store import WeightSetStore


@dataclass(frozen=True)
class StepResult:
    """Result of one online MDN step."""

    selected_skill_id: str | None
    behavior_probability: float | None
    weights_used: np.ndarray | None
    decision_record: MDNDecisionRecord | None
    policy_metrics: dict[str, float] | None
    certified_skill_ids: tuple[str, ...]


class MDNOnlineRunner:
    """Wire the MDN runtime pieces into one decision/update step."""

    def __init__(
        self,
        *,
        model: MotiveDecompositionNetwork,
        certification_pipeline: RuntimeCertificationPipeline,
        policy_trainer: MDNTrainer,
        baseline_stats: dict[str, Any],
        checkpoint_path: str = "models/mdn_policy_best.pth",
        store_path: Optional[str] = None,
        save_every_n_steps: int = 10,
        auxiliary_trainer: Optional[MDNAuxiliaryTrainer] = None,
        device: Optional[str] = None,
    ) -> None:
        if save_every_n_steps <= 0:
            raise ValueError("save_every_n_steps must be positive")

        self.model = model
        self.certification_pipeline = certification_pipeline
        self.policy_trainer = policy_trainer
        self.auxiliary_trainer = auxiliary_trainer
        self.baseline_stats = dict(baseline_stats)
        self.checkpoint_path = checkpoint_path
        self.store_path = store_path or certification_pipeline.config.store_path
        self.save_every_n_steps = int(save_every_n_steps)
        self.selector = MDNRuntimeSelector(model=model, device=device or str(policy_trainer.device))
        self._step_count = 0

    def step(
        self,
        *,
        observation: np.ndarray,
        candidate_skill_payloads: list[dict[str, Any]],
        execute_skill: Callable[[str], dict[str, Any]],
        payoff_weight: Optional[float] = None,
    ) -> StepResult:
        """Run one full runtime step from observation to policy update."""
        if not candidate_skill_payloads:
            raise ValueError("candidate_skill_payloads must not be empty")

        context = np.asarray(observation, dtype=np.float32).reshape(-1)
        candidate_skills = list(
            build_candidate_skill_records(
                skill_outcomes=candidate_skill_payloads,
                baseline_stats=self.baseline_stats,
                weight_store=self.certification_pipeline.weight_store,
            )
        )
        certified_candidates = self.certification_pipeline.certify_candidate_skills(
            context=context,
            candidate_skills=candidate_skills,
            baseline_stats=self.baseline_stats,
        )
        certified_skill_ids = tuple(
            candidate.skill_id for candidate in certified_candidates if candidate.is_certified
        )
        if not certified_skill_ids:
            self._step_count += 1
            self._maybe_save()
            return StepResult(
                selected_skill_id=None,
                behavior_probability=None,
                weights_used=None,
                decision_record=None,
                policy_metrics=None,
                certified_skill_ids=certified_skill_ids,
            )

        selection = self.selector.select(context, certified_candidates)
        outcome = dict(execute_skill(selection.selected_skill_id))
        if "actual_payoff" not in outcome or "actual_motives" not in outcome:
            raise ValueError("execute_skill must return 'actual_payoff' and 'actual_motives'")

        record = selection.build_decision_record(
            actual_payoff=float(outcome["actual_payoff"]),
            actual_motives=outcome["actual_motives"],
            utility=outcome.get("utility"),
            payoff_weight=self.policy_trainer.config.payoff_weight if payoff_weight is None else float(payoff_weight),
        )
        policy_metrics = self.policy_trainer.training_step(record)

        self.certification_pipeline.certify_candidate_skills(
            context=context,
            candidate_skills=certified_candidates,
            baseline_stats=self.baseline_stats,
            weights_used=selection.weights_used,
        )

        self._step_count += 1
        self._maybe_save()
        return StepResult(
            selected_skill_id=selection.selected_skill_id,
            behavior_probability=selection.behavior_probability,
            weights_used=selection.weights_used,
            decision_record=record,
            policy_metrics=policy_metrics,
            certified_skill_ids=certified_skill_ids,
        )

    def save(self) -> None:
        """Persist policy checkpoint and weight store."""
        self.policy_trainer.save_checkpoint(self.checkpoint_path)
        if self.store_path is not None:
            self.certification_pipeline.save_store(self.store_path)

    def _maybe_save(self) -> None:
        if self._step_count % self.save_every_n_steps == 0:
            self.save()

    @classmethod
    def load(
        cls,
        *,
        model: MotiveDecompositionNetwork,
        certification_pipeline: RuntimeCertificationPipeline,
        policy_trainer: MDNTrainer,
        baseline_stats: dict[str, Any],
        checkpoint_path: str = "models/mdn_policy_best.pth",
        store_path: Optional[str] = None,
        save_every_n_steps: int = 10,
        auxiliary_trainer: Optional[MDNAuxiliaryTrainer] = None,
        device: Optional[str] = None,
    ) -> "MDNOnlineRunner":
        """Load persisted runtime state into a new online runner."""
        runner = cls(
            model=model,
            certification_pipeline=certification_pipeline,
            policy_trainer=policy_trainer,
            baseline_stats=baseline_stats,
            checkpoint_path=checkpoint_path,
            store_path=store_path,
            save_every_n_steps=save_every_n_steps,
            auxiliary_trainer=auxiliary_trainer,
            device=device,
        )
        checkpoint_file = Path(checkpoint_path)
        if checkpoint_file.exists():
            restored = MDNTrainer.from_checkpoint(
                checkpoint_file,
                model=model,
                device=device or str(policy_trainer.device),
            )
            runner.policy_trainer = restored
            runner.selector = MDNRuntimeSelector(model=restored.model, device=device or str(restored.device))
            runner.model = restored.model

        store_file = Path(runner.store_path) if runner.store_path is not None else None
        if store_file is not None and store_file.exists():
            loaded_store = WeightSetStore.load(store_file)
            runner.certification_pipeline.weight_store._store = loaded_store._store

        return runner
