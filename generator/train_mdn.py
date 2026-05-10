"""Offline entrypoints for utility-driven MDN training from decision data."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

from utils.mdn_logging import build_decision_record
from utils.mdn_record_builder import (
    PreparedCandidateOutcome,
    build_candidate_skill_records,
    group_candidate_outcomes_by_context,
)
from utils.mdn_reward import compute_advantage, compute_mdn_utility
from utils.mdn_selection import alpha_to_mean_weights, select_best_candidate

from generator.mdn import MotiveDecompositionNetwork
from generator.mdn_trainer import MDNTrainer, MDNTrainerConfig, create_trainer_for_model
from utils.mdn_contracts import MDNDecisionRecord


def train_mdn_from_records(
    records: Iterable[MDNDecisionRecord],
    *,
    checkpoint_path: str = "models/mdn_policy_best.pth",
    seed: int = 0,
    device: Optional[str] = None,
) -> dict[str, float | str]:
    """Train MDN from prebuilt offline decision records and save a checkpoint."""
    records = list(records)
    if not records:
        raise ValueError("train_mdn_from_records requires at least one decision record")

    context_dim = len(records[0].context)
    num_objectives = len(records[0].alpha)
    model = MotiveDecompositionNetwork(input_dim=context_dim, num_objectives=num_objectives)
    trainer = create_trainer_for_model(model, seed=seed, device=device)
    trainer.config.checkpoint_path = checkpoint_path

    metrics = trainer.train_records(records)
    saved_path = trainer.save_checkpoint(checkpoint_path)
    return {**metrics, "checkpoint_path": saved_path}


def build_records_from_prepared_candidate_outcomes(
    *,
    prepared_outcomes: Iterable[dict[str, Any] | PreparedCandidateOutcome],
    baseline_stats: dict[str, Any],
    checkpoint_path: str = "models/mdn_policy_best.pth",
    seed: int = 0,
    device: Optional[str] = None,
    payoff_weight: float = 0.0,
) -> list[MDNDecisionRecord]:
    """Build offline MDN decision records from prepared candidate outcome payloads."""
    grouped = group_candidate_outcomes_by_context(prepared_outcomes)
    if not grouped:
        raise ValueError("prepared_outcomes must contain at least one candidate outcome")

    first_context = next(iter(grouped))
    model = MotiveDecompositionNetwork(input_dim=len(first_context), num_objectives=2)
    trainer = create_trainer_for_model(model, seed=seed, device=device)
    trainer.config.checkpoint_path = checkpoint_path

    records: list[MDNDecisionRecord] = []
    for context, group in grouped.items():
        candidate_skills = build_candidate_skill_records(
            skill_outcomes=group,
            baseline_stats=baseline_stats,
        )
        alpha, support_values = trainer.model(
            __import__("torch").tensor(context, dtype=__import__("torch").float32, device=trainer.device)
        )
        alpha_np = alpha.detach().cpu().numpy()
        support_np = support_values.detach().cpu().numpy()
        weights_used = alpha_to_mean_weights(alpha_np)
        selected_skill_id, selected_score = select_best_candidate(candidate_skills, weights_used)

        selected_outcome = next(outcome for outcome in group if outcome.skill_id == selected_skill_id)
        utility = compute_mdn_utility(
            actual_motives=selected_outcome.motives,
            weights_used=weights_used,
            actual_payoff=selected_outcome.payoff,
            payoff_weight=payoff_weight,
        )
        record = build_decision_record(
            context=context,
            alpha=alpha_np,
            support_values=support_np,
            weights_used=weights_used,
            candidate_skills=candidate_skills,
            selected_skill_id=selected_skill_id,
            selected_score=selected_score,
            actual_payoff=selected_outcome.payoff,
            actual_motives=selected_outcome.motives,
            utility=utility,
        )
        records.append(record)
    return records


def train_mdn_from_prepared_outcomes(
    *,
    prepared_outcomes: Iterable[dict[str, Any] | PreparedCandidateOutcome],
    baseline_stats: dict[str, Any],
    checkpoint_path: str = "models/mdn_policy_best.pth",
    seed: int = 0,
    device: Optional[str] = None,
    payoff_weight: float = 0.0,
) -> dict[str, float | str]:
    """Build records from prepared outcomes, then train MDN offline."""
    records = build_records_from_prepared_candidate_outcomes(
        prepared_outcomes=prepared_outcomes,
        baseline_stats=baseline_stats,
        checkpoint_path=checkpoint_path,
        seed=seed,
        device=device,
        payoff_weight=payoff_weight,
    )
    return train_mdn_from_records(records, checkpoint_path=checkpoint_path, seed=seed, device=device)


def train() -> None:
    """Placeholder CLI entrypoint until external data loading is fully wired."""
    raise NotImplementedError(
        "train() requires external loading of prepared candidate outcomes; use train_mdn_from_prepared_outcomes(...) or train_mdn_from_records(...) from code for now."
    )


if __name__ == "__main__":
    train()
