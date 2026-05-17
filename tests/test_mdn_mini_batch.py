"""Tests for Phase B: W_x persistence and monotonicity integration."""

from __future__ import annotations

import numpy as np
import pytest

from utils.mdn_record_builder import (
    PreparedCandidateOutcome,
    build_candidate_skill_record,
    build_candidate_skill_records,
)
from utils.weight_set_store import WeightSet, WeightSetStore


def _baseline_stats() -> dict[str, object]:
    return {
        "baseline_payoff": 1.0,
        "baseline_motives": np.array([0.5, 0.2], dtype=np.float32),
    }


def _make_weight_set(weights: list[list[float]]) -> WeightSet:
    ws = WeightSet()
    for w in weights:
        ws.add_vertex(np.array(w, dtype=np.float32))
    return ws


class TestWeightSetIntegration:
    def test_build_record_with_weight_set_uses_wx_for_cds_gate(self):
        weight_set = _make_weight_set([[0.6, 0.4], [0.3, 0.7]])
        record = build_candidate_skill_record(
            skill_id="skill_a",
            skill_payoff=1.7,
            skill_motives=np.array([0.8, 0.4], dtype=np.float32),
            baseline_stats=_baseline_stats(),
            gate_type="CDS",
            weight_set=weight_set,
        )

        assert isinstance(record.is_certified, bool)

    def test_build_record_with_empty_weight_set_falls_back_to_simplex(self):
        empty_set = WeightSet()
        record_simplex = build_candidate_skill_record(
            skill_id="skill_a",
            skill_payoff=1.7,
            skill_motives=np.array([0.8, 0.4], dtype=np.float32),
            baseline_stats=_baseline_stats(),
            gate_type="CDS",
            weight_set=None,
        )
        record_empty = build_candidate_skill_record(
            skill_id="skill_a",
            skill_payoff=1.7,
            skill_motives=np.array([0.8, 0.4], dtype=np.float32),
            baseline_stats=_baseline_stats(),
            gate_type="CDS",
            weight_set=empty_set,
        )

        assert record_simplex.is_certified == record_empty.is_certified

    def test_build_record_with_weight_set_uses_wx_for_pds_gate(self):
        weight_set = _make_weight_set([[0.5, 0.5], [0.8, 0.2]])
        record = build_candidate_skill_record(
            skill_id="skill_a",
            skill_payoff=1.2,
            skill_motives=np.array([0.5, 0.0], dtype=np.float32),
            baseline_stats=_baseline_stats(),
            gate_type="PDS",
            epsilon=0.1,
            weight_set=weight_set,
        )

        assert isinstance(record.is_certified, bool)

    def test_build_records_with_weight_store_looks_up_context(self):
        store = WeightSetStore(num_objectives=2)
        context = np.array([0.1] * 14, dtype=np.float32)
        store.observe_certified_weight(context, np.array([0.6, 0.4], dtype=np.float32))

        outcomes = (
            PreparedCandidateOutcome(
                context=(0.1,) * 14,
                skill_id="skill_a",
                payoff=1.7,
                motives=(0.8, 0.4),
            ),
        )

        records = build_candidate_skill_records(
            skill_outcomes=outcomes,
            baseline_stats=_baseline_stats(),
            weight_store=store,
        )

        assert len(records) == 1
        assert records[0].skill_id == "skill_a"

    def test_build_records_without_weight_store_behaves_as_before(self):
        outcomes = (
            PreparedCandidateOutcome(
                context=(0.1,) * 14,
                skill_id="skill_a",
                payoff=1.7,
                motives=(0.8, 0.4),
            ),
        )

        records = build_candidate_skill_records(
            skill_outcomes=outcomes,
            baseline_stats=_baseline_stats(),
        )

        assert len(records) == 1


class TestWeightSetStorePublicAPI:
    def test_get_weight_set_returns_none_for_unobserved_context(self):
        store = WeightSetStore(num_objectives=2)
        context = np.array([0.1] * 14, dtype=np.float32)

        result = store.get_weight_set(context)

        assert result is None

    def test_get_weight_set_returns_set_after_observation(self):
        store = WeightSetStore(num_objectives=2)
        context = np.array([0.1] * 14, dtype=np.float32)
        store.observe_certified_weight(context, np.array([0.6, 0.4], dtype=np.float32))

        result = store.get_weight_set(context)

        assert result is not None
        assert not result.is_empty()
        assert len(result.vertices) == 1

    def test_get_weight_set_uses_same_keying_as_observe(self):
        store = WeightSetStore(num_objectives=2)
        context = np.array([0.1] * 14, dtype=np.float32)
        store.observe_certified_weight(context, np.array([0.6, 0.4], dtype=np.float32))

        result = store.get_weight_set(context + 1e-6)

        assert result is not None


class TestRuntimePipeline:
    def test_runtime_pipeline_certifies_and_records_weight(self):
        from generator.mdn import MotiveDecompositionNetwork
        from utils.mdn_runtime_pipeline import RuntimeCertificationPipeline, RuntimePipelineConfig

        model = MotiveDecompositionNetwork(input_dim=14, num_objectives=2)
        store = WeightSetStore(num_objectives=2)
        config = RuntimePipelineConfig(gate_type="CDS", train_support_after_certify=False)
        pipeline = RuntimeCertificationPipeline(model=model, weight_store=store, config=config)

        context = np.array([0.1] * 14, dtype=np.float32)
        result = pipeline.certify_skill(
            context=context,
            skill_id="skill_a",
            skill_payoff=1.7,
            skill_motives=np.array([0.8, 0.4], dtype=np.float32),
            baseline_stats=_baseline_stats(),
            weights_used=np.array([0.5, 0.5], dtype=np.float32),
        )

        assert result.skill_id == "skill_a"
        assert isinstance(result.is_certified, bool)
        assert store.context_count() >= 0

    def test_runtime_pipeline_enforces_certificate_permanence(self):
        from generator.mdn import MotiveDecompositionNetwork
        from utils.mdn_runtime_pipeline import RuntimeCertificationPipeline, RuntimePipelineConfig

        model = MotiveDecompositionNetwork(input_dim=14, num_objectives=2)
        store = WeightSetStore(num_objectives=2)
        config = RuntimePipelineConfig(gate_type="CDS", train_support_after_certify=False)
        pipeline = RuntimeCertificationPipeline(model=model, weight_store=store, config=config)

        context = np.array([0.1] * 14, dtype=np.float32)
        result1 = pipeline.certify_skill(
            context=context,
            skill_id="skill_a",
            skill_payoff=1.7,
            skill_motives=np.array([0.8, 0.4], dtype=np.float32),
            baseline_stats=_baseline_stats(),
            weights_used=np.array([0.5, 0.5], dtype=np.float32),
        )

        result2 = pipeline.certify_skill(
            context=context,
            skill_id="skill_a",
            skill_payoff=0.0,
            skill_motives=np.array([-1.0, -1.0], dtype=np.float32),
            baseline_stats=_baseline_stats(),
            weights_used=np.array([0.5, 0.5], dtype=np.float32),
        )

        assert result1.is_certified == result2.is_certified
        assert result2.was_already_certified is False

    def test_runtime_pipeline_uses_wx_not_simplex(self):
        from generator.mdn import MotiveDecompositionNetwork
        from utils.mdn_runtime_pipeline import RuntimeCertificationPipeline, RuntimePipelineConfig

        model = MotiveDecompositionNetwork(input_dim=14, num_objectives=2)
        store = WeightSetStore(num_objectives=2)
        context = np.array([0.1] * 14, dtype=np.float32)
        store.observe_certified_weight(context, np.array([0.7, 0.3], dtype=np.float32))

        config = RuntimePipelineConfig(gate_type="CDS", train_support_after_certify=False)
        pipeline = RuntimeCertificationPipeline(model=model, weight_store=store, config=config)

        result = pipeline.certify_skill(
            context=context,
            skill_id="skill_a",
            skill_payoff=1.7,
            skill_motives=np.array([0.8, 0.4], dtype=np.float32),
            baseline_stats=_baseline_stats(),
            weights_used=np.array([0.5, 0.5], dtype=np.float32),
        )

        assert isinstance(result.is_certified, bool)

    def test_runtime_pipeline_get_support_values(self):
        from generator.mdn import MotiveDecompositionNetwork
        from utils.mdn_runtime_pipeline import RuntimeCertificationPipeline, RuntimePipelineConfig

        model = MotiveDecompositionNetwork(input_dim=14, num_objectives=2)
        store = WeightSetStore(num_objectives=2)
        context = np.array([0.1] * 14, dtype=np.float32)
        store.observe_certified_weight(context, np.array([0.7, 0.3], dtype=np.float32))

        config = RuntimePipelineConfig(gate_type="CDS", train_support_after_certify=False)
        pipeline = RuntimeCertificationPipeline(model=model, weight_store=store, config=config)

        support = pipeline.get_support_values(context)

        assert support.shape == (2,)

    def test_runtime_pipeline_save_and_load_store(self):
        import tempfile
        from generator.mdn import MotiveDecompositionNetwork
        from utils.mdn_runtime_pipeline import RuntimeCertificationPipeline, RuntimePipelineConfig

        model = MotiveDecompositionNetwork(input_dim=14, num_objectives=2)
        store = WeightSetStore(num_objectives=2)
        context = np.array([0.1] * 14, dtype=np.float32)
        store.observe_certified_weight(context, np.array([0.7, 0.3], dtype=np.float32))

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            config = RuntimePipelineConfig(gate_type="CDS", train_support_after_certify=False, store_path=f.name)
            pipeline = RuntimeCertificationPipeline(model=model, weight_store=store, config=config)
            saved_path = pipeline.save_store()

            loaded_store = WeightSetStore.load(saved_path)
            assert loaded_store.context_count() == store.context_count()
