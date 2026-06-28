"""
test_mdn_stub.py — Unit tests for the deterministic MDN testing stub.

Run with:
    python -m pytest tests/test_mdn_stub.py -v
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from generator.mdn import MotiveDecompositionNetwork
from generator.mdn_runtime_selector import MDNRuntimeSelector
from utils.mdn_stub import StubMDN, load_mdn_or_stub


class TestStubMDN:
    def test_single_obs_forward(self):
        stub = StubMDN(
            input_dim=8,
            num_objectives=2,
            fixed_alpha=[2.5, 3.5],
            fixed_support_values=[0.1, 0.9],
        )
        obs = torch.zeros(8, dtype=torch.float32)

        alpha, support = stub.forward_inference(obs)

        assert alpha.shape == (2,)
        assert support.shape == (2,)
        assert alpha.tolist() == [2.5, 3.5]
        # Use approximate comparison for float32 precision
        support_list = support.tolist()
        assert abs(support_list[0] - 0.1) < 1e-6, f"Expected ~0.1, got {support_list[0]}"
        assert abs(support_list[1] - 0.9) < 1e-6, f"Expected ~0.9, got {support_list[1]}"

    def test_batched_obs_forward(self):
        stub = StubMDN(
            input_dim=8,
            num_objectives=2,
            fixed_alpha=[5.0, 5.0],
            fixed_support_values=[1.0, 1.0],
        )
        batch_size = 5
        obs = torch.zeros((batch_size, 8), dtype=torch.float32)

        alpha, support = stub.forward_inference(obs)

        assert alpha.shape == (batch_size, 2)
        assert support.shape == (batch_size, 2)
        
        # Verify all batches get identical deterministic values
        for i in range(batch_size):
            assert alpha[i].tolist() == [5.0, 5.0]
            assert support[i].tolist() == [1.0, 1.0]

    def test_validation_errors(self):
        stub = StubMDN(input_dim=8)

        with pytest.raises(ValueError, match="shape"):
            stub.forward_inference(torch.zeros(7))  # Wrong single dim

        with pytest.raises(ValueError, match="shape"):
            stub.forward_inference(torch.zeros((4, 7)))  # Wrong batch dim

        with pytest.raises(ValueError, match="shape"):
            stub.forward_inference(torch.zeros((1, 2, 8)))  # Wrong ndim

    def test_device_mock(self):
        stub = StubMDN()
        stub = stub.to("cpu")  # Should not raise
        assert stub.device == "cpu"
        # Test that .to() returns self for method chaining
        returned_stub = stub.to("cpu")
        assert returned_stub is stub  # Verify it returns self


class TestLoadMDNOrStub:
    def test_missing_file_returns_stub(self):
        model = load_mdn_or_stub("does_not_exist_xyz.pt")
        assert isinstance(model, StubMDN)
        assert model.input_dim == 8

    def test_corrupted_file_returns_stub_gracefully(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_path = Path(tmpdir) / "corrupt.pt"
            bad_path.write_text("this is not a valid pytorch checkpoint", encoding="utf-8")

            # Must catch the torch.load exception internally and return a stub
            model = load_mdn_or_stub(bad_path)
            assert isinstance(model, StubMDN)

    def test_valid_checkpoint_returns_real_mdn(self):
        # Create a tiny real model and save it
        real_model = MotiveDecompositionNetwork(input_dim=4, num_objectives=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "valid.pt"
            torch.save(real_model.state_dict(), ckpt_path)

            loaded_model = load_mdn_or_stub(ckpt_path, input_dim=4)
            # Should be the real PyTorch module, not the stub
            assert isinstance(loaded_model, MotiveDecompositionNetwork)
            assert not isinstance(loaded_model, StubMDN)
            assert loaded_model.input_dim == 4

    def test_stub_functions_in_mdn_runtime_selector(self):
        """Prove that the stub is interface-compatible with the main runtime selector."""
        stub = StubMDN(fixed_alpha=[2.0, 2.0], fixed_support_values=[1.0, 1.0])
        # The selector calls model.eval() and model.to(device) internally
        selector = MDNRuntimeSelector(stub)
        
        # Build fake certified candidate
        from utils.mdn_contracts import CandidateSkillRecord
        candidates = [
            CandidateSkillRecord(
                skill_id="test_candidate",
                delta_r=5.0,
                delta_n=(2.0, 3.0),
                is_certified=True,
                gate_type="CDS",
                admission_margin=7.0,
                epsilon=0.0,
                baseline_id=None,
            )
        ]
        
        obs = np.zeros(8, dtype=np.float32)
        result = selector.select(obs, candidates)

        assert result.selected_skill_id == "test_candidate"
        # Ensure our fixed stub values propagated fully through the selector
        assert np.array_equal(result.alpha, np.array([2.0, 2.0], dtype=np.float32))
        assert np.array_equal(result.support_values, np.array([1.0, 1.0], dtype=np.float32))
