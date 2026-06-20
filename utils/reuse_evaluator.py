"""
Zero-shot reuse evaluator for SubRep certified skills.

Validates that certified skills can be safely reused under new motive weights
without retraining. Supports two certification modes:

    FULL_SIMPLEX — Globally reusable: safe under any valid simplex weight.
    MDN_WX      — Contextually reusable: safe under the current learned
                  motive geometry described by support directions/values.

Mathematical safety is the primary guarantee; empirical performance checks
are secondary validation.
"""

from __future__ import annotations

import numpy as np
from certification.certificate_schema import Certificate

# ── Region-type constants ────────────────────────────────────────────────────
FULL_SIMPLEX = "FULL_SIMPLEX"
MDN_WX = "MDN_WX"


class ZeroShotEvaluator:
    """Evaluates whether a certified skill is safe to reuse under new weights.

    Two modes are supported:

    1. **FULL_SIMPLEX** — A skill certified over the full simplex is safe under
       *any* valid weight vector. Only the validity of ``new_weight`` is checked.

    2. **MDN_WX** — A skill certified within a context-conditioned weight set
       W_x is safe if delta_r covers the worst-case motive cost h_Wx(-delta_n)
       computed from the current context's support descriptor.
    """

    # ── Primary: Mathematical safety check ───────────────────────────────────

    def is_safe_mathematically(
        self,
        certificate: Certificate,
        new_weight: list | np.ndarray,
        support_directions: list | np.ndarray | None = None,
        support_values: list | np.ndarray | None = None,
    ) -> bool:
        """Check whether a certified skill is safe to reuse at *new_weight*.

        Args:
            certificate: The skill's admission certificate.
            new_weight:  Target weight vector (must be a valid simplex point).
            support_directions: (MDN_WX only) Query directions describing W_x.
            support_values:     (MDN_WX only) Support thresholds h_j for W_x.
                                NOTE: These are *not* weight vectors and do NOT
                                need to sum to 1.

        Returns:
            True if reuse is mathematically safe; False otherwise.
        """
        w = np.asarray(new_weight, dtype=np.float64)
        self._validate_simplex(w)

        # ── Mode 1: Full-simplex global reuse ────────────────────────────
        if support_directions is None or support_values is None:
            # Any valid simplex weight is safe for a full-simplex certificate.
            return True

        # ── Mode 2: MDN/contextual reuse ─────────────────────────────────
        dirs = np.asarray(support_directions, dtype=np.float64)
        vals = np.asarray(support_values, dtype=np.float64)
        # NOTE: Do NOT validate that vals sums to 1 — they are thresholds, not
        # weights.  E.g., [0.8, 0.4] is valid even though 0.8 + 0.4 ≠ 1.

        delta_n = np.asarray(certificate.delta_n, dtype=np.float64)
        delta_r = float(certificate.delta_r)
        epsilon = float(certificate.epsilon)

        h_wx = self._compute_h_wx(neg_delta_n=-delta_n, directions=dirs, values=vals)

        if certificate.gate_type == "CDS":
            return delta_r >= h_wx
        elif certificate.gate_type == "PDS":
            return delta_r >= h_wx - epsilon
        else:
            raise ValueError(f"Unknown gate_type: {certificate.gate_type!r}")

    # ── Secondary: Empirical performance validation ──────────────────────────

    def evaluate_performance(
        self,
        certificate: Certificate,
        new_weight: list | np.ndarray,
    ) -> dict:
        """Compute the weighted performance score under *new_weight*.

        Uses the certificate's stored delta values (computed at certification
        time) as a controlled evaluation — no environment re-simulation needed.

        Args:
            certificate: The skill's admission certificate.
            new_weight:  Target weight vector (valid simplex point).

        Returns:
            Dictionary with keys: delta_r, delta_n, weighted_score,
            beats_baseline.
        """
        w = np.asarray(new_weight, dtype=np.float64)
        self._validate_simplex(w)

        delta_r = float(certificate.delta_r)
        delta_n = np.asarray(certificate.delta_n, dtype=np.float64)

        weighted_score = delta_r + float(np.dot(w, delta_n))

        return {
            "delta_r": delta_r,
            "delta_n": tuple(float(v) for v in delta_n),
            "weighted_score": weighted_score,
            "beats_baseline": weighted_score > 0.0,
        }

    # ── Library Integration ──────────────────────────────────────────────────

    def is_reusable_via_library(
        self,
        library,
        skill_id: str,
        current_weight: list | np.ndarray,
        support_directions: list | np.ndarray | None = None,
        support_values: list | np.ndarray | None = None,
    ) -> bool:
        """Check if a skill is reusable via the runtime SkillLibrary.

        Delegates the admissibility check to the library's unified
        `query_admissible()` method. This is the recommended path for
        runtime selection, replacing standalone mathematical evaluation.

        Args:
            library: The SkillLibrary instance.
            skill_id: The ID of the skill to check.
            current_weight: The active motive weight.
            support_directions: (MDN_WX only) Core W_x directions.
            support_values: (MDN_WX only) Core W_x thresholds.

        Returns:
            True if the skill is found in the returned admissible list.
        """
        admissible = library.query_admissible(
            current_weight=current_weight,
            support_directions=support_directions,
            support_values=support_values,
        )
        return any(entry.skill_id == skill_id for entry in admissible)

    # ── Internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _validate_simplex(w: np.ndarray) -> None:
        """Raise ValueError if *w* is not a valid simplex vector."""
        if w.ndim != 1 or len(w) == 0:
            raise ValueError(f"Weight must be a 1-D non-empty array, got shape {w.shape}")
        if np.any(w < 0.0):
            raise ValueError(f"Weight components must be >= 0, got {w}")
        if not np.isclose(w.sum(), 1.0, atol=1e-6):
            raise ValueError(
                f"Weight components must sum to 1, got sum={w.sum():.6f}"
            )

    @staticmethod
    def _compute_h_wx(
        neg_delta_n: np.ndarray,
        directions: np.ndarray,
        values: np.ndarray,
    ) -> float:
        """Compute h_Wx(-delta_n): worst-case motive cost over W_x.

        For the 2-objective case:
            Given support constraints  u_j · w <= h_j  and  sum(w) == 1,
            the feasible weight set W_x is an interval whose vertices are:
                v1 = [h[0],  1 - h[0]]
                v2 = [1 - h[1],  h[1]]
            h_Wx(-delta_n) = max over vertices of  v · (-delta_n).

        For future 3+ objective cases, this would need a proper support-function
        or linear-feasibility calculation (LP).
        """
        num_objectives = len(neg_delta_n)

        if num_objectives == 2 and len(values) == 2:
            # Derive interval vertices from support constraints.
            v1 = np.array([values[0], 1.0 - values[0]], dtype=np.float64)
            v2 = np.array([1.0 - values[1], values[1]], dtype=np.float64)

            # Worst-case cost is the max dot product across vertices.
            score_1 = float(np.dot(v1, neg_delta_n))
            score_2 = float(np.dot(v2, neg_delta_n))
            return max(score_1, score_2)

        # General case (3+ objectives): would require an LP solver.
        # For now, fall back to evaluating at the vertices implied by each
        # direction individually.
        raise NotImplementedError(
            f"h_Wx computation for {num_objectives} objectives is not yet "
            f"implemented. Requires a linear-feasibility solver."
        )
