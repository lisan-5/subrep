"""
Skill Library — runtime storage for certified SubRep skills.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from .skill_metadata import SkillEntry, FULL_SIMPLEX, MDN_WX
from utils.cone_utils import validate_simplex_weights
from certification.certificate_schema import Certificate
from certification.cds_test import CDSGate
from certification.pds_test import PDSGate


def _compute_wx_worst_case(
    delta_n: np.ndarray,
    support_directions: np.ndarray,
    support_values: np.ndarray,
) -> float:

    neg_delta_n = -np.asarray(delta_n, dtype=np.float64)
    sd = np.asarray(support_directions, dtype=np.float64)
    sv = np.asarray(support_values, dtype=np.float64)
    num_obj = len(sv)

    if num_obj != 2:
        raise ValueError(
            f"W_x vertex reconstruction requires M=2, got M={num_obj}. General-dimension support requires a linear program."
        )

    expected_shape = (num_obj, num_obj)
    if sd.shape != expected_shape:
        raise ValueError(
            f"support_directions must have shape {expected_shape}, got {sd.shape}"
        )

    basis = np.eye(num_obj, dtype=np.float64)
    if not np.allclose(sd, basis, atol=1e-6):
        raise ValueError(
            "W_x vertex reconstruction requires standard basis support directions (identity matrix), got {sd.tolist()}"
        )

    vertices = np.array([
        [sv[0], 1.0 - sv[0]],      # maximize along direction e₁
        [1.0 - sv[1], sv[1]],       # maximize along direction e₂
    ], dtype=np.float64)

    scores = vertices @ neg_delta_n
    return float(np.max(scores))


class SkillLibrary:
    """ In-memory store of certified skills """

    def __init__(self, cert_store=None, save_path: str = "data/library.json") -> None:
        self.cert_store = cert_store
        self.save_path = save_path
        self._skills: Dict[str, SkillEntry] = {}

    def add_skill(self, skill_id: str, certificate: Certificate, policy: Callable) -> bool:
        """ Add a certified skill to the library. """

        # 1. Identity & Store Check
        if skill_id != certificate.skill_id:
            return False
            
        if self.cert_store is not None:
            if not self.cert_store.contains(certificate.skill_id):
                return False

        # 2. Mathematical Check (The "Chain of Safety")
        # We re-verify the certificate's math at the library entry point.
        if certificate.gate_type == "CDS":
            gate = CDSGate()
        elif certificate.gate_type == "PDS":
            gate = PDSGate(epsilon=certificate.epsilon)
        else:
            return False

        # Ensure delta_n is numpy-ready for the gate
        delta_n_vec = np.asarray(certificate.delta_n, dtype=np.float64)
        if not gate.admit(certificate.delta_r, delta_n_vec):
            return False

        entry = SkillEntry(
            skill_id=skill_id,
            gate_type=certificate.gate_type,
            certificate=certificate,
            policy=policy,
        )

        self._skills[skill_id] = entry
        return True

    def remove_skill(self, skill_id: str) -> bool:
        """ Remove a skill from the library. """
        return self._skills.pop(skill_id, None) is not None

    def get_skill(self, skill_id: str) -> Optional[SkillEntry]:
        """ Retrieve a single skill by its unique ID. """
        return self._skills.get(skill_id)

    def get_admitted_skills(self) -> List[SkillEntry]:
        """ Return all skills currently in the library. """
        return list(self._skills.values())

    def query_by_gate_type(self, gate_type: str) -> List[SkillEntry]:
        """ Filter skills by the gate that admitted them. """
        return [s for s in self._skills.values() if s.gate_type == gate_type]

    def query_by_weights(self, weights: List[float]) -> List[SkillEntry]:
        """ 
        Return skills that are admissible under a specific weight vector. 

        For a given weight vector w, a skill is admissible if:
            Δr + w^T Δn  ≥  -ε

        where ε = 0 for CDS skills (they pass for ALL w by definition)
        and ε = certificate.epsilon for PDS skills.
        
        """
        w = np.asarray(weights, dtype=np.float64)

        if not validate_simplex_weights(w):
            raise ValueError(
                f"weights must be a valid simplex vector (non-negative, sum to 1), "
                f"got {weights}"
            )

        admissible = []
        for entry in self._skills.values():
            if entry.gate_type == "CDS":
                # CDS skills pass for all weight vectors
                admissible.append(entry)
            else:
                # PDS: check  Δr + w^T Δn ≥ -ε  for this specific w
                delta_n = np.asarray(entry.delta_n, dtype=np.float64)
                score = entry.delta_r + float(np.dot(w, delta_n))
                if score >= -entry.epsilon:
                    admissible.append(entry)

        return admissible

    def query_admissible(
        self,
        current_weight: np.ndarray,
        support_directions: Optional[np.ndarray] = None,
        support_values: Optional[np.ndarray] = None,
    ) -> List[SkillEntry]:
        """Return skills admissible under the current MDN weight and W_x region."""
        w = np.asarray(current_weight, dtype=np.float64).reshape(-1)
        if not validate_simplex_weights(w):
            raise ValueError(
                f"current_weight must be a valid simplex vector, got {current_weight}"
            )

        admissible: list[SkillEntry] = []
        for entry in self._skills.values():
            if entry.weight_region_type == FULL_SIMPLEX:
                # Globally certified
                admissible.append(entry)

            elif entry.weight_region_type == MDN_WX:
                if support_directions is None or support_values is None:
                    raise ValueError(
                        f"MDN_WX skill '{entry.skill_id}' requires current support_directions and support_values for runtime admissibility, but they were not provided."
                    )
                sd = np.asarray(support_directions, dtype=np.float64)
                sv = np.asarray(support_values, dtype=np.float64)
                delta_n = np.asarray(entry.delta_n, dtype=np.float64)

                h_wx = _compute_wx_worst_case(delta_n, sd, sv)

                if entry.gate_type == "CDS":
                    if entry.delta_r >= h_wx:
                        admissible.append(entry)
                elif entry.gate_type == "PDS":
                    if entry.delta_r >= h_wx - entry.epsilon:
                        admissible.append(entry)

        return admissible

    def count(self) -> int:
        """Return the number of skills in the library."""
        return len(self._skills)

    def register_policy(self, skill_id: str, policy: Callable) -> bool:
        """ Attach a policy to a skill that was loaded from disk. """
        entry = self._skills.get(skill_id)
        if entry is None:
            return False
        entry.policy = policy
        return True

    def save(self, path: Optional[str] = None) -> None:
        """ Save the library to a JSON file. """
        path = path or self.save_path
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": 1,
            "skill_count": self.count(),
            "skills": {
                sid: entry.to_dict()
                for sid, entry in self._skills.items()
            },
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: Optional[str] = None) -> None:
        """ Load a library from a JSON file """
        path = path or self.save_path
        filepath = Path(path)

        with open(filepath, "r") as f:
            data = json.load(f)

        self._skills = {
            sid: SkillEntry.from_dict(entry_data)
            for sid, entry_data in data["skills"].items()
        }