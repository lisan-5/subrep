from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Optional
from certification.certificate_schema import Certificate

@dataclass
class SkillEntry:
    """ Runtime record for a single admitted skill in the Skill Library. """

    skill_id: str
    gate_type: str
    certificate: Certificate
    policy: Optional[Callable] = field(default=None, repr=False) # runtime only, can't be serialized
    executions: int = 0
    success_rate: float = 0.0
    avg_payoff: float = 0.0

    def __post_init__(self) -> None:
        """post-init validation to ensure gate_type is valid and matches the certificate."""
        valid_gates = {"CDS", "PDS"}
        if self.gate_type not in valid_gates:
            raise ValueError(
                f"gate_type must be one of {valid_gates}, got '{self.gate_type}'"
            )
        if self.gate_type != self.certificate.gate_type:
            raise ValueError(
                f"gate_type '{self.gate_type}' does not match "
                f"certificate gate_type '{self.certificate.gate_type}'"
            )

    @property
    def delta_r(self) -> float:
        """Scalar payoff improvement from the certificate."""
        return self.certificate.delta_r

    @property
    def delta_n(self) -> tuple[float, float]:
        """Motive improvement vector from the certificate."""
        return self.certificate.delta_n

    @property
    def admission_margin(self) -> float:
        """Admission margin from the certification gate."""
        return self.certificate.admission_margin

    @property
    def epsilon(self) -> float:
        """PDS epsilon budget (0.0 for CDS skills)."""
        return self.certificate.epsilon

    def to_dict(self) -> dict:
        """ 
        Convert to a JSON-safe dictionary.

        The `policy` field is intentionally excluded — callables cannot be
        serialized to JSON.  After loading, the caller must re-register
        policies via SkillLibrary.register_policy()
        """
        return {
            "skill_id": self.skill_id,
            "gate_type": self.gate_type,
            "certificate": self.certificate.to_dict(),
            "executions": int(self.executions),
            "success_rate": float(self.success_rate),
            "avg_payoff": float(self.avg_payoff),
        }

    @classmethod
    def from_dict(cls, data: dict) -> SkillEntry:
        """ Reconstruct a SkillEntry from a JSON-loaded dictionary. """
        certificate = Certificate.from_dict(data["certificate"])
        return cls(
            skill_id=data["skill_id"],
            gate_type=data["gate_type"],
            certificate=certificate,
            policy=None, 
            executions=int(data.get("executions", 0)),
            success_rate=float(data.get("success_rate", 0.0)),
            avg_payoff=float(data.get("avg_payoff", 0.0)),
        )