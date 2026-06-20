# Zero-Shot Reuse Protocol

## 1. Overview

Zero-shot reuse is the core SubRep promise: **a certified skill can be safely deployed under new motive weights without retraining.** This protocol defines how that claim is validated mathematically and, secondarily, empirically.

## 2. Two Reuse Modes

| Mode | Constant | When It Applies |
|---|---|---|
| **Full-Simplex** | `FULL_SIMPLEX` | Skill was certified over the entire simplex — safe under *any* valid weight. |
| **MDN/Contextual** | `MDN_WX` | Skill was certified within a context-conditioned weight set W_x learned by the MDN. |

## 3. Mathematical Safety is Primary

Mathematical verification is the **primary guarantee** of reuse safety. If `is_safe_mathematically()` returns `True`, the skill is provably safe to deploy.

Empirical performance checks (`evaluate_performance()`) are **secondary validation** — they provide supporting evidence but are not required for safety.

## 4. Full-Simplex Reuse

A full-simplex certificate means the skill was proven safe across **every** combination of motive weights. To reuse:

1. Provide a new weight vector `w`.
2. Validate that `w` is a legal simplex point (all components ≥ 0, sum = 1).
3. If valid → **safe to reuse.** No support values are needed.

## 5. MDN/Contextual Reuse

An MDN/contextual certificate means the skill's safety depends on the **current context's learned motive geometry**, described by support directions and support values.

**Important:** Support values are threshold constraints, **not weight vectors**. They do not need to sum to 1. For example, `[0.8, 0.4]` is a valid support values vector even though `0.8 + 0.4 = 1.2`.

To validate reuse:
1. Obtain `support_directions` and `support_values` from the current context.
2. Compute the worst-case motive cost `h_Wx(-Δn)` (see Section 6).
3. **CDS:** Safe if `Δr ≥ h_Wx(-Δn)`.
4. **PDS:** Safe if `Δr ≥ h_Wx(-Δn) - ε`.

## 6. Computing h_Wx(-Δn)

`h_Wx(-Δn)` is the **worst-case motive cost** a skill can incur over the admissible weight region W_x.

Formally: `h_Wx(-Δn) = max_{w ∈ W_x} w · (-Δn)`

The weight region W_x is described by support constraints: each pair `(u_j, h_j)` gives `u_j · w ≤ h_j`.

### 2-Objective Example

**Support descriptor:**
- Directions: `[1,0]`, `[0,1]`
- Values: `[0.8, 0.4]`

**Deriving W_x:**
- Constraint 1: `w[0] ≤ 0.8`
- Constraint 2: `w[1] ≤ 0.4`
- Combined with `w[0] + w[1] = 1`:
  - Vertex 1: `w = [0.8, 0.2]`
  - Vertex 2: `w = [0.6, 0.4]`

**Example skill:** `Δn = [-0.2, 0.1]`, so `-Δn = [0.2, -0.1]`

| Vertex | Dot with -Δn | Value |
|---|---|---|
| `[0.8, 0.2]` | `0.8×0.2 + 0.2×(-0.1)` | `0.14` |
| `[0.6, 0.4]` | `0.6×0.2 + 0.4×(-0.1)` | `0.08` |

**Result:** `h_Wx(-Δn) = max(0.14, 0.08) = 0.14`

- **CDS** passes if `Δr ≥ 0.14`
- **PDS** passes if `Δr ≥ 0.14 - ε`

## 7. Future 3+ Objective Scaling

The current implementation handles the **2-objective** case by deriving interval endpoints algebraically. For **3 or more objectives**, the weight region W_x becomes a higher-dimensional polytope, and computing `h_Wx(-Δn)` will require a proper **support-function evaluation** or **linear-feasibility solver** (e.g., linear programming) rather than simple interval arithmetic.

The current architecture is designed to make this extension straightforward — the `_compute_h_wx` method is isolated and can be replaced with an LP-based implementation when needed.

## 8. Independence from SkillLibrary

This protocol is **intentionally independent** of `SkillLibrary` and its runtime selection logic. The zero-shot evaluator validates reuse safety as a standalone mathematical check, decoupled from how skills are stored or selected at runtime. This separation ensures:
- Clear responsibility boundaries between certification and selection.
- The evaluator can be used by any downstream consumer (tests, demos, future selectors).
- No risk of circular dependencies with the library's admission logic.

## 9. Runtime Integration

While the evaluator remains a standalone mathematical tool, the SubRep runtime provides a unified entry point via `SkillLibrary.query_admissible()`. This unified method handles both globally and contextually certified skills simultaneously:
- **`FULL_SIMPLEX`** skills are automatically returned as admissible (they are universally safe).
- **`MDN_WX`** skills are processed through the $h_{W_x}(-\Delta n)$ validation logic using the context geometry passed at runtime.
