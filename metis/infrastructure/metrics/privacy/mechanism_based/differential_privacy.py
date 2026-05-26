"""
Differential Privacy (DP) Verification Metric.

Verifies or estimates the differential privacy guarantees of a
generative model based on its reported epsilon value.

Score is normalized to [0, 1] where 1 = most private (low epsilon).
"""

from typing import Any

import numpy as np

from metis.domain.entities import MetricResult
from metis.infrastructure.metrics.registry import register

from ..privacy_base import MechanismPrivacyMetric


@register("privacy.differential_privacy")
class DifferentialPrivacyMetric(MechanismPrivacyMetric):
    """
    Differential Privacy (DP) Verification Metric.

    This metric verifies the differential privacy guarantees of a
    generative model. It works in two modes:

    1. Model-reported epsilon: If the generative model exposes an
       `epsilon` attribute, we use that value directly.

    2. Hardcoded epsilon: Uses the configured epsilon value
       (default: 1.0) when model doesn't expose epsilon.

    The privacy score is computed based on the epsilon value:
    - Lower epsilon = stronger privacy guarantee = higher score
    - epsilon = 0: Perfect privacy (impossible in practice) → Score = 1.0
    - epsilon = ∞: No privacy guarantee → Score = 0.0

    Process:
    1. Extract or use configured epsilon value
    2. Optionally verify DP guarantee empirically (future work)
    3. Convert epsilon to normalized privacy score

    Interpretation:
        - ε ≤ 0.1: Very strong privacy → Score ≥ 0.9
        - ε ≤ 1.0: Good privacy → Score ≥ 0.5
        - ε ≤ 10: Weak privacy → Score ≥ 0.1
        - ε > 10: Minimal privacy → Score < 0.1

    References:
        - Dwork & Roth (2014): The Algorithmic Foundations of Differential Privacy
        - Abadi et al. (2016): Deep Learning with Differential Privacy
    """

    name: str = "differential_privacy"
    purpose_tags: set = {
        "privacy",
        "mechanism_based",
        "differential_privacy",
        "epsilon",
    }

    def __init__(
        self,
        epsilon: float | None = None,
        delta: float | None = None,
        verify_empirically: bool = False,
    ):
        """
        Initialize DP metric.

        Args:
            epsilon: Privacy budget. If None, uses DEFAULT_EPSILON (1.0) or model's epsilon.
            delta: Relaxation parameter for (ε,δ)-DP. If None, assumes pure ε-DP.
            verify_empirically: Whether to attempt empirical verification (future work).
        """
        super().__init__(epsilon=epsilon)
        self._delta = delta
        self._verify_empirically = verify_empirically

    @property
    def delta(self) -> float | None:
        """Get the delta parameter."""
        return self._delta

    def _epsilon_to_score(self, epsilon: float) -> float:
        """
        Convert epsilon to a privacy score in [0, 1].

        Uses an exponential decay function:
        score = exp(-epsilon / scale)

        where scale controls how quickly the score decreases.
        With scale=2: ε=1 → score≈0.6, ε=2 → score≈0.37
        """
        if epsilon <= 0:
            return 1.0
        if epsilon >= 100:
            return 0.0

        # Exponential decay with scale factor
        # Chosen so that: ε=0.1→0.95, ε=1→0.61, ε=5→0.08
        scale = 2.0
        score = np.exp(-epsilon / scale)
        return float(max(0.0, min(1.0, score)))

    def _get_effective_epsilon(self) -> tuple:
        """
        Get the effective epsilon value and its source.

        Returns:
            tuple of (epsilon_value, source_description)
        """
        # Priority 1: Model's epsilon attribute
        if self._generative_model is not None:
            model_epsilon = getattr(self._generative_model, "epsilon", None)
            if model_epsilon is not None:
                return float(model_epsilon), "model_attribute"

            # Check for privacy_budget or similar attributes
            for attr_name in ["privacy_budget", "eps", "privacy_epsilon"]:
                attr_val = getattr(self._generative_model, attr_name, None)
                if attr_val is not None:
                    return float(attr_val), f"model_{attr_name}"

        # Priority 2: Configured epsilon
        return self._epsilon, "configured"

    def _get_model_info(self) -> dict[str, Any]:
        """Extract information about the generative model."""
        if self._generative_model is None:
            return {"available": False}

        info = {"available": True}

        # Try to get model type/name
        model_type = type(self._generative_model).__name__
        info["model_type"] = model_type

        # Check for common DP-related attributes
        dp_attributes = [
            "epsilon",
            "delta",
            "privacy_budget",
            "noise_multiplier",
            "max_grad_norm",
        ]
        for attr in dp_attributes:
            val = getattr(self._generative_model, attr, None)
            if val is not None:
                info[attr] = val

        return info

    def compute(self) -> MetricResult:
        """
        Compute Differential Privacy metric.

        Returns:
            MetricResult with privacy score in [0, 1] where 1 = most private
        """
        try:
            # Get effective epsilon
            epsilon, epsilon_source = self._get_effective_epsilon()

            # Get model information
            model_info = self._get_model_info()

            # Compute privacy score from epsilon
            privacy_score = self._epsilon_to_score(epsilon)

            # Get delta if available
            effective_delta = self._delta
            if self._generative_model is not None:
                model_delta = getattr(self._generative_model, "delta", None)
                if model_delta is not None:
                    effective_delta = model_delta

            # Adjust score if delta is significant
            # For (ε,δ)-DP, higher delta weakens the guarantee
            if effective_delta is not None and effective_delta > 0:
                # Penalize high delta
                delta_penalty = min(0.2, effective_delta * 100)  # Cap penalty at 0.2
                privacy_score = max(0.0, privacy_score - delta_penalty)

            # Build details
            details = {
                "epsilon": float(epsilon),
                "epsilon_source": epsilon_source,
                "delta": effective_delta,
                "privacy_score": float(privacy_score),
                "dp_type": "(ε,δ)-DP" if effective_delta else "ε-DP",
                "model_info": model_info,
                "score_formula": "exp(-ε/2)",
                "default_epsilon_used": epsilon_source == "configured"
                and epsilon == self.DEFAULT_EPSILON,
                "interpretation": self._interpret_score(epsilon, privacy_score),
                "epsilon_scale": {
                    "very_strong": "ε ≤ 0.1",
                    "strong": "0.1 < ε ≤ 1",
                    "moderate": "1 < ε ≤ 5",
                    "weak": "5 < ε ≤ 10",
                    "minimal": "ε > 10",
                },
            }

            # Add warning if using default epsilon
            if details["default_epsilon_used"]:
                details["warning"] = (
                    f"Using default epsilon={self.DEFAULT_EPSILON}. "
                    "For accurate results, provide epsilon via model attribute or configuration."
                )

            return MetricResult(
                id="privacy.differential_privacy",
                value=float(privacy_score),
                details=details,
                family=self.family,
                purpose_tags=self.purpose_tags,
            )

        except Exception as e:
            return self._create_error_result(f"DP computation failed: {str(e)}")

    def _create_error_result(self, error_msg: str) -> MetricResult:
        """Create a MetricResult for error cases (NaN so aggregation skips it)."""
        return MetricResult(
            id="privacy.differential_privacy",
            value=float("nan"),
            details={"error": error_msg},
            family=self.family,
            purpose_tags=self.purpose_tags,
        )

    def _interpret_score(self, epsilon: float, score: float) -> str:
        """Provide human-readable interpretation of the DP guarantee."""
        if epsilon <= 0.1:
            return (
                f"Very strong privacy (ε={epsilon:.2f}) - excellent differential privacy guarantee"
            )
        if epsilon <= 1.0:
            return f"Strong privacy (ε={epsilon:.2f}) - good differential privacy guarantee"
        if epsilon <= 5.0:
            return f"Moderate privacy (ε={epsilon:.2f}) - reasonable privacy/utility tradeoff"
        if epsilon <= 10.0:
            return f"Weak privacy (ε={epsilon:.2f}) - limited privacy protection"
        return f"Minimal privacy (ε={epsilon:.2f}) - privacy guarantee too weak for sensitive data"
