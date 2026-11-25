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

    Scores the differential-privacy budget the generator was trained with.
    It does not verify it empirically: it trusts the declared value.

    Where epsilon comes from, in order:

    1. ``epsilon`` configured for the metric (``metric_params``);
    2. the generative model's ``epsilon`` (or ``privacy_budget``, ``eps``,
       ``privacy_epsilon``) attribute, via ``context["generative_model"]``.

    Without either the score is NaN with ``details["error"]``: there is no
    guarantee to score. (It used to fall back to a hard-coded ε = 1 and
    report 0.61 for every table, which the privacy family then averaged as
    if it had been measured.) A non-finite or negative epsilon is an error.

    Score: ``exp(−ε/2)`` (ε ≥ 100 → 0).  δ is a validity condition, not a
    penalty (Dwork & Roth 2014, §2.3): with δ ≥ 1/n a mechanism that
    publishes a whole record satisfies (ε, δ)-DP, so the guarantee is void
    and the score is 0; with δ < 1/n the score is that of ε.  n is the
    number of training records.  Until 2026-10-19 the score subtracted
    ``min(0.2, 100·δ)``, constants without a reference: a void δ = 0.01 cost
    only 0.2 and a sound δ = 1e-5 cost 0.001.

        - ε = 0.1 → 0.95
        - ε = 1 → 0.61
        - ε = 5 → 0.08
        - ε = 10 → 0.007

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
            epsilon: Declared privacy budget; takes precedence over the model's.
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

    _MODEL_EPSILON_ATTRS = ("epsilon", "privacy_budget", "eps", "privacy_epsilon")

    def _get_effective_epsilon(self) -> tuple[float | None, str | None]:
        """Return ``(epsilon, source)``; ``(None, None)`` when nothing declares one."""
        if self._epsilon is not None:
            return float(self._epsilon), "configured"
        if self._generative_model is not None:
            for attr_name in self._MODEL_EPSILON_ATTRS:
                attr_val = getattr(self._generative_model, attr_name, None)
                if attr_val is not None:
                    source = "model_attribute" if attr_name == "epsilon" else f"model_{attr_name}"
                    return float(attr_val), source
        return None, None

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
            epsilon, epsilon_source = self._get_effective_epsilon()
            if epsilon is None:
                return self._create_not_applicable_result(
                    "No differential-privacy budget: configure "
                    "metric_params['privacy.differential_privacy'].epsilon or pass a "
                    "context['generative_model'] exposing 'epsilon'"
                )
            if np.isnan(epsilon) or epsilon < 0:
                return self._create_error_result(f"Invalid epsilon {epsilon!r} ({epsilon_source})")

            # Get model information
            model_info = self._get_model_info()

            # Compute privacy score from epsilon
            privacy_score = self._epsilon_to_score(epsilon)

            # Get delta if available
            effective_delta = self._delta
            if effective_delta is None and self._generative_model is not None:
                effective_delta = getattr(self._generative_model, "delta", None)
            if effective_delta is not None and not 0.0 <= float(effective_delta) <= 1.0:
                return self._create_error_result(f"Invalid delta {effective_delta!r}")

            # δ ≥ 1/n voids the guarantee (Dwork & Roth 2014): see the class docstring.
            n_records = len(self._real_data) if self._real_data is not None else 0
            delta_limit = 1.0 / n_records if n_records else None
            delta_valid = not (
                effective_delta is not None
                and effective_delta > 0
                and delta_limit is not None
                and effective_delta >= delta_limit
            )
            if not delta_valid:
                privacy_score = 0.0

            # Build details
            details = {
                "epsilon": float(epsilon),
                "epsilon_source": epsilon_source,
                "delta": effective_delta,
                "delta_limit_1_over_n": delta_limit,
                "delta_valid": delta_valid,
                "privacy_score": float(privacy_score),
                "dp_type": "(ε,δ)-DP" if effective_delta else "ε-DP",
                "model_info": model_info,
                "score_formula": "exp(-ε/2) if δ < 1/n else 0",
                "interpretation": self._interpret_score(epsilon, privacy_score),
                "epsilon_scale": {
                    "very_strong": "ε ≤ 0.1",
                    "strong": "0.1 < ε ≤ 1",
                    "moderate": "1 < ε ≤ 5",
                    "weak": "5 < ε ≤ 10",
                    "minimal": "ε > 10",
                },
            }

            return MetricResult(
                id="privacy.differential_privacy",
                value=float(privacy_score),
                details=details,
                family=self.family,
                purpose_tags=self.purpose_tags,
            )

        except Exception as e:
            return self._create_error_result(f"DP computation failed: {str(e)}")

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
