"""Shannon Entropy Delta metric for coverage distribution comparison."""

import pandas as pd
from scipy import stats

from metis.infrastructure.metrics.registry import register
from metis.shared.distributions import align_distributions

from ...fidelity_base import UniversalColumnMetric as UniversalMarginalMetric


@register("fidelity.entropy_delta")
class ShannonEntropyDeltaMetric(UniversalMarginalMetric):
    """
    Delta of Shannon Entropy between distributions.

    Measures the relative difference in Shannon entropy (information content)
    between real and synthetic distributions.

    Shannon entropy: H(P) = -sum(p * log(p))

    High entropy indicates more uniform/diverse distribution.
    Low entropy indicates more concentrated distribution.

    Works for both categorical and numeric columns.
    """

    name: str = "entropy_delta"
    is_distance: bool = True

    def _compute_column(self, real_col: pd.Series, synth_col: pd.Series) -> float:
        """
        Compute Shannon Entropy Delta for a single column.

        Args:
            real_col: Column from real dataset
            synth_col: Column from synthetic dataset

        Returns:
            Relative entropy difference in [0, ∞), lower is better
        """
        p, q = align_distributions(real_col, synth_col)

        # Shannon entropy: -sum(p * log(p))
        h_real = stats.entropy(p)
        h_synth = stats.entropy(q)

        # Relative entropy difference
        if h_real > 1e-10:
            return abs(h_real - h_synth) / h_real
        return abs(h_real - h_synth)
