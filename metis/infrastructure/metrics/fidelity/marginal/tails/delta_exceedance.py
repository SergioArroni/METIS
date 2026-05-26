"""Delta Exceedance metric for tail distribution comparison."""

import numpy as np
import pandas as pd

from metis.infrastructure.metrics.registry import register

from ...fidelity_base import NumericColumnMetric as TailMetric


@register("fidelity.delta_exceedance")
class DeltaExceedanceMetric(TailMetric):
    """
    Delta Exceedance probability metric.

    Compares the probability of exceeding certain thresholds (quantiles)
    between real and synthetic distributions. Focuses on tail behavior
    by examining exceedance probabilities at multiple quantile levels.

    Computes the average absolute difference in exceedance probabilities
    at the 90th, 95th, and 99th percentiles of the real data.
    """

    name: str = "delta_exceedance"
    is_distance: bool = True  # Lower difference = better
    quantiles: tuple = (0.90, 0.95, 0.99)  # Quantile levels to check

    def _compute_column(self, real_col: pd.Series, synth_col: pd.Series) -> float:
        """
        Compute delta exceedance for a single column.

        For each quantile level, computes the difference in exceedance
        probability between real and synthetic data.

        Args:
            real_col: Column from real dataset
            synth_col: Column from synthetic dataset

        Returns:
            Average absolute difference in exceedance probabilities
        """
        real_values = real_col.values
        synth_values = synth_col.values

        deltas = []
        for q in self.quantiles:
            # Threshold from real data
            threshold = np.percentile(real_values, q * 100)

            # Exceedance probability in real data
            p_real = np.mean(real_values > threshold)

            # Exceedance probability in synthetic data
            p_synth = np.mean(synth_values > threshold)

            # Absolute difference
            delta = abs(p_real - p_synth)
            deltas.append(delta)

        # Also check lower tail (below 1st, 5th, 10th percentiles)
        lower_quantiles = (0.01, 0.05, 0.10)
        for q in lower_quantiles:
            threshold = np.percentile(real_values, q * 100)
            p_real = np.mean(real_values < threshold)
            p_synth = np.mean(synth_values < threshold)
            delta = abs(p_real - p_synth)
            deltas.append(delta)

        return float(np.mean(deltas))
