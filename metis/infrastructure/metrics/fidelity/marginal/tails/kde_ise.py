"""KDE Integrated Squared Error metric for tail distribution comparison."""

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from metis.infrastructure.metrics.registry import register

from ...fidelity_base import NumericColumnMetric as TailMetric


@register("fidelity.kde_ise")
class KDEISEMetric(TailMetric):
    """
    Kernel Density Estimation Integrated Squared Error.

    Measures the integrated squared difference between KDE estimates
    of real and synthetic distributions. Provides a smooth comparison
    that captures differences across the entire distribution including tails.

    ISE = ∫(f_real(x) - f_synth(x))² dx

    The ISE is in [0, ∞) and will be normalized using robust percentiles.
    """

    name: str = "kde_ise"
    is_distance: bool = True  # Lower ISE = better
    n_points: int = 1000  # Number of points for numerical integration

    def _compute_column(self, real_col: pd.Series, synth_col: pd.Series) -> float:
        """
        Compute KDE-ISE for a single column.

        Args:
            real_col: Column from real dataset
            synth_col: Column from synthetic dataset

        Returns:
            Integrated Squared Error between KDEs
        """
        real_values = real_col.values
        synth_values = synth_col.values

        # Check for sufficient variance
        if np.std(real_values) < 1e-10 or np.std(synth_values) < 1e-10:
            # Near-constant columns: compare means
            return abs(np.mean(real_values) - np.mean(synth_values))

        try:
            # Fit KDEs
            kde_real = gaussian_kde(real_values)
            kde_synth = gaussian_kde(synth_values)

            # Determine evaluation range (cover both distributions)
            x_min = min(np.min(real_values), np.min(synth_values))
            x_max = max(np.max(real_values), np.max(synth_values))

            # Add margin for tails
            margin = 0.1 * (x_max - x_min)
            x_min -= margin
            x_max += margin

            # Evaluation points
            x = np.linspace(x_min, x_max, self.n_points)

            # Evaluate KDEs
            f_real = kde_real(x)
            f_synth = kde_synth(x)

            # Numerical integration of squared difference
            dx = x[1] - x[0]
            ise = np.sum((f_real - f_synth) ** 2) * dx

            return float(ise)

        except np.linalg.LinAlgError:
            # KDE fitting can fail for degenerate data
            return 0.0
