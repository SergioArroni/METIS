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

    The ISE has units of 1/x (squared density times dx); it is multiplied by
    the real column's standard deviation, which makes it the ISE of the
    standardised densities, before the absolute normalization of the central
    registry.
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

        # Constant columns have a Dirac density: the ISE against anything else
        # is infinite. It used to return |mean difference|, so a synthetic
        # column collapsed onto the real mean scored a perfect 1.
        real_const = np.ptp(real_values) == 0
        synth_const = np.ptp(synth_values) == 0
        if real_const or synth_const:
            same = real_const and synth_const and real_values[0] == synth_values[0]
            return 0.0 if same else float("inf")

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
            # KDE fitting can fail for degenerate data: the value is unknown,
            # not perfect.
            return float("nan")

    def _scale_free_value(self, raw_value: float, real_col: pd.Series) -> float:
        """ISE of the densities of the standardised variable (ISE · σ_real)."""
        sigma = float(real_col.std())
        if not sigma > 1e-12:
            return 0.0 if raw_value <= 1e-12 else float("inf")
        return raw_value * sigma
