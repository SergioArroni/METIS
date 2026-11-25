"""Anderson-Darling discrepancy for tail-weighted distribution comparison."""

import math

import numpy as np
import pandas as pd

from metis.infrastructure.metrics.registry import register

from ...fidelity_base import NumericColumnMetric as TailMetric

# Largest value of the population discrepancy (two distributions with disjoint
# supports, equal-weight mixture H): 2 * integral_0^1 2F / (2 - F) dF.
AD_MAX_DISCREPANCY = 4.0 * (2.0 * math.log(2.0) - 1.0)  # ~1.545


def ad_discrepancy(x: np.ndarray, y: np.ndarray) -> float:
    """Bias-corrected Anderson-Darling discrepancy of two samples, in ``[0, 1]``.

    The two-sample Anderson-Darling statistic (Pettitt 1976; Scholz & Stephens
    1987) is ``A^2 = (nm/N) * D_hat``, with ``D_hat`` the plug-in estimate of the
    population discrepancy::

        D(F, G) = integral (F - G)^2 / (H (1 - H)) dH

    a Cramér-von Mises distance whose weight ``1 / (H (1 - H))`` emphasises the
    tails (Anderson & Darling 1952). ``A^2`` is a *test statistic*: for a fixed
    difference between F and G it grows linearly with the sample size, so a
    score built on it (as before: ``exp(-A2_std / 5)``) went from 0.82 to 0.003
    for the same shift when n went from 100 to 2000. ``D`` is the effect size
    behind it and does not depend on n.

    Estimator (explicit derivation, not taken from a single reference):

    * ``H = (F + G) / 2``, the equal-weight mixture, so that ``D`` does not
      depend on the ratio of the two sample sizes either.
    * Ties use mid-values (``F(z-) + f/2n``) and the tie term of the midrank
      statistic of Scholz & Stephens (1987, eq. 7): denominator
      ``H (1 - H) - dH / 4``.
    * Under ``F = G``, ``E[A^2] = k - 1 = 1`` (Scholz & Stephens 1987), i.e.
      ``E[D_hat] ~ 1/n + 1/m``; it is subtracted (floored at 0).
    * Divided by its maximum ``4 (2 ln 2 - 1)`` (disjoint supports), so 0 means
      "same distribution" and 1 "no overlap at all".

    References:
        Anderson, T. W. & Darling, D. A. (1952). Asymptotic theory of certain
        "goodness of fit" criteria based on stochastic processes. Annals of
        Mathematical Statistics, 23(2), 193-212.
        Pettitt, A. N. (1976). A two-sample Anderson-Darling rank statistic.
        Biometrika, 63(1), 161-168.
        Scholz, F. W. & Stephens, M. A. (1987). K-sample Anderson-Darling
        tests. Journal of the American Statistical Association, 82(399),
        918-924.
    """
    x = np.sort(np.asarray(x, dtype=float))
    y = np.sort(np.asarray(y, dtype=float))
    n, m = x.size, y.size
    if n == 0 or m == 0:
        return float("nan")
    z = np.unique(np.concatenate([x, y]))
    xl, xr = x.searchsorted(z, "left"), x.searchsorted(z, "right")
    yl, yr = y.searchsorted(z, "left"), y.searchsorted(z, "right")
    f = (xl + xr) / (2.0 * n)
    g = (yl + yr) / (2.0 * m)
    dh = ((xr - xl) / n + (yr - yl) / m) / 2.0
    h = (f + g) / 2.0
    den = h * (1.0 - h) - dh / 4.0
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(den > 0, dh * (f - g) ** 2 / den, 0.0)
    d = max(0.0, float(terms.sum()) - (1.0 / n + 1.0 / m))
    return min(d / AD_MAX_DISCREPANCY, 1.0)


@register("fidelity.anderson_darling")
class AndersonDarlingMetric(TailMetric):
    """
    Anderson-Darling discrepancy (tail-weighted), as an effect size.

    The raw value per column is :func:`ad_discrepancy`: the population quantity
    that the two-sample Anderson-Darling test estimates, corrected for its
    sampling bias and divided by its maximum. It is in ``[0, 1]`` (0 = same
    distribution), does not depend on n nor on the real/synthetic size ratio,
    and is normalized as a bounded distance (``1 - value``).
    """

    name: str = "anderson_darling"
    is_distance: bool = True  # Higher discrepancy = more different = worse

    def _compute_column(self, real_col: pd.Series, synth_col: pd.Series) -> float:
        """Bias-corrected AD discrepancy of one column, in ``[0, 1]``."""
        return ad_discrepancy(real_col.to_numpy(dtype=float), synth_col.to_numpy(dtype=float))
