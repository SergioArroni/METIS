"""Binning fitted on the real column, shared by the binned divergences.

Used by PSI, KL, TVD, JS (coverage) and Hellinger (tails) on continuous
columns. The grid depends only on the real data, so the synthetic table can
never move the edges.
"""

import numpy as np
import pandas as pd


def is_continuous(col: pd.Series) -> bool:
    """Same rule as ``align_distributions``: numeric with >= 20 distinct values."""
    return pd.api.types.is_numeric_dtype(col) and col.nunique() >= 20


def quantile_bin_distributions(
    real_col: pd.Series, synth_col: pd.Series, n_bins: int = 10, eps: float = 1e-4
) -> tuple[np.ndarray, np.ndarray]:
    """Distributions of both columns on the quantile bins of the real column.

    Edges are the interior ``n_bins``-quantiles of the real data (outer bins
    open-ended), so both tables are measured on the same grid fitted on the
    reference. The previous grid, 50 equal-width bins over the *pooled* data,
    had two problems with a few hundred rows: near-empty tail bins (the
    plug-in bias alone gave PSI ~0.3, KL ~0.2-0.3 and TVD ~0.18 for data of the
    same process, and it shrank with n), and edges set by the synthetic table
    (one synthetic outlier stretched the grid until both tables fell in the
    same bin and the divergence went to ~0). ``eps`` is the usual PSI floor
    for empty bins.
    """
    real = real_col.to_numpy(dtype=float)
    synth = synth_col.to_numpy(dtype=float)
    edges = np.unique(np.quantile(real, np.linspace(0, 1, n_bins + 1)[1:-1]))
    p = np.bincount(np.searchsorted(edges, real, side="right"), minlength=len(edges) + 1)
    q = np.bincount(np.searchsorted(edges, synth, side="right"), minlength=len(edges) + 1)
    p = np.clip(p / p.sum(), eps, 1)
    q = np.clip(q / q.sum(), eps, 1)
    return p / p.sum(), q / q.sum()
