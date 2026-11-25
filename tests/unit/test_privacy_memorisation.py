"""Privacy regression tests: detect memorisation / leakage of training rows.

These tests assert two privacy-by-design invariants:

* When the synthetic dataset is identical to the real dataset (perfect
  memorisation), the DCR-derived privacy score MUST collapse toward 0.
  A high score in that scenario would silently bless a synthesizer that
  is overfitting to / leaking training rows.
* When synthetic rows are far from any real row (well-separated cloud),
  the score MUST be visibly higher than the memorisation case.

This guards against future regressions in the privacy aggregation that
would rebalance the score without keeping the memorisation signal
meaningful (paper review found this issue, hence the explicit test).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from metis.infrastructure.metrics.privacy.dataset_based.empirical_similarity.dcr import DCRMetric


def _train_df(n: int, dim: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(size=(n, dim)),
        columns=[f"x{i}" for i in range(dim)],
    )


def _fit_dcr(real: pd.DataFrame, synth: pd.DataFrame) -> float:
    # DCR needs non-members (2026-10-14): a holdout from the same population.
    holdout = _train_df(len(real), dim=real.shape[1], seed=99)
    metric = DCRMetric()
    metric.fit(real, synth, context={"holdout": holdout})
    result = metric.compute()
    return float(result.value)


def test_dcr_flags_perfect_memorisation():
    """Synth == Real should be visibly worse than well-separated synth."""
    real = _train_df(200, dim=4, seed=0)

    # Perfect memorisation: synth == real (worst case).
    score_memo = _fit_dcr(real, real.copy())

    # Well-separated cloud, far from training data.
    far = _train_df(200, dim=4, seed=1) + 50.0
    score_far = _fit_dcr(real, far)

    assert score_far > score_memo, (
        f"Privacy score must increase when synth moves away from train set "
        f"(memo={score_memo:.4f}, far={score_far:.4f})"
    )
    assert score_memo < 0.5, (
        f"Memorised synth (synth==real) must yield a clearly low DCR score (got {score_memo:.4f})"
    )


def test_dcr_score_in_unit_interval():
    real = _train_df(150, dim=3, seed=2)
    synth = _train_df(150, dim=3, seed=3) + 5.0
    score = _fit_dcr(real, synth)
    assert 0.0 <= score <= 1.0, f"DCR score must be in [0, 1], got {score}"
