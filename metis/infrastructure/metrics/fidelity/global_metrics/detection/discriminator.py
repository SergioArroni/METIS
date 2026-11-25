"""
Discriminator: can a classifier tell real rows from synthetic ones?

Classifier two-sample test (Lopez-Paz and Oquab, 2017, ICLR; the
"detection" metric of SDMetrics and synthcity).  The real and the synthetic
table, at the same number of rows, are stacked with a 0/1 label and a
CatBoost classifier is scored by ROC AUC in stratified ``N_FOLDS``-fold
cross-validation.  Indistinguishable tables give AUC 0.5; the score is
``clip(2·(1 − AUC), 0, 1)`` (1 = indistinguishable, 0 = perfectly separable).

What it adds to the other fidelity metrics (study of 2026-10-30 over the
defect bank of the validator comparison, ``doctorado.tex`` §8): every other
metric looks at one column, one pair or a distance between rows; a
classifier sees any combination of columns, whether the record as a whole
is plausible.  It detected, with no good table flagged, rows whose columns
were drawn from different rows (rrhh), extreme tails, 30 % label noise and
the "broken trio" defect of adult and telco, none of which the fidelity
score saw.  Measured afterwards (§26), the three-way interaction of those
trios is not significant in the real tables: what the classifier sees is
the resampled column coming loose from the rest of the row.

Subcategory ``detection`` of ``fidelity.global`` (§26), next to the
distances between the joint distributions: global measures relations among
more than two variables, conditional the pairs.

Gaps: numeric NaN stays NaN (CatBoost handles it natively), a categorical
gap is its own level, so a different missingness pattern also separates.
Ordinal columns enter by their categorical twin (``_get_all_columns``).
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from metis.infrastructure.metrics.base import _env_int
from metis.shared.normalization import normalize_metric_value

from ....registry import register
from ...fidelity_base import GlobalFidelityMetric

N_FOLDS = 3
# Trees per fold.  Fixed, no early stopping: the folds are scored on rows the
# model never saw, and 300 trees separated every defect of the study.
ITERATIONS = 300
# As in the utility trainer: one-hot up to 10 levels instead of CTR.
ONE_HOT_MAX_SIZE = 10
# Columns reported as the ones that give the synthetic table away.
TOP_COLUMNS = 3


@register("fidelity.discriminator")
class DiscriminatorMetric(GlobalFidelityMetric):
    """``2·(1 − AUC)`` of a CatBoost classifier separating real from synthetic rows."""

    name: str = "discriminator"

    def __init__(self, max_samples: int = 5000):
        """
        Args:
            max_samples: Rows per table at most (both tables are cut to the
                size of the smaller one, then to this).
        """
        super().__init__()
        self.max_samples = max_samples

    @staticmethod
    def _threads() -> int:
        # Same budget as the utility trainer (METIS_N_JOBS), else CatBoost's default.
        return max(1, _env_int("METIS_N_JOBS", 1)) if "METIS_N_JOBS" in os.environ else -1

    def _compute_global(self) -> tuple[float, float, dict]:
        cols = self._get_all_columns()
        if not cols:
            raise ValueError("No common columns")
        real, synth = self._real_data[cols], self._synth_data[cols]
        n = min(len(real), len(synth), self.max_samples)
        if n < 2 * N_FOLDS:
            raise ValueError(f"Too few rows to cross-validate ({n} per table)")
        rng = np.random.default_rng(self.seed)
        if len(real) > n:
            real = real.iloc[rng.choice(len(real), n, replace=False)]
        if len(synth) > n:
            synth = synth.iloc[rng.choice(len(synth), n, replace=False)]

        X = pd.concat([real, synth], ignore_index=True)
        cat_cols = [c for c in cols if not pd.api.types.is_numeric_dtype(X[c])]
        for c in cat_cols:
            X[c] = X[c].astype(object).where(X[c].notna(), "missing").astype(str)
        y = np.r_[np.zeros(len(real)), np.ones(len(synth))]

        aucs: list[float] = []
        importance = np.zeros(len(cols))
        folds = StratifiedKFold(N_FOLDS, shuffle=True, random_state=self.seed)
        for train, test in folds.split(X, y):
            model = CatBoostClassifier(
                iterations=ITERATIONS,
                one_hot_max_size=ONE_HOT_MAX_SIZE,
                random_seed=self.seed,
                verbose=False,
                thread_count=self._threads(),
                allow_writing_files=False,
                cat_features=cat_cols,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X.iloc[train], y[train])
            aucs.append(float(roc_auc_score(y[test], model.predict_proba(X.iloc[test])[:, 1])))
            importance += np.asarray(model.get_feature_importance())

        auc = float(np.mean(aucs))
        score = normalize_metric_value(self.metric_id, float(np.clip(2 * (1 - auc), 0, 1)))
        order = np.argsort(-importance)[:TOP_COLUMNS]
        details = {
            "auc": auc,
            "auc_per_fold": aucs,
            "formula": "clip(2 * (1 - AUC), 0, 1); AUC of CatBoost real-vs-synthetic, "
            f"{N_FOLDS}-fold stratified cross-validation",
            "n_rows_per_table": n,
            "n_features": len(cols),
            "top_columns": [str(cols[i]) for i in order],
        }
        return auc, score, details
