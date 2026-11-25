"""
Unified ML Efficiency Metric.

Auto-detects classification vs regression and evaluates with configurable
strategies.  TTR (Train Real, Test Real) is computed **once** as the
baseline and reused by every strategy comparison.

Evaluation protocol
-------------------
Every strategy is scored on ``n_splits`` train/test splits (repeated 80/20
holdout, seeds ``seed .. seed + n_splits − 1``; TTR, TSTR and TTRS share the
real test rows of each split).  When the evaluation context carries a real
``holdout`` table, the strategies tested on real data use it as test set and
train on the whole real/synthetic table; the repetitions then only vary the
model seed.  CatBoost runs with its defaults and early stopping on a
validation slice of the training data of each split — never on the test rows
(see ``catboost_trainer``).

Score
-----
Absolute scale between a model without signal (0) and the real baseline (1).
Per strategy *s* and split *k*, with ``m`` the macro one-vs-rest ROC AUC
(classification; weighted F1 until 2026-10-19, see ``ClassificationTrainer``),
the MAE (regression) or the QWK (ordinal), and ``m0`` the best model that
ignores the features (``_no_signal_score``: 0.5, the median's MAE, 0)::

    gain(x) = x − m0     (classification)      gain(x) = m0 − x     (regression)

    utility_s = clip((g_TTR − l − |mean_k gain(m_s,k) − g_TTR|) / (u − l), 0, 1)
    g_TTR = mean_k gain(m_TTR,k)     u = g_TTR − z·√2·σ     l = z·σ

1 means the synthetic table trains as well as another real sample of the
same size would; 0 means it teaches nothing about the target that guessing
from the label frequencies would not.  ``σ`` is the sampling noise of the
TTR gain due to *which* real rows were drawn, not only to the model seed:
the TTR model is refitted on a random half of the real training rows of each
split and the spread of those gains, paired with the TTR gain of the same
split (the test rows, shared with the synthetic table, cancel), estimates it
(half-sampling, the
delete-d jackknife with d = n/2: Shao & Wu 1989, Ann. Stat. 17(3)).  Another
real sample's gain differs from the TTR estimate by ``√2·σ``, a model without
signal scatters ``σ`` around 0, and each anchor sits at its one-sided 95 %
quantile (``metis.shared.anchors``, the rule of the calibration): an
independent real sample scores 1 and a useless table 0.  With the anchors at
the plain means (until 2026-10-10) an unseen real third of rrhh scored 0.54
on one seed because TTR, trained on another third, happened to gain 0.08
over chance and it gained 0.04.

When the TTR gain is below ``UTILITY_SIGNAL_MULTIPLE`` (3; 5 until 2026-10-25) times that noise the real
table carries no signal measurable at its size: the metric is not
applicable (NaN, with the reason), there is nothing to preserve.  With
``n_splits = 5`` half-samples σ itself is uncertain by ~35 %, so on a table
near the threshold that decision flipped with the seed (rrhh: not applicable
on 4 of 5 seeds of the 2026-10-10 benchmark).  Everything that depends
only on the real table is therefore measured once per table
(``context["utility_signal"]``, from the calibration,
``metis.calibrate.core.utility_signal``, or from
``Evaluator.measure_utility_signal``) with 20 half-samples: the decision and
the anchors ``l`` and ``u``.  A run then trains only the strategies; it
measures TTR and its own anchors only on tables of another size than the
ones the signal was measured on (the gain depends on it), and still follows
the decision.  The previous score,
``1 − clip(mean_k(m_TTR,k − m_s,k), 0, 1)``, gave a useless synthetic table
0.65–0.95 depending on how hard the target was, because it measured the loss
in metric units and a model with no signal already scores 0.5 F1 on a
balanced binary target.  ``delta_from_ttr`` still reports that raw loss.

Symmetric (author's decision, 2026-10-19): the real baseline is the ideal,
not a mark that can be beaten, so landing above TTR costs as much as landing
the same distance below it; within ``z·√2·σ`` of ``g_TTR`` (another real
sample's noise) the score is 1.  Below TTR the formula is the one-sided one
used until then.  ``utility.ml_efficiency`` averages ``utility_s`` over the selected
strategies; by default only TSTR (the strategy that measures utility as a
substitute for real training data).  TTS and TRTS measure how learnable the
synthetic table is by itself, and TTRS dilutes the synthetic signal with the
real training data (a useless synthetic table still scores ≈ TTR), so they
are opt-in via ``ml_efficiency_config.strategies``.
"""

import hashlib
import math
import threading
from collections import OrderedDict
from typing import Any

import numpy as np
import pandas as pd

from metis.domain.entities import MetricResult
from metis.infrastructure.metrics.base import ORDINAL_SHADOW_SUFFIX, MetricBase
from metis.infrastructure.metrics.registry import register
from metis.infrastructure.metrics.utility.ml_common import (
    DEFAULT_N_SPLITS,
    NO_TARGET,
    MLMetricMixin,
)
from metis.infrastructure.metrics.utility.ml_efficiency.catboost_trainer import LOG_LOSS_EPS
from metis.infrastructure.metrics.utility.ml_efficiency.base import (
    StrategyResult,
    TrainingStrategy,
    subsample,
)
from metis.infrastructure.metrics.utility.ml_efficiency.trts import TRTSMetric
from metis.infrastructure.metrics.utility.ml_efficiency.tstr import TSTRMetric
from metis.infrastructure.metrics.utility.ml_efficiency.ttr import TTRMetric
from metis.infrastructure.metrics.utility.ml_efficiency.ttrs import TTRSMetric
from metis.infrastructure.metrics.utility.ml_efficiency.tts import TTSMetric
from metis.shared.anchors import ANCHOR_Z, UTILITY_SIGNAL_MULTIPLE

# Strategies the user can request (TTR is always the baseline, not selectable)
AVAILABLE_STRATEGIES: set[str] = {"tstr", "trts", "tts", "ttrs"}
# Strategies averaged by ``utility.ml_efficiency`` when none are configured.
DEFAULT_STRATEGIES: list[str] = ["tstr"]

_STRATEGY_METRICS: dict[TrainingStrategy, type] = {
    TrainingStrategy.TTR: TTRMetric,
    TrainingStrategy.TTS: TTSMetric,
    TrainingStrategy.TRTS: TRTSMetric,
    TrainingStrategy.TSTR: TSTRMetric,
    TrainingStrategy.TTRS: TTRSMetric,
}

# Strategies whose test rows are real: they use ``context["holdout"]`` when given.
_REAL_TEST_STRATEGIES = {TrainingStrategy.TTR, TrainingStrategy.TSTR, TrainingStrategy.TTRS}

# Synthetic labels never seen in the real target are kept as one extra class.
_UNSEEN_LABEL = "__unseen__"

_SCALE = (
    "; l = z*sigma, u = mean_k(gain_TTR,k) - z*sqrt(2)*sigma, z = 1.645, "
    "sigma = spread of the TTR gain refitted on half-samples of the real rows"
)
_FORMULAS = {
    "classification": (
        "utility_s = clip((g_TTR - l - |mean_k(AUC_s,k - 0.5) - g_TTR|) / (u - l), 0, 1); "
        "AUC = macro one-vs-rest ROC AUC of the predicted probabilities, "
        "0.5 for a model without signal" + _SCALE
    ),
    "regression": (
        "utility_s = clip((g_TTR - l - |mean_k(MAE_0,k - MAE_s,k) - g_TTR|) / (u - l), 0, 1); "
        "MAE_0 = predicting the median of the real training target" + _SCALE
    ),
    "ordinal": (
        "utility_s = clip((g_TTR - l - |mean_k(QWK_s,k) - g_TTR|) / (u - l), 0, 1); "
        "QWK = quadratic weighted kappa on the level ranks, 0 for a model without signal" + _SCALE
    ),
    "log_loss": (
        "utility_s = clip((g_TTR - l - |mean_k(LL_0,k - LL_s,k) - g_TTR|) / (u - l), 0, 1); "
        "LL = mean -log p(true class) on the real test rows (p floored at 1e-6), "
        "LL_0 = predicting the class frequencies of the real training rows" + _SCALE
    ),
}

_METRIC_NAMES = {
    "classification": "ROC AUC (macro one-vs-rest)",
    "regression": "MAE",
    "ordinal": "QWK (quadratic weighted kappa on the level ranks)",
    "log_loss": "log-loss (cross-entropy of the predicted class probabilities)",
}

# Scores where lower is better: their gain over the model without signal is
# ``m0 − m``.
_LOWER_IS_BETTER = {"regression", "log_loss"}

# Seed offset of the half-sample draws, so they do not reuse the split seeds.
_HALF_SAMPLE_SEED_OFFSET = 7919


# ── helpers ──────────────────────────────────────────────────────────────

# Key used to store the shared TTR results in the evaluation context dict.
# All utility metrics of one run read from it, so TTR is computed once per
# (reference, budget) instead of once per metric.
_TTR_CACHE_KEY = "_ttr_baseline_result"

# Process-wide TTR cache.  TTR depends only on the real table, the target,
# the features, the seed and the training budget, so callers that score many
# synthetic tables against one reference (calibration, generator search) pay
# for it once instead of once per call.  Bounded LRU; disable per run with
# ``evaluation.ml_efficiency_config.cache_ttr: false``.
_TTR_PROCESS_CACHE: OrderedDict[tuple, StrategyResult] = OrderedDict()
_TTR_PROCESS_CACHE_MAX = 16
_TTR_PROCESS_CACHE_LOCK = threading.Lock()


def clear_ttr_cache() -> None:
    """Empty the process-wide TTR cache."""
    with _TTR_PROCESS_CACHE_LOCK:
        _TTR_PROCESS_CACHE.clear()


def _digest(df: pd.DataFrame | None, cols: list[str]) -> str | None:
    if df is None:
        return None
    content = pd.util.hash_pandas_object(df[cols], index=False)
    return hashlib.sha1(content.to_numpy().tobytes()).hexdigest()


def _ttr_cache_key(
    real: pd.DataFrame,
    target_col: str | list[str],
    feature_cols: list[str],
    trainer: Any,
    seed: int,
    n_splits: int = 1,
    holdout: pd.DataFrame | None = None,
) -> tuple:
    targets = [target_col] if isinstance(target_col, str) else list(target_col)
    cols = list(feature_cols) + targets
    return (
        _digest(real, cols),
        len(real),
        tuple(cols),
        tuple(map(str, real[cols].dtypes)),
        type(trainer).__name__,
        trainer.n_runs,
        seed,
        n_splits,
        _digest(holdout, cols),
    )


def _run_extras(trainer: Any) -> dict[str, float]:
    """Mean over the refits of the last ``train_and_evaluate`` of each secondary score."""
    runs = getattr(trainer, "last_secondary", None) or []
    keys = {k for r in runs for k in r}
    return {k: float(np.mean([r[k] for r in runs if k in r])) for k in keys}


def _extras_per_split(per_split: list[dict[str, float]]) -> dict[str, list[float]]:
    """``{score: [value of split k]}``, NaN where a split lacks it (aligned with the splits)."""
    keys = sorted({k for d in per_split for k in d})
    return {k: [d.get(k, float("nan")) for d in per_split] for k in keys}


def _view(result: StrategyResult, key: str) -> StrategyResult:
    """*result* scored with its secondary score *key* (same fits, same splits)."""
    values = result.extra_scores_per_run.get(key)
    if not result.is_valid or values is None:
        return StrategyResult(
            strategy=result.strategy,
            score=float("nan"),
            raw_metric=float("nan"),
            scores_per_run=[],
            std=float("nan"),
            best_params=result.best_params,
            is_valid=False,
            error=result.error or f"no {key} score",
        )
    finite = [v for v in values if np.isfinite(v)]
    mean = float(np.mean(finite)) if finite else float("nan")
    return StrategyResult(
        strategy=result.strategy,
        score=mean,
        raw_metric=mean,
        scores_per_run=list(values),
        std=float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
        best_params=result.best_params,
        is_valid=bool(finite),
        error=result.error,
    )


def _strategy_data(
    strategy: TrainingStrategy,
    real: pd.DataFrame,
    synth: pd.DataFrame,
    target_col: str | list[str],
    feature_cols: list[str],
    *,
    seed: int,
    holdout: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Train/test data of *strategy* for one split."""
    if holdout is None or strategy not in _REAL_TEST_STRATEGIES:
        metric = _STRATEGY_METRICS[strategy]()
        return metric.prepare_data(real, synth, target_col, feature_cols, seed=seed)

    if strategy is TrainingStrategy.TTR:
        X_train, y_train = real[feature_cols], real[target_col].values
    elif strategy is TrainingStrategy.TSTR:
        X_train, y_train = synth[feature_cols], synth[target_col].values
    else:  # TTRS: real + as many synthetic rows
        X_s, y_s = subsample(
            synth[feature_cols].reset_index(drop=True), synth[target_col].values, len(real), seed
        )
        X_train = pd.concat([real[feature_cols], X_s], ignore_index=True)
        y_train = np.concatenate([real[target_col].values, y_s])
    return {
        "X_train": X_train.reset_index(drop=True),
        "X_test": holdout[feature_cols].reset_index(drop=True),
        "y_train": y_train,
        "y_test": holdout[target_col].values,
    }


def _evaluate_strategy(
    strategy: TrainingStrategy,
    real: pd.DataFrame,
    synth: pd.DataFrame,
    target_col: str | list[str],
    feature_cols: list[str],
    trainer: Any,
    seed: int,
    n_splits: int = 1,
    holdout: pd.DataFrame | None = None,
) -> StrategyResult:
    """Score *strategy* on ``n_splits`` splits.

    Each split early-stops on a validation slice of its own training data
    (nothing chosen on one split is carried to another).  ``scores_per_run``
    holds one score per split (the mean over the ``n_runs`` refits of that
    split); ``best_params`` are those settled on the first split.
    """
    try:
        best_params: dict[str, Any] = {}
        per_split: list[float] = []
        extras: list[dict[str, float]] = []
        for k in range(n_splits):
            data = _strategy_data(
                strategy, real, synth, target_col, feature_cols, seed=seed + k, holdout=holdout
            )
            args = (data["X_train"], data["X_test"], data["y_train"], data["y_test"])
            # New model seed per split: with a fixed holdout it is the only source of variation.
            params, scores, _, _ = trainer.train_and_evaluate(*args, seed=seed + k)
            if k == 0:
                best_params = params
            per_split.append(float(np.mean(scores)))
            extras.append(_run_extras(trainer))
        return StrategyResult(
            strategy=strategy,
            score=float(np.mean(per_split)),
            raw_metric=float(np.mean(per_split)),
            scores_per_run=per_split,
            std=float(np.std(per_split, ddof=1)) if len(per_split) > 1 else 0.0,
            best_params=best_params,
            is_valid=True,
            extra_scores_per_run=_extras_per_split(extras),
        )
    except Exception as e:
        return StrategyResult(
            strategy=strategy,
            score=float("nan"),
            raw_metric=float("nan"),
            scores_per_run=[],
            std=float("nan"),
            best_params={},
            is_valid=False,
            error=f"{type(e).__name__}: {e}",
        )


def _half_sample_scores(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    target_col: str | list[str],
    feature_cols: list[str],
    trainer: Any,
    seed: int,
    n_splits: int = 1,
    holdout: pd.DataFrame | None = None,
) -> StrategyResult:
    """TTR refitted on a random half of the real training rows of each split.

    Same test rows and model seed as the TTR split; the half is drawn
    without replacement.  A split whose half cannot be fitted (e.g. a class
    left out) is skipped: the spread only needs the ones that fit.
    """
    per_split: list[float] = []
    extras: list[dict[str, float]] = []
    errors: list[str] = []
    for k in range(n_splits):
        data = _strategy_data(
            TrainingStrategy.TTR,
            real,
            synth,
            target_col,
            feature_cols,
            seed=seed + k,
            holdout=holdout,
        )
        X_half, y_half = subsample(
            data["X_train"],
            np.asarray(data["y_train"]),
            len(data["X_train"]) // 2,
            seed + _HALF_SAMPLE_SEED_OFFSET + k,
        )
        try:
            _, scores, _, _ = trainer.train_and_evaluate(
                X_half, data["X_test"], y_half, data["y_test"], seed=seed + k
            )
        except Exception as e:  # noqa: BLE001 - reported, the others still count
            errors.append(f"{type(e).__name__}: {e}")
            per_split.append(float("nan"))
            extras.append({})
            continue
        per_split.append(float(np.mean(scores)))
        extras.append(_run_extras(trainer))
    finite = [v for v in per_split if np.isfinite(v)]
    mean = float(np.mean(finite)) if finite else float("nan")
    return StrategyResult(
        strategy=TrainingStrategy.TTR,
        score=mean,
        raw_metric=mean,
        scores_per_run=per_split,
        std=float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan"),
        best_params={},
        is_valid=bool(finite),
        error="; ".join(errors) or None,
        extra_scores_per_run=_extras_per_split(extras),
    )


def _get_or_compute_ttr(
    context: dict[str, Any],
    real: pd.DataFrame,
    synth: pd.DataFrame,
    target_col: str | list[str],
    feature_cols: list[str],
    trainer: Any,
    seed: int,
    n_splits: int = 1,
    holdout: pd.DataFrame | None = None,
) -> StrategyResult:
    """Return the cached TTR baseline, computing it on first call.

    Cached in ``context[_TTR_CACHE_KEY]`` (per evaluation run, keyed by
    reference and budget, so metrics with different budgets do not share a
    baseline) and in a process-wide LRU (unless ``cache_ttr: false``).
    """
    key = _ttr_cache_key(real, target_col, feature_cols, trainer, seed, n_splits, holdout)
    return _cached(
        context,
        key,
        lambda: _evaluate_strategy(
            TrainingStrategy.TTR,
            real,
            synth,
            target_col,
            feature_cols,
            trainer,
            seed,
            n_splits,
            holdout,
        ),
    )


def _get_or_compute_half_scores(
    context: dict[str, Any],
    real: pd.DataFrame,
    synth: pd.DataFrame,
    target_col: str | list[str],
    feature_cols: list[str],
    trainer: Any,
    seed: int,
    n_splits: int = 1,
    holdout: pd.DataFrame | None = None,
) -> StrategyResult:
    """``_half_sample_scores``, cached like the TTR baseline (it depends on
    the same inputs only)."""
    key = _ttr_cache_key(real, target_col, feature_cols, trainer, seed, n_splits, holdout)
    return _cached(
        context,
        key + ("half",),
        lambda: _half_sample_scores(
            real, synth, target_col, feature_cols, trainer, seed, n_splits, holdout
        ),
    )


def _cached(context: dict[str, Any], key: tuple, compute: Any) -> StrategyResult:
    run_cache: dict = context.setdefault(_TTR_CACHE_KEY, {})
    cached: StrategyResult | None = run_cache.get(key)
    if cached is not None:
        return cached

    use_process_cache = context.get("_ttr_process_cache", True)
    if use_process_cache:
        with _TTR_PROCESS_CACHE_LOCK:
            hit = _TTR_PROCESS_CACHE.get(key)
            if hit is not None:
                _TTR_PROCESS_CACHE.move_to_end(key)
        if hit is not None:
            run_cache[key] = hit
            return hit

    ttr = compute()
    run_cache[key] = ttr
    if use_process_cache and ttr.is_valid:
        with _TTR_PROCESS_CACHE_LOCK:
            _TTR_PROCESS_CACHE[key] = ttr
            while len(_TTR_PROCESS_CACHE) > _TTR_PROCESS_CACHE_MAX:
                _TTR_PROCESS_CACHE.popitem(last=False)
    return ttr


def _strategy_delta(
    result: StrategyResult,
    ttr: StrategyResult,
    task_type: str,
) -> tuple[float, float]:
    """Return ``(delta, standard error)`` of *result* against the TTR baseline.

    ``delta > 0`` means the strategy does worse than TTR.  The differences
    are paired by split (same real test rows for TSTR/TTRS).
    """
    s = np.asarray(result.scores_per_run, dtype=float)
    t = np.asarray(ttr.scores_per_run, dtype=float)
    n = min(len(s), len(t))
    s, t = s[:n], t[:n]
    if task_type == "regression":
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.where(t > 0, (s - t) / t, np.where(s > t, 1.0, 0.0))
    elif task_type == "log_loss":
        d = s - t
    else:
        d = t - s
    delta = float(np.mean(d))
    se = float(np.std(d, ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    return delta, se


def _no_signal_score(y_train: np.ndarray, y_test: np.ndarray, task_type: str) -> float:
    """Score of the best model that ignores the features.

    Classification (ROC AUC): 0.5, the AUC of any score that ignores the
    features (a constant, or one independent of the truth in expectation).
    Regression: MAE of the training median (the best constant under MAE).
    Ordinal (QWK): 0.  Log-loss: cross-entropy of the training class
    frequencies on the test labels.  Multi-target: mean over the targets, as
    ``_compute_metric`` does.
    """
    if task_type == "ordinal":
        # Kappa is chance-corrected: a constant prediction, or one independent
        # of the truth, has QWK 0 (exactly, resp. in expectation).
        return 0.0
    if task_type == "classification":
        return 0.5
    y_train, y_test = np.asarray(y_train), np.asarray(y_test)
    if task_type == "log_loss":
        # Cross-entropy of the training class frequencies (the best constant
        # under log-loss) on the test labels; an unseen class gets the floor.
        labels, counts = np.unique(y_train, return_counts=True)
        freq = dict(zip(labels.tolist(), (counts / counts.sum()).tolist()))
        p = np.array([freq.get(v, 0.0) for v in y_test.tolist()])
        return float(-np.mean(np.log(np.clip(p, LOG_LOSS_EPS, 1.0))))
    if y_train.ndim == 2:
        return float(
            np.mean(
                [
                    _no_signal_score(y_train[:, j], y_test[:, j], task_type)
                    for j in range(y_train.shape[1])
                ]
            )
        )
    if task_type == "regression":
        y_train, y_test = y_train.astype(float), y_test.astype(float)
        return float(np.mean(np.abs(y_test - np.median(y_train))))
    raise ValueError(f"unknown task type {task_type!r}")


def _chance_scores(
    strategy: TrainingStrategy,
    real: pd.DataFrame,
    synth: pd.DataFrame,
    target_col: str | list[str],
    feature_cols: list[str],
    *,
    task_type: str,
    seed: int,
    n_splits: int,
    holdout: pd.DataFrame | None = None,
) -> list[float]:
    """No-signal score per split, on the test rows of *strategy*.

    The model without signal is fitted on the real training rows of the
    split (what TTR trains on), so its score does not reward a synthetic
    table for copying the label frequencies.
    """
    out = []
    for k in range(n_splits):
        kwargs = {"seed": seed + k, "holdout": holdout}
        train = _strategy_data(
            TrainingStrategy.TTR, real, synth, target_col, feature_cols, **kwargs
        )
        test = (
            train
            if strategy is TrainingStrategy.TTR
            else _strategy_data(strategy, real, synth, target_col, feature_cols, **kwargs)
        )
        out.append(_no_signal_score(train["y_train"], test["y_test"], task_type))
    return out


def _gains(scores: list[float], chance: list[float], task_type: str) -> np.ndarray:
    """Per-split gain over the model without signal (NaN splits kept)."""
    sign = -1.0 if task_type in _LOWER_IS_BETTER else 1.0
    n = min(len(scores), len(chance))
    return sign * (np.asarray(scores[:n], dtype=float) - np.asarray(chance[:n], dtype=float))


def _utility_scale(
    ttr: StrategyResult,
    half: StrategyResult,
    chance_ttr: list[float],
    task_type: str,
) -> tuple[dict[str, Any], str | None]:
    """Anchors of the absolute scale (module docstring) and, if the real
    table has no signal measurable at its size, why the score is undefined.

    ``σ`` is the spread of the half-sample gains paired with the TTR gain of
    the same split (same test rows, so the test-set noise, which a synthetic
    table shares with TTR, cancels); with fewer than two pairs it falls back
    to the spread of the TTR gains (model seed only).
    """
    gain_ttr = _gains(ttr.scores_per_run, chance_ttr, task_type)
    gain_half = _gains(half.scores_per_run, chance_ttr, task_type)
    n = min(len(gain_ttr), len(gain_half))
    paired = gain_half[:n] - gain_ttr[:n]
    paired = paired[np.isfinite(paired)]
    gain = float(np.mean(gain_ttr))
    if len(paired) > 1:
        sigma, source = float(np.std(paired, ddof=1)), "half-samples of the real rows"
    elif len(gain_ttr) > 1:
        sigma, source = float(np.std(gain_ttr, ddof=1)), "TTR model seeds"
    else:
        sigma, source = 0.0, "none (a single split)"
    noise = math.sqrt(2.0) * sigma
    scale = {
        "ttr_gain": gain,
        "sigma": sigma,
        "sigma_source": source,
        "noise": noise,
        "lower": ANCHOR_Z * sigma,
        "upper": gain - ANCHOR_Z * noise,
    }
    return scale, _no_signal_reason(gain, noise)


def _no_signal_reason(gain: float, noise: float) -> str | None:
    """Why the real table has no signal measurable at its size, or None."""
    if gain > 0 and gain > UTILITY_SIGNAL_MULTIPLE * noise:
        return None
    return (
        "The real data carries no detectable signal for the target: TTR does not beat "
        f"a model without features by {UTILITY_SIGNAL_MULTIPLE:g}× its sampling noise "
        f"(gain {gain:.4f}, needs > {UTILITY_SIGNAL_MULTIPLE * noise:.4f}); utility is undefined"
    )


def _table_scale(signal: dict[str, Any] | None, rows: dict[str, Any]) -> dict[str, Any] | None:
    """The anchors fixed once for the real table, or None when this run must
    measure its own: no table-level signal, or one measured on tables of
    another size (the TTR gain and σ depend on it)."""
    if not signal or not signal.get("applicable") or signal.get("rows") != rows:
        return None
    if "lower" not in signal or "upper" not in signal:
        return None
    return {
        "ttr_gain": signal["ttr_gain"],
        "sigma": signal["sigma"],
        "sigma_source": f"table level, {signal['n_half_samples']} half-samples",
        "noise": signal["noise"],
        "lower": signal["lower"],
        "upper": signal["upper"],
    }


def pooled_utility_signal(probes: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Signal decision of the real table from several ``scale_only`` runs.

    *probes* are the ``details`` of ``utility.ml_efficiency`` run with
    ``ml_efficiency_config.scale_only`` on different splits of the real
    table.  Same rule as ``_utility_scale``, with every split pooled: the
    gain is the mean TTR gain and σ the pooled within-split spread of the
    paired half-sample gains (what one run measures on its own split).
    None when no split has two pairs.
    """
    gains: list[float] = []
    squares, dof, n_pairs = 0.0, 0, 0
    task_type = None
    sizes = {tuple(sorted((d.get("rows") or {}).items())) for d in probes}
    for details in probes:
        task_type = details["task_type"]
        chance = details["no_signal_scores_per_split"]
        gain_ttr = _gains(details["ttr_scores_per_split"], chance, task_type)
        gain_half = _gains(details["ttr_half_sample_scores_per_split"], chance, task_type)
        n = min(len(gain_ttr), len(gain_half))
        gains.extend(gain_ttr[np.isfinite(gain_ttr)])
        paired = gain_half[:n] - gain_ttr[:n]
        paired = paired[np.isfinite(paired)]
        if len(paired) > 1:
            squares += float(np.sum((paired - paired.mean()) ** 2))
            dof += len(paired) - 1
            n_pairs += len(paired)
    if dof == 0 or not gains:
        return None
    gain = float(np.mean(gains))
    sigma = math.sqrt(squares / dof)
    noise = math.sqrt(2.0) * sigma
    reason = _no_signal_reason(gain, noise)
    # With signal (gain > 3·noise = 4.24·σ > z·(1 + √2)·σ) the anchors are ordered.
    lower, upper = ANCHOR_Z * sigma, gain - ANCHOR_Z * noise
    signal = {
        "applicable": reason is None,
        "reason": reason,
        "lower": lower,
        "upper": upper,
        # Rows the anchors were measured on; a run on tables of another size
        # reuses only the decision (``_table_scale``).
        "rows": probes[0].get("rows") if len(sizes) == 1 else None,
        "task_type": task_type,
        "ttr_gain": gain,
        "sigma": sigma,
        "noise": noise,
        "threshold": UTILITY_SIGNAL_MULTIPLE * noise,
        "n_splits": len(probes),
        "n_half_samples": n_pairs,
        "sigma_dof": dof,
    }
    # The log-loss of the same fits (``utility.log_loss``): its own gain,
    # noise and anchors, applicable only where the task's metric is.
    log_loss = pooled_utility_signal([d["log_loss"] for d in probes if "log_loss" in d])
    if log_loss is not None:
        if reason is not None:
            log_loss.update(applicable=False, reason=reason)
        signal["log_loss"] = log_loss
    return signal


def _absolute_utility(
    result: StrategyResult,
    chance: list[float],
    scale: dict[str, Any],
    task_type: str,
) -> tuple[float, float]:
    """``(utility, standard error)`` on the absolute scale (module docstring).

    The standard error is that of the mean gain over the splits, rescaled,
    before clipping (the anchors are taken as fixed).
    """
    gain = _gains(result.scores_per_run, chance, task_type)
    lower, upper, ttr_gain = scale["lower"], scale["upper"], scale["ttr_gain"]
    # Symmetric around the TTR gain (module docstring); below it, the same as
    # (gain − lower) / (upper − lower).
    distance = abs(float(np.mean(gain)) - ttr_gain)
    value = float((ttr_gain - lower - distance) / (upper - lower))
    se = float("nan")
    if len(gain) > 1:
        se = float(np.std(gain, ddof=1) / math.sqrt(len(gain)) / (upper - lower))
    return min(1.0, max(0.0, value)), se


def _ordinal_level_ranks(
    target: str,
    real: pd.DataFrame,
    synth: pd.DataFrame,
    holdout: pd.DataFrame | None,
) -> dict[int, float] | None:
    """Class code → level rank ``i/(K−1)`` when *target* is an ordinal, else None.

    An ordinal target reaches the metric with its rank ``<target>__ord`` next
    to it (``SimpleCaster._cast_ordinal``); the codes are those of
    ``_encode_classification_labels``.  A code whose rows carry no rank or
    disagreeing ones (the shared code of unseen synthetic labels) gets none.
    """
    rank_col = f"{target}{ORDINAL_SHADOW_SUFFIX}"
    frames = [f for f in (real, synth, holdout) if f is not None]
    if not all(rank_col in f.columns for f in frames):
        return None
    pairs = pd.concat([f[[target, rank_col]] for f in frames], ignore_index=True).dropna()
    ranks: dict[int, float] = {}
    for code, values in pairs.groupby(target)[rank_col]:
        distinct = values.round(9).unique()
        if len(distinct) == 1:
            ranks[int(code)] = float(distinct[0])
    return ranks or None


def _label_key(value: Any) -> Any:
    """Canonical class key: ``1``, ``1.0`` and ``"1"`` are the same class."""
    if value is None or value is pd.NA:
        return ("nan",)
    if isinstance(value, (bool, np.bool_)):
        return ("b", bool(value))
    if isinstance(value, (int, float, np.integer, np.floating)):
        return ("nan",) if np.isnan(value) else ("n", float(value))
    text = str(value)
    try:
        number = float(text)
    except ValueError:
        return ("s", text)
    return ("nan",) if np.isnan(number) else ("n", number)


def _encode_classification_labels(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    target_col: str | list[str],
    holdout: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, Any, dict[str, int]]:
    """Map the target classes to integer codes fitted on the real table.

    Never truncates (``astype(int)`` turned 0.7 into 0 and could merge
    classes).  Synthetic (or holdout) labels that do not occur in the real
    target become one extra class; their row counts are returned so the
    caller can report them.

    Returns ``(real, synth, holdout, n_classes, unseen_rows)``.
    """
    real, synth = real.copy(), synth.copy()
    holdout = holdout.copy() if holdout is not None else None
    cols = [target_col] if isinstance(target_col, str) else list(target_col)
    n_classes_list: list[int] = []
    unseen_rows: dict[str, int] = {}

    for col in cols:
        real_keys = real[col].map(_label_key)
        classes = sorted(set(real_keys), key=repr)
        mapping = {k: i for i, k in enumerate(classes)}
        unseen_code = len(classes)
        real[col] = real_keys.map(mapping).astype(np.int64).to_numpy()
        n_classes_list.append(len(classes))

        for name, df in (("synthetic", synth), ("holdout", holdout)):
            if df is None:
                continue
            codes = df[col].map(_label_key).map(mapping)
            n_unseen = int(codes.isna().sum())
            if n_unseen:
                unseen_rows[f"{name}.{col}"] = n_unseen
            df[col] = codes.fillna(unseen_code).astype(np.int64).to_numpy()

    n_classes = n_classes_list[0] if isinstance(target_col, str) else n_classes_list
    return real, synth, holdout, n_classes, unseen_rows


def _parse_strategies(strategies: list[str] | None) -> list[str]:
    if strategies is None:
        return list(DEFAULT_STRATEGIES)
    if isinstance(strategies, str) or not strategies:
        raise ValueError(
            f"strategies must be a non-empty list, got {strategies!r}. "
            f"Available: {sorted(AVAILABLE_STRATEGIES)}"
        )
    invalid = {s.lower() for s in strategies} - AVAILABLE_STRATEGIES
    if invalid:
        raise ValueError(f"Invalid strategies: {invalid}. Available: {AVAILABLE_STRATEGIES}")
    return list(dict.fromkeys(s.lower() for s in strategies))


def _error_result(metric_id: str, error: str, purpose_tags: set[str], **details) -> MetricResult:
    return MetricResult(
        id=metric_id,
        value=float("nan"),
        details={"error": error, **details},
        family="utility",
        purpose_tags=purpose_tags,
    )


def _check_forced_task(metric: Any, target_col: str, task_type: str) -> str | None:
    """Target checks of ``classification_/regression_efficiency``."""
    y = metric._real_data[target_col]
    if task_type == "classification":
        n_classes = y.map(_label_key).nunique()
        # Up to 10 classes always pass (as in ``_determine_task_type``): with
        # fewer than 20 rows ``len(y) // 10`` rejected even a binary target.
        if n_classes > max(10, min(50, len(y) // 10)):
            return f"Target has too many unique values ({n_classes}) for classification"
        return None
    if not pd.api.types.is_numeric_dtype(y):
        return "Target column is not numeric"
    if y.nunique() <= 10:
        return f"Target has only {y.nunique()} unique values. Use classification."
    return None


def run_ml_efficiency(
    metric: Any,
    strategies: list[str],
    metric_id: str,
    purpose_tags: set[str],
    task_type: str | None = None,
    scoring: str | None = None,
) -> MetricResult:
    """Shared computation of every ML-efficiency metric.

    *scoring* ``"log_loss"`` scores the same fits with the log-loss instead of
    the task's metric (``utility.log_loss``); TTR, its half-samples and each
    strategy are fitted once per evaluation and shared by both.

    *metric* is a fitted ``MLMetricMixin``/``MetricBase`` instance (it carries
    the data, the context and the budget).  *strategies* are averaged; the
    principal one is TSTR when selected, else the first.  If the TTR baseline
    or the principal strategy fails the score is NaN with ``details.error``;
    other failures are excluded from the mean and listed in
    ``details.failed_strategies``.
    """
    target_cols, error = metric._validate_target()
    if error == NO_TARGET:
        return MetricResult.not_applicable(metric_id, "utility", error, purpose_tags)
    if error:
        return _error_result(metric_id, error, purpose_tags)
    if task_type is None:
        task_type = metric._determine_task_type(target_cols[0])
    else:
        error = _check_forced_task(metric, target_cols[0], task_type)
        if error:
            return _error_result(metric_id, error, purpose_tags)

    feature_cols = metric._get_feature_cols(target_cols)
    if not feature_cols:
        return _error_result(metric_id, "No feature columns found", purpose_tags)

    ctx = metric._context
    seed = ctx.get("seed", 42)
    n_splits = int(getattr(metric, "n_splits", DEFAULT_N_SPLITS))
    target_arg: str | list[str] = target_cols[0] if len(target_cols) == 1 else target_cols

    real, synth = metric._real_data, metric._synth_data
    holdout = ctx.get("holdout")
    if holdout is None:
        # Without real rows the generator never saw, the test rows are a split
        # of the table it was trained on: a shuffled copy of it scores 1.
        # Required like the privacy metrics (2026-10-19).
        return MetricResult.not_applicable(
            metric_id,
            "utility",
            "Utility needs a holdout (real rows the generator never saw): without it the "
            "test rows are part of the generator's training table and a copy scores 1",
            purpose_tags,
        )
    if holdout is not None:
        missing = [c for c in feature_cols + target_cols if c not in holdout.columns]
        if missing:
            return _error_result(
                metric_id, f"Columns missing from the holdout: {missing}", purpose_tags
            )

    # A row with a missing target teaches nothing and cannot be scored: only
    # those rows leave the utility metrics (every other metric keeps them, and
    # missing *features* stay: CatBoost handles them natively).
    rows_without_target: dict[str, int] = {}

    def _with_target(df: pd.DataFrame | None, name: str) -> pd.DataFrame | None:
        if df is None:
            return None
        keep = df[target_cols].notna().all(axis=1)
        if not keep.all():
            rows_without_target[name] = int((~keep).sum())
            return df.loc[keep].reset_index(drop=True)
        return df

    real, synth = _with_target(real, "real"), _with_target(synth, "synthetic")
    holdout = _with_target(holdout, "holdout")
    if len(real) < 2:
        return MetricResult.not_applicable(
            metric_id, "utility", "Fewer than 2 real rows have a target value", purpose_tags
        )
    if len(synth) < 2:
        return _error_result(
            metric_id, "Fewer than 2 synthetic rows have a target value", purpose_tags
        )
    n_classes = None
    unseen_rows: dict[str, int] = {}
    if task_type == "classification":
        real, synth, holdout, n_classes, unseen_rows = _encode_classification_labels(
            real, synth, target_arg, holdout
        )
        if min([n_classes] if isinstance(n_classes, int) else n_classes) < 2:
            # Nothing to predict: the metric does not apply (it did not fail).
            return MetricResult.not_applicable(
                metric_id,
                "utility",
                "Target has fewer than 2 classes in the real data",
                purpose_tags,
            )
    level_ranks = None
    if task_type == "classification" and isinstance(target_arg, str):
        level_ranks = _ordinal_level_ranks(target_arg, real, synth, holdout)
        if level_ranks is not None:
            task_type = "ordinal"
    trainer = metric._get_trainer(task_type, n_targets=len(target_cols), level_ranks=level_ranks)
    # What the score measures: the task's metric, or the log-loss of the same fits.
    kind = task_type
    if scoring == "log_loss":
        if task_type != "classification" or not isinstance(target_arg, str):
            return MetricResult.not_applicable(
                metric_id,
                "utility",
                "The log-loss scores predicted class probabilities: it applies to a single "
                f"nominal or binary target, not to this one ({task_type})",
                purpose_tags,
            )
        kind = "log_loss"

    principal = "tstr" if "tstr" in strategies else strategies[0]
    details: dict[str, Any] = {
        "task_type": task_type,
        "metric": _METRIC_NAMES[kind],
        "model": "CatBoost",
        "formula": _FORMULAS[kind] + "; score = mean over selected strategies",
        "evaluation": (
            f"fixed real holdout ({len(holdout)} rows), {n_splits} model seed(s)"
            if holdout is not None
            else f"repeated 80/20 holdout, {n_splits} split(s), seeds {seed}..{seed + n_splits - 1}"
        ),
        "tuning": (
            "CatBoost defaults, early stopping on a validation slice of the "
            "training data of each split"
        ),
        "n_features": len(feature_cols),
        "n_targets": len(target_cols),
        "target_columns": target_cols,
        "selected_strategies": list(strategies),
        "principal_strategy": principal,
        "n_runs": trainer.n_runs,
        "n_splits": n_splits,
    }
    if rows_without_target:
        details["rows_without_target"] = rows_without_target
    if n_classes is not None:
        details["n_classes"] = n_classes
        details["unseen_label_rows"] = unseen_rows

    split_kwargs = {"seed": seed, "n_splits": n_splits, "holdout": holdout}
    chance_ttr = _chance_scores(
        TrainingStrategy.TTR,
        real,
        synth,
        target_arg,
        feature_cols,
        task_type=kind,
        **split_kwargs,
    )
    details["no_signal_scores_per_split"] = chance_ttr
    ml_config = (ctx.get("config") or {}).get("evaluation", {}).get("ml_efficiency_config") or {}
    scale_only = bool(ml_config.get("scale_only"))
    signal = None if scale_only else ctx.get("utility_signal")
    rows = {"real": len(real), "holdout": None if holdout is None else len(holdout)}

    # 1. Scale of the real table: fixed once per table when given (module
    # docstring), else measured by this run.
    ttr: StrategyResult | None = None
    if signal is not None:
        details["signal_decision"] = {"source": "calibration", **signal}
        if not signal["applicable"]:
            return MetricResult.not_applicable(
                metric_id,
                "utility",
                f"{signal['reason']} (decided once for this real table)",
                purpose_tags,
                **details,
            )
        if kind == "log_loss":
            # Its own anchors (and gain over the noise), measured with the same
            # half-samples; a calibration without them leaves this run to
            # measure its own.
            signal = signal.get("log_loss")
            if signal is not None and not signal["applicable"]:
                return MetricResult.not_applicable(
                    metric_id,
                    "utility",
                    f"{signal['reason']} (log-loss, decided once for this real table)",
                    purpose_tags,
                    **details,
                )
    scale = _table_scale(signal, rows)
    if scale is None:
        ttr_fits = _get_or_compute_ttr(
            ctx, real, synth, target_arg, feature_cols, trainer, seed, n_splits, holdout
        )
        ttr = _view(ttr_fits, "log_loss") if kind == "log_loss" else ttr_fits
        details["ttr_baseline"] = ttr.score
        details["ttr_std"] = ttr.std
        details["ttr_scores_per_split"] = ttr.scores_per_run
        if not ttr.is_valid:
            return _error_result(
                metric_id, f"TTR baseline failed: {ttr.error}", purpose_tags, **details
            )
        half_fits = _get_or_compute_half_scores(
            ctx, real, synth, target_arg, feature_cols, trainer, seed, n_splits, holdout
        )
        half = _view(half_fits, "log_loss") if kind == "log_loss" else half_fits
        details["ttr_half_sample_scores_per_split"] = half.scores_per_run
        if half.error:
            details["ttr_half_sample_errors"] = half.error
        scale, no_signal = _utility_scale(ttr, half, chance_ttr, kind)
        details["scale"] = scale
        if scale_only:
            details["rows"] = rows
            if kind == "classification" and "log_loss" in ttr_fits.extra_scores_per_run:
                # The same probe also measures the log-loss scale (``utility.log_loss``).
                details["log_loss"] = {
                    "task_type": "log_loss",
                    "no_signal_scores_per_split": _chance_scores(
                        TrainingStrategy.TTR,
                        real,
                        synth,
                        target_arg,
                        feature_cols,
                        task_type="log_loss",
                        **split_kwargs,
                    ),
                    "ttr_scores_per_split": ttr_fits.extra_scores_per_run["log_loss"],
                    "ttr_half_sample_scores_per_split": half_fits.extra_scores_per_run.get(
                        "log_loss", []
                    ),
                    "rows": rows,
                }
            return MetricResult.not_applicable(
                metric_id, "utility", "scale only (calibration probe)", purpose_tags, **details
            )
        if signal is not None:
            # The table has signal, but it was measured at another size:
            # only the decision is reused, the anchors are this run's.
            no_signal = (
                None
                if scale["upper"] > scale["lower"]
                else (
                    "The table-level decision found signal, but this run's anchors are not "
                    f"ordered (lower {scale['lower']:.4f}, upper {scale['upper']:.4f}); "
                    "utility is undefined"
                )
            )
        if no_signal is not None:
            # A property of the real table, not a failure of the computation.
            return MetricResult.not_applicable(
                metric_id, "utility", no_signal, purpose_tags, **details
            )
    else:
        details["scale"] = scale

    # 2. Strategies
    per_strategy: dict[str, dict[str, Any]] = {}
    failed: dict[str, str] = {}
    utilities: dict[str, float] = {}
    ses: dict[str, float] = {}
    run_cache: dict = ctx.setdefault(_TTR_CACHE_KEY, {})
    fits_key = _ttr_cache_key(real, target_arg, feature_cols, trainer, seed, n_splits, holdout)
    targets = [target_arg] if isinstance(target_arg, str) else list(target_arg)
    for name in strategies:
        # Per evaluation run: utility.tstr and utility.log_loss score the same fits.
        strategy_key = fits_key + ("strategy", name, _digest(synth, feature_cols + targets))
        result = run_cache.get(strategy_key)
        if result is None:
            result = _evaluate_strategy(
                TrainingStrategy(name),
                real,
                synth,
                target_arg,
                feature_cols,
                trainer,
                seed,
                n_splits,
                holdout,
            )
            run_cache[strategy_key] = result
        if kind == "log_loss":
            result = _view(result, "log_loss")
        entry: dict[str, Any] = {
            "score": result.score,
            "std": result.std,
            "scores_per_split": result.scores_per_run,
            "best_params": result.best_params,
            "is_valid": result.is_valid,
            "error": result.error,
        }
        if result.is_valid:
            strategy = TrainingStrategy(name)
            chance = (
                chance_ttr
                if strategy in _REAL_TEST_STRATEGIES
                else _chance_scores(
                    strategy,
                    real,
                    synth,
                    target_arg,
                    feature_cols,
                    task_type=kind,
                    **split_kwargs,
                )
            )
            delta, delta_se = (
                _strategy_delta(result, ttr, kind)
                if ttr is not None
                else (float("nan"), float("nan"))
            )
            utilities[name], ses[name] = _absolute_utility(result, chance, scale, kind)
            entry.update(
                delta_from_ttr=delta,
                delta_se=delta_se,
                no_signal_scores_per_split=chance,
                utility=utilities[name],
                utility_se=ses[name],
            )
        else:
            failed[name] = result.error or "unknown error"
        per_strategy[name] = entry

    details["strategies"] = per_strategy
    details["delta_scores"] = {
        n: e["delta_from_ttr"] for n, e in per_strategy.items() if n in utilities
    }
    details["failed_strategies"] = failed

    if principal in failed:
        return _error_result(
            metric_id,
            f"Principal strategy {principal} failed: {failed[principal]}",
            purpose_tags,
            **details,
        )

    score = float(np.mean(list(utilities.values())))
    # SE of the mean of the per-strategy utilities, ignoring their correlation.
    se_values = [v for v in ses.values() if not math.isnan(v)]
    details["score_se"] = (
        float(math.sqrt(sum(v * v for v in se_values)) / len(se_values))
        if len(se_values) == len(ses) and se_values
        else float("nan")
    )
    return MetricResult(
        id=metric_id,
        value=score,
        details=details,
        family="utility",
        purpose_tags=purpose_tags,
    )


# ── standalone strategy metrics ──────────────────────────────────────────


class BaseStandaloneStrategyMetric(MLMetricMixin, MetricBase):
    """Base for ``utility.tts``, ``utility.tstr``, etc.

    Computes TTR once, then the requested strategy, with the same protocol
    and score as ``utility.ml_efficiency`` restricted to that strategy.
    """

    def __init__(
        self,
        n_runs: int = 1,
        thread_count: int | None = None,
        n_splits: int = DEFAULT_N_SPLITS,
    ):
        super().__init__()
        self.n_runs = n_runs
        self.thread_count = thread_count
        self.n_splits = n_splits

    def fit(
        self,
        real_data: pd.DataFrame,
        synth_data: pd.DataFrame,
        context: dict[str, Any],
    ) -> BaseStandaloneStrategyMetric:
        self._setup(real_data, synth_data, context)
        ml_config = self._apply_ml_config(context)
        context["_ttr_process_cache"] = bool(ml_config.get("cache_ttr", True))
        return self

    def _compute_strategy(self, strategy: TrainingStrategy) -> MetricResult:
        metric_id = f"utility.{strategy.value}"
        try:
            result = run_ml_efficiency(self, [strategy.value], metric_id, self.purpose_tags)
        except Exception as e:
            return _error_result(metric_id, f"{strategy.value} computation failed: {e}", set())
        entry = result.details.get("strategies", {}).get(strategy.value)
        if entry is not None:
            result.details.update(
                strategy=strategy.value,
                strategy_score=entry["score"],
                delta_from_ttr=entry.get("delta_from_ttr"),
                std=entry["std"],
                best_params=entry["best_params"],
            )
        return result


# ── registered standalone metrics ────────────────────────────────────────


@register("utility.tts")
class TTSStandaloneMetric(BaseStandaloneStrategyMetric, TTSMetric):
    """Train on Synthetic, Test on Synthetic."""

    name: str = "tts"
    family: str = "utility"
    purpose_tags: set[str] = {"utility", "ml_efficiency", "tts"}

    def compute(self) -> MetricResult:
        return self._compute_strategy(TrainingStrategy.TTS)


@register("utility.tstr")
class TSTRStandaloneMetric(BaseStandaloneStrategyMetric, TSTRMetric):
    """Train on Synthetic, Test on Real."""

    name: str = "tstr"
    family: str = "utility"
    purpose_tags: set[str] = {"utility", "ml_efficiency", "tstr"}

    def compute(self) -> MetricResult:
        return self._compute_strategy(TrainingStrategy.TSTR)


@register("utility.trts")
class TRTSStandaloneMetric(BaseStandaloneStrategyMetric, TRTSMetric):
    """Train on Real, Test on Synthetic."""

    name: str = "trts"
    family: str = "utility"
    purpose_tags: set[str] = {"utility", "ml_efficiency", "trts"}

    def compute(self) -> MetricResult:
        return self._compute_strategy(TrainingStrategy.TRTS)


@register("utility.ttrs")
class TTRSStandaloneMetric(BaseStandaloneStrategyMetric, TTRSMetric):
    """Train on Real+Synthetic, Test on Real."""

    name: str = "ttrs"
    family: str = "utility"
    purpose_tags: set[str] = {"utility", "ml_efficiency", "ttrs"}

    def compute(self) -> MetricResult:
        return self._compute_strategy(TrainingStrategy.TTRS)


# ── aggregate metric ─────────────────────────────────────────────────────


@register("utility.ml_efficiency")
class MLEfficiencyMetric(MLMetricMixin, MetricBase):
    """Unified ML Efficiency Metric with auto task detection.

    TTR is computed **once** as the baseline; the score is the mean of the
    per-strategy utilities on the absolute scale (see the module docstring).
    """

    name: str = "ml_efficiency"
    family: str = "utility"
    purpose_tags: set[str] = {"utility", "ml_efficiency", "auto"}

    def __init__(
        self,
        strategies: list[str] | None = None,
        n_runs: int = 1,
        thread_count: int | None = None,
        n_splits: int = DEFAULT_N_SPLITS,
    ):
        super().__init__()
        self.n_runs = n_runs
        self.thread_count = thread_count
        self.n_splits = n_splits
        self._selected_strategies = self._parse_strategies(strategies)

    _parse_strategies = staticmethod(_parse_strategies)

    def fit(
        self,
        real_data: pd.DataFrame,
        synth_data: pd.DataFrame,
        context: dict[str, Any],
    ) -> MLEfficiencyMetric:
        self._setup(real_data, synth_data, context)
        ml_config = self._apply_ml_config(context)
        if "strategies" in ml_config:
            self._selected_strategies = self._parse_strategies(ml_config["strategies"])
        context["_ttr_process_cache"] = bool(ml_config.get("cache_ttr", True))
        return self

    def compute(self) -> MetricResult:
        try:
            return run_ml_efficiency(
                self, self._selected_strategies, "utility.ml_efficiency", self.purpose_tags
            )
        except Exception as e:
            return self._err(f"ML efficiency computation failed: {e}")

    def _err(self, msg: str) -> MetricResult:
        return _error_result("utility.ml_efficiency", msg, self.purpose_tags)


# ── log-loss metric ──────────────────────────────────────────────────────


@register("utility.log_loss")
class LogLossUtilityMetric(BaseStandaloneStrategyMetric, TSTRMetric):
    """TSTR scored with the log-loss instead of the ROC AUC.

    The AUC only ranks the rows: a model trained on synthetic labels with
    30 % noise ranks the real rows almost as well as TTR, so ``utility.tstr``
    scored it as useful (study of 2026-10-30, ``doctorado.tex`` §8).  The
    log-loss also scores how sure the model is, and that noise makes it
    unsure.  Same absolute scale, anchors and applicability rule as
    ``utility.tstr``, measured on the log-loss of the same fits (no extra
    model is trained).  Single nominal or binary targets only.
    """

    name: str = "log_loss"
    family: str = "utility"
    purpose_tags: set[str] = {"utility", "probabilistic", "tstr", "log_loss"}

    def compute(self) -> MetricResult:
        metric_id = "utility.log_loss"
        try:
            result = run_ml_efficiency(
                self, ["tstr"], metric_id, self.purpose_tags, scoring="log_loss"
            )
        except Exception as e:
            return _error_result(metric_id, f"log-loss computation failed: {e}", set())
        entry = result.details.get("strategies", {}).get("tstr")
        if entry is not None:
            result.details.update(
                strategy="tstr",
                strategy_score=entry["score"],
                delta_from_ttr=entry.get("delta_from_ttr"),
                std=entry["std"],
            )
        return result
