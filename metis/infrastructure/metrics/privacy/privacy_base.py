"""
Base classes for privacy metrics.

This module provides the unified hierarchy of base classes for all privacy metrics:

    BasePrivacyMetric (abstract)
        │
        ├── DatasetPrivacyMetric (dataset-based metrics)
        │       │
        │       ├── AttackMetric (MIA, Inference Attack, Record Linkage)
        │       ├── SyntacticMetric (k-Anonymity, l-Diversity, t-Closeness)
        │       └── EmpiricalSimilarityMetric (DCR, NNAA)
        │
        └── MechanismPrivacyMetric (mechanism-based metrics)
                └── DifferentialPrivacyMetric (DP verification)

All metrics follow the fit/compute pattern and produce normalized scores in [0, 1]
where 1 = most private (best privacy preservation).
"""

import warnings
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import beta, rankdata
from sklearn.neighbors import NearestNeighbors

from metis.domain.entities import MetricResult
from metis.infrastructure.metrics.base import (
    ORDINAL_SHADOW_SUFFIX,
    MetricBase,
    ordinal_shadows,
)

# =============================================================================
# Shared helpers
# =============================================================================

_NAN_TOKEN = "__nan__"


class HoldoutColumnsWarning(UserWarning):
    """The holdout passed to the privacy metrics is partly or wholly unusable."""


class MissingHoldoutWarning(UserWarning):
    """Privacy was measured without a holdout of real rows the generator never saw."""


# One-hot entries are weighted so that a categorical mismatch adds exactly 1 to
# the squared Euclidean distance, the same as a one-standard-deviation
# difference in a standardised numeric column.
_ONE_HOT_WEIGHT = 1.0 / np.sqrt(2.0)


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _as_category_strings(series: pd.Series) -> np.ndarray:
    """Categorical values as strings, with missing values as their own category."""
    return series.astype(object).where(series.notna(), _NAN_TOKEN).astype(str).to_numpy()


# Encodings at least this wide are built sparse (``MixedRecordEncoder``).
SPARSE_MIN_WIDTH = 64


def _frame_fingerprint(frame: pd.DataFrame | None) -> str:
    if frame is None:
        return "-"
    return f"{frame.shape}:{int(pd.util.hash_pandas_object(frame, index=False).sum())}"


class MixedRecordEncoder:
    """Encode mixed-type records into a Euclidean space fitted on the real table.

    - Numeric columns: z-score with the real mean and standard deviation.
      Missing values are imputed with the real mean (0 after scaling) and,
      when the real column has missing values, a 0/1 missingness indicator is
      added, so missingness is part of the record instead of a silent
      ``fillna(0)`` in the raw scale.
    - Categorical columns: one-hot over the categories of the real table and
      of the *vocabulary_from* tables (missing is its own category), weighted
      by ``1/sqrt(2)``, so two different categories are at squared distance 1
      and equal ones at 0.  A category outside that vocabulary goes to one
      slot shared by all of them (squared distance 1 from every known one).
      The privacy metrics pass every table they compare (holdout and
      synthetic): with the vocabulary of the real table alone, every holdout
      and synthetic value of a near-unique column (airbnb ``name``) fell in
      the shared slot, at distance 0 from each other and 1 from every
      training row, and an independent synthetic table looked closer to the
      holdout than to the training rows (DCR share 0.37 for 0.50 expected,
      with 10 % exact copies scoring 1.0).
    - *unscaled* columns (the rank view ``i/(K-1)`` of an ordinal, see
      ``PrivacyMetricBase._distance_columns``): kept in [0, 1] without
      z-score, Gower's ordinal distance, so the largest jump (first to last
      level) costs 1, the same as a categorical mismatch, and one level costs
      ``1/(K-1)``.  Standardising it inflated the step of low-variance
      ordinals (1.6 sd per level in a K=3 column).  They always carry the
      missingness indicator, so an unknown or missing level is at distance
      >= 1 from every observed one.

    Encodings wider than ``SPARSE_MIN_WIDTH`` columns come out as a CSR
    matrix: a text or identifier-like column one-hot encodes into thousands
    of columns with a single non-zero per row (airbnb ``name``: ~6 000 per
    third), and the dense brute-force neighbour search over them took ~20 min
    per evaluation.  The distances are the same.
    """

    def __init__(
        self,
        real: pd.DataFrame,
        columns: list[str],
        unscaled: set[str] | None = None,
        vocabulary_from: list[pd.DataFrame] | None = None,
    ):
        self.columns = list(columns)
        self.numeric = [c for c in self.columns if _is_numeric(real[c])]
        numeric_set = set(self.numeric)
        self.categorical = [c for c in self.columns if c not in numeric_set]
        unscaled = set(unscaled or ()) & numeric_set
        self._num_stats: dict[str, tuple[float, float, bool]] = {}
        for col in self.numeric:
            values = pd.to_numeric(real[col], errors="coerce").astype(float)
            if col in unscaled:
                self._num_stats[col] = (0.0, 1.0, True)
                continue
            mean = float(values.mean()) if values.notna().any() else 0.0
            std = float(values.std(ddof=0)) if values.notna().sum() > 1 else 0.0
            self._num_stats[col] = (mean, std if std > 0 else 1.0, bool(values.isna().any()))
        tables = [real, *(t for t in vocabulary_from or () if t is not None)]
        self._categories = {
            c: np.unique(
                np.concatenate([_as_category_strings(t[c]) for t in tables if c in t.columns])
            )
            for c in self.categorical
        }

    @property
    def width(self) -> int:
        """Number of columns of the encoding."""
        numeric = sum(
            2 if with_indicator else 1 for _, _, with_indicator in self._num_stats.values()
        )
        return numeric + sum(len(c) + 1 for c in self._categories.values())

    def transform(self, data: pd.DataFrame) -> np.ndarray | sparse.csr_matrix:
        n_rows = len(data)
        blocks: list[np.ndarray] = []
        for col in self.numeric:
            mean, std, with_indicator = self._num_stats[col]
            values = pd.to_numeric(data[col], errors="coerce").astype(float).to_numpy()
            missing = np.isnan(values)
            blocks.append(np.where(missing, 0.0, (values - mean) / std)[:, None])
            if with_indicator:
                blocks.append(missing.astype(float)[:, None])
        codes = []  # per categorical column: (slot of each row, number of slots)
        for col in self.categorical:
            categories = self._categories[col]
            values = _as_category_strings(data[col])
            pos = np.clip(np.searchsorted(categories, values), 0, len(categories) - 1)
            hit = categories[pos] == values
            codes.append((np.where(hit, pos, len(categories)), len(categories) + 1))
        if self.width >= SPARSE_MIN_WIDTH:
            return self._sparse(n_rows, blocks, codes)
        for slot, width in codes:
            onehot = np.zeros((n_rows, width))
            onehot[np.arange(n_rows), slot] = _ONE_HOT_WEIGHT
            blocks.append(onehot)
        if not blocks:
            return np.zeros((n_rows, 0))
        return np.hstack(blocks)

    @staticmethod
    def _sparse(n_rows: int, blocks: list[np.ndarray], codes: list) -> sparse.csr_matrix:
        parts = [sparse.csr_matrix(np.hstack(blocks))] if blocks else []
        for slot, width in codes:
            parts.append(
                sparse.csr_matrix(
                    (np.full(n_rows, _ONE_HOT_WEIGHT), (np.arange(n_rows), slot)),
                    shape=(n_rows, width),
                )
            )
        return sparse.hstack(parts, format="csr")


def excess_risk_score(p_attack: float, p_baseline: float) -> tuple[float, float]:
    """Privacy score from an attack success rate and its baseline.

    ``risk = (p_attack - p_baseline) / (1 - p_baseline)`` clipped to [0, 1]
    (the excess success over what the attacker achieves against records the
    generator never saw, as in Anonymeter, Giomi et al. 2023), and
    ``privacy = 1 - risk``.

    Returns:
        ``(privacy_score, risk)``
    """
    if not np.isfinite(p_attack) or not np.isfinite(p_baseline):
        return float("nan"), float("nan")
    if p_attack <= p_baseline or p_baseline >= 1.0:
        risk = 0.0
    else:
        risk = min(1.0, (p_attack - p_baseline) / (1.0 - p_baseline))
    return 1.0 - risk, risk


def _emd(shares: np.ndarray, global_dist: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Earth Mover's Distance of each row of *shares* to *global_dist*.

    t-closeness (Li et al. 2007) measures the distance between a class
    and the whole table with the EMD over a ground distance between values:
    the *ordered* distance ``|i − j| / (m − 1)`` for numeric and ordinal
    values, for which ``EMD = Σ_i |Σ_{j≤i} (q_j − p_j)| / (m − 1)``, and the
    equal distance for nominal ones, for which it is the total variation.
    Values outside the order (*order* NaN: missing, pooled) are at distance
    1 from every other value: their mass difference counts as in the total
    variation and the rest is moved along the order,

        EMD = TV(partition: each unordered value, ordered block)
              + min(q_ord, p_ord) · EMD_ordered(q | ordered, p | ordered).

    It equals Li et al.'s formula when every value is ordered and stays in
    [0, 1].  Before 2026-10-14 numeric sensitive columns used the total
    variation over their bins: a class of only the next bin was as far from
    the table as one of only the extreme bin.
    """
    ordered = np.flatnonzero(np.isfinite(order))
    loose = np.flatnonzero(~np.isfinite(order))
    ordered = ordered[np.argsort(order[ordered], kind="stable")]
    q_ord = shares[:, ordered].sum(axis=1)
    p_ord = float(global_dist[ordered].sum())
    tv = 0.5 * (np.abs(shares[:, loose] - global_dist[loose]).sum(axis=1) + np.abs(q_ord - p_ord))
    if len(ordered) < 2 or p_ord <= 0:
        return tv
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(q_ord[:, None] > 0, shares[:, ordered] / q_ord[:, None], 0.0)
    p = global_dist[ordered] / p_ord
    cum = np.cumsum(q - p, axis=1)[:, :-1]
    emd_ord = np.where(q_ord > 0, np.abs(cum).sum(axis=1) / (len(ordered) - 1), 0.0)
    return tv + np.minimum(q_ord, p_ord) * emd_ord


# False-positive rates at which the membership audit sets its thresholds
# (``audit_epsilon``); 0 means "above every non-member".
AUDIT_FPR_GRID = (0.0, 0.01, 0.02, 0.05, 0.1, 0.25)
AUDIT_CONFIDENCE = 0.95


def clopper_pearson(k: int, n: int, alpha: float) -> tuple[float, float]:
    """Exact two-sided ``1 − alpha`` interval for a binomial proportion ``k/n``."""

    if n <= 0:
        return 0.0, 1.0
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def audit_epsilon(
    member_scores: np.ndarray,
    non_member_scores: np.ndarray,
    *,
    fpr_grid: tuple[float, ...] = AUDIT_FPR_GRID,
    confidence: float = AUDIT_CONFIDENCE,
) -> dict[str, Any]:
    """Empirical lower bound on ε from a membership attack, and its score.

    Privacy is a worst-case property: an attack that confidently singles out
    a few training records has succeeded even if it is no better than chance
    on average (Carlini et al. 2022, S&P; Ganev & De Cristofaro 2025, S&P).
    So instead of an average (AUC, mean excess rate) the attack is audited
    as in differential-privacy auditing (Jagielski et al. 2020; Nasr et al.
    2021): for thresholds set at low false-positive rates on the
    non-members, a DP mechanism with budget ε must satisfy
    ``TPR ≤ e^ε · FPR``, so

        ε̂ = max_t log(TPR_lo(t) / FPR_hi(t)),

    with Clopper–Pearson bounds at ``confidence`` (Bonferroni over the
    thresholds and the two bounds), and ``privacy = e^{−max(0, ε̂)}``.

    An independent sample (members and non-members exchangeable) gives
    ε̂ ≤ 0 with probability ≥ ``confidence``: privacy 1.  Copying a few
    percent of the training rows gives exact matches that no non-member has:
    with 400 + 400 records, 5 % of copies give ε̂ ≈ 1.4 (privacy ≈ 0.24), where
    the averages moved by ≈0.05.  The bound needs enough records: with
    ``n`` non-members the largest measurable ε̂ is ≈ log(n/4).

    *member_scores* / *non_member_scores*: attack score of each training
    record / holdout record, higher = "looks like a member".
    """
    m = np.asarray(member_scores, dtype=float)
    h = np.asarray(non_member_scores, dtype=float)
    m, h = m[np.isfinite(m)], h[np.isfinite(h)]
    if len(m) == 0 or len(h) == 0:
        return {
            "epsilon": float("nan"),
            "privacy_score": float("nan"),
            "advantage": float("nan"),
            "thresholds": [],
        }
    alpha = (1.0 - confidence) / len(fpr_grid)
    rows = []
    best = 0.0
    for target in fpr_grid:
        if target <= 0:
            tp, fp = int(np.sum(m > h.max())), 0
        else:
            t = float(np.quantile(h, 1.0 - target, method="higher"))
            tp, fp = int(np.sum(m >= t)), int(np.sum(h >= t))
        tpr_lo, _ = clopper_pearson(tp, len(m), alpha)
        _, fpr_hi = clopper_pearson(fp, len(h), alpha)
        eps = float(np.log(tpr_lo / fpr_hi)) if tpr_lo > 0 else float("-inf")
        best = max(best, eps)
        rows.append(
            {
                "target_fpr": target,
                "tpr": tp / len(m),
                "fpr": fp / len(h),
                "epsilon_lower_bound": eps if np.isfinite(eps) else None,
            }
        )
    return {
        "epsilon": best,
        "privacy_score": float(np.exp(-best)),
        "advantage": float(max(r["tpr"] - r["fpr"] for r in rows)),
        "confidence": confidence,
        "n_members": len(m),
        "n_non_members": len(h),
        "thresholds": rows,
    }


# =============================================================================
# Partial exact matches: the second strategy of every attack (2026-11-12)
# =============================================================================

PARTIAL_MATCH_MAX_RECORDS = 2000
# Synthetic rows searched for each record.  Above it a fixed sample is used:
# the search is records × synthetic rows × columns (50 000 × 30 took ~7 min
# per attack), and a copied row is as likely to be in the sample as any other.
PARTIAL_MATCH_MAX_SYNTH = 20_000


def _plain_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _canonical_key(value: Any) -> str:
    """Key of a cell of a column whose type differs between tables: a number
    and the text that writes it are the same value (``1``, ``1.0``, ``"1"``,
    ``True``)."""
    if pd.isna(value):
        return ""
    if isinstance(value, (bool, np.bool_, int, float, np.integer, np.floating)):
        number = float(value)
        return repr(number) if np.isfinite(number) else ""
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    return repr(number) if np.isfinite(number) else text


def _value_keys(series: pd.Series, mode: str = "auto") -> np.ndarray:
    """Exact value of each cell as text (numbers compared exactly; missing = "").

    *mode* is the type of the column across the tables compared
    (``_key_mode``): ``"numeric"``, ``"numeric32"`` (some table holds it in
    single precision, so all are compared in it), ``"text"`` or ``"mixed"``.
    ``"auto"`` takes it from *series* alone.
    """
    if mode == "auto":
        mode = "numeric" if _plain_numeric(series) else "text"
    if mode in ("numeric", "numeric32"):
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        if mode == "numeric32":
            values = values.astype(np.float32).astype(float)
        return np.array(["" if not np.isfinite(v) else repr(float(v)) for v in values], dtype=object)
    if mode == "mixed":
        return np.array([_canonical_key(v) for v in series], dtype=object)
    return np.array(["" if pd.isna(v) else str(v).strip() for v in series], dtype=object)


def _key_mode(columns: list[pd.Series]) -> str:
    """How to compare a column that several tables hold, maybe with different types.

    The same value used to get a different key in each table when their types
    differed (a number against its text, a boolean against 0/1, single against
    double precision), so the strategy saw no match at all, or saw them for
    the members only.
    """
    columns = [c for c in columns if len(c)]
    numeric = [_plain_numeric(c) for c in columns]
    if not columns or not any(numeric):
        return "text"
    if not all(numeric):
        return "mixed"
    single = [c.dtype == np.float32 for c in columns]
    return "numeric32" if any(single) and not all(single) else "numeric"


def partial_match(
    frames: list[pd.DataFrame],
    synth: pd.DataFrame,
    real: pd.DataFrame,
    columns: list[str],
    chunk: int = 64,
    max_synth: int | None = PARTIAL_MATCH_MAX_SYNTH,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """For each record of each frame, the synthetic row that reproduces most of it value by value.

    A whole-record distance misses a generator that copies training rows and
    replaces part of their values with the mean, the mode or a constant: each
    copy moves as far from its source as from anybody else, while the values
    it keeps still identify the person (on Telco the benchmark's
    ``delete_mean``, half of each training row replaced, kept the three
    attacks at 1.0).  Each value carries the information ``-log2 p``, with
    ``p`` its frequency in *real* (training and holdout): a near-unique number
    ~log2(n) bits, a common category almost none.  The score of a record is
    the information of the values it shares with its best-matching synthetic
    row.  Missing values and values absent from *real* never match.

    A synthetic table of more than *max_synth* rows is searched through a
    fixed sample of them (``None`` = all).

    Returns, per frame, ``(bits, index of the best synthetic row)``.
    """
    kept = None
    if max_synth is not None and len(synth) > max_synth:
        kept = np.sort(np.random.default_rng(0).choice(len(synth), max_synth, replace=False))
        synth = synth.iloc[kept]
    codes = [[] for _ in frames]
    infos = [[] for _ in frames]
    s_codes = []
    for col in columns:
        mode = _key_mode([real[col], synth[col], *(f[col] for f in frames)])
        r_keys = _value_keys(real[col], mode)
        freq = pd.Series(r_keys[r_keys != ""]).value_counts(normalize=True)
        vocab = {v: i for i, v in enumerate(freq.index)}
        s_codes.append(np.array([vocab.get(v, -3) if v != "" else -3 for v in _value_keys(synth[col], mode)], dtype=np.int64))
        for i, frame in enumerate(frames):
            keys = _value_keys(frame[col], mode)
            codes[i].append(np.array([vocab.get(v, -2) if v != "" else -1 for v in keys], dtype=np.int64))
            p = pd.Series(keys).map(freq).to_numpy(dtype=float)
            ok = np.isfinite(p) & (p > 0)
            infos[i].append(np.where(ok, -np.log2(np.where(ok, p, 1.0)), 0.0))
    s = np.column_stack(s_codes)
    # Each block is chunk × rows of synth × columns: keep it under ~256 MB on large tables.
    chunk = max(1, min(chunk, int(2**25 // max(s.size, 1))))
    out = []
    for c_list, w_list in zip(codes, infos, strict=True):
        c, w = np.column_stack(c_list), np.column_stack(w_list)
        bits = np.empty(len(c))
        best = np.empty(len(c), dtype=np.int64)
        for start in range(0, len(c), chunk):
            shared = ((c[start : start + chunk, None, :] == s[None, :, :]) * w[start : start + chunk, None, :]).sum(axis=2)
            best[start : start + chunk] = shared.argmax(axis=1)
            bits[start : start + chunk] = shared.max(axis=1)
        out.append((bits, best if kept is None else kept[best]))
    return out


def cap_records(frame: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """At most *n* records of *frame*, drawn without replacement and kept in order."""
    if len(frame) <= n:
        return frame
    return frame.iloc[np.sort(rng.choice(len(frame), n, replace=False))]


def audit_strategies(pairs: list[tuple[np.ndarray, np.ndarray]], names: list[str]) -> dict[str, Any]:
    """Audit several attack strategies and keep the strongest one.

    An adversary uses whichever strategy works best, so the attack's ε is the
    largest of the strategies' lower bounds.  Each bound is computed at
    confidence ``1 − (1 − AUDIT_CONFIDENCE)/k`` (Bonferroni over the ``k``
    strategies), so an independent sample still scores 1 with probability
    ``AUDIT_CONFIDENCE``.
    """
    confidence = 1.0 - (1.0 - AUDIT_CONFIDENCE) / len(pairs)
    audits = [audit_epsilon(m, h, confidence=confidence) for m, h in pairs]
    eps = [a["epsilon"] if np.isfinite(a["epsilon"]) else -np.inf for a in audits]
    k = int(np.argmax(eps))
    best = dict(audits[k])
    best["strategy"] = names[k]
    best["strategies"] = {n: a["epsilon"] for n, a in zip(names, audits, strict=True)}
    return best


def relative_closeness(d_synth: np.ndarray, d_local: np.ndarray) -> np.ndarray:
    """Attack score ``−log((d_synth + c) / (d_local + c))``, higher = closer.

    *d_synth*: distance of each record to the closest synthetic record;
    *d_local*: to the closest other real record (its neighbourhood).  ``c``
    (1 % of the median local distance) keeps exact duplicates finite.
    """
    d_synth = np.asarray(d_synth, dtype=float)
    d_local = np.asarray(d_local, dtype=float)
    pooled = d_local[np.isfinite(d_local)]
    c = 0.01 * float(np.median(pooled)) if len(pooled) else 0.0
    c = c if c > 0 else 1e-6
    return -np.log((d_synth + c) / (d_local + c))


def interpret_epsilon(score: float, epsilon: float, what: str) -> str:
    """Human-readable reading of an ``audit_epsilon`` result."""
    if not np.isfinite(epsilon):
        return "Not measured"
    msg = f"ε ≥ {epsilon:.2f} at {AUDIT_CONFIDENCE:.0%} confidence"
    if epsilon <= 0:
        return f"No leak detected - no more {what} than for unseen records ({msg})"
    if score >= 0.7:
        return f"Small leak - a few more {what} than for unseen records ({msg})"
    if score >= 0.3:
        return f"Leak - {what} clearly in excess of unseen records ({msg})"
    return f"Severe leak - {what} far in excess of unseen records ({msg})"


def signed_excess_risk(p_attack: float, p_baseline: float) -> float:
    """``(p_attack − p_baseline) / (1 − p_baseline)`` without clipping.

    Diagnostic only (``details["excess_risk_signed"]``): the score clips it at
    0, so "as private as the reference" and "more private than it" tie at 1;
    the signed value keeps them apart.  NaN when the baseline is 1.
    """
    if not np.isfinite(p_attack) or not np.isfinite(p_baseline) or p_baseline >= 1.0:
        return float("nan")
    return float((p_attack - p_baseline) / (1.0 - p_baseline))


def probability_less(a: np.ndarray, b: np.ndarray, decimals: int = 9) -> float:
    """``P(A < B) + 0.5·P(A = B)`` for draws from samples *a* and *b*.

    Mann–Whitney statistic (the AUC of a score that ranks *a* below *b*).
    Values are rounded to *decimals* so floating-point noise around exact
    copies counts as a tie.
    """
    a = np.round(np.asarray(a, dtype=float), decimals)
    b = np.round(np.asarray(b, dtype=float), decimals)
    ranks = rankdata(np.concatenate([a, b]))
    u_a = ranks[: len(a)].sum() - len(a) * (len(a) + 1) / 2.0
    return float(1.0 - u_a / (len(a) * len(b)))


# =============================================================================
# Base Privacy Metric
# =============================================================================


class BasePrivacyMetric(ABC):
    """
    Abstract base class for all privacy metrics.

    Provides common infrastructure for measuring privacy preservation
    in synthetic data generation.

    Attributes:
        name: Short identifier for the metric
        family: Always "privacy" for privacy metrics
        purpose_tags: set of tags describing metric purpose
        higher_is_better: Always True (1 = most private)
    """

    name: str = "base_privacy"
    family: str = "privacy"
    purpose_tags: set = {"privacy"}
    higher_is_better: bool = True  # 1 = most private

    def __init__(self):
        self._real_data: pd.DataFrame | None = None
        self._synth_data: pd.DataFrame | None = None
        self._context: dict[str, Any] = {}

    def fit(
        self,
        real_data: pd.DataFrame,
        synth_data: pd.DataFrame,
        context: dict[str, Any],
    ) -> BasePrivacyMetric:
        """
        Initialize metric with data and context.

        Args:
            real_data: Original dataset
            synth_data: Synthetic dataset
            context: Execution context (may contain generative_model, etc.)

        Returns:
            Self for method chaining
        """
        # Fair comparison: use same sample size for both datasets
        n_samples = min(len(real_data), len(synth_data))
        seed = context.get("seed", 42)

        # Sample down to fair comparison size
        if len(real_data) > n_samples:
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(real_data), size=n_samples, replace=False)
            real_data = real_data.iloc[indices].reset_index(drop=True)

        if len(synth_data) > n_samples:
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(synth_data), size=n_samples, replace=False)
            synth_data = synth_data.iloc[indices].reset_index(drop=True)

        self._real_data = real_data
        self._synth_data = synth_data
        self._context = context
        self._on_fit()
        return self

    def _on_fit(self) -> None:
        """Hook for subclasses to perform additional setup after fit."""
        return

    @abstractmethod
    def compute(self) -> MetricResult:
        """
        Compute the privacy metric.

        Returns:
            MetricResult with value in [0, 1] where 1 = most private
        """
        pass

    def _get_numeric_columns(self) -> list[str]:
        """Get common numeric columns between real and synthetic data."""
        if self._real_data is None or self._synth_data is None:
            return []
        real_num = set(self._real_data.select_dtypes(include=[np.number]).columns)
        synth_num = set(self._synth_data.select_dtypes(include=[np.number]).columns)
        return sorted(real_num & synth_num)

    def _get_categorical_columns(self) -> list[str]:
        """Get common categorical columns between real and synthetic data."""
        if self._real_data is None or self._synth_data is None:
            return []
        real_cat = set(self._real_data.select_dtypes(include=["object", "category"]).columns)
        synth_cat = set(self._synth_data.select_dtypes(include=["object", "category"]).columns)
        return sorted(real_cat & synth_cat)

    def _create_error_result(self, error_msg: str) -> MetricResult:
        """Create a MetricResult for error cases (NaN so aggregation skips it)."""
        return MetricResult(
            id=f"privacy.{self.name}",
            value=float("nan"),
            details={"error": error_msg},
            family=self.family,
            purpose_tags=self.purpose_tags,
        )

    def _create_not_applicable_result(self, reason: str) -> MetricResult:
        """The metric needs something these data or this config do not have."""
        return MetricResult.not_applicable(
            f"privacy.{self.name}", self.family, reason, self.purpose_tags
        )


# =============================================================================
# Dataset-based Privacy Metrics
# =============================================================================


class DatasetPrivacyMetric(BasePrivacyMetric, MetricBase):
    """
    Base class for dataset-based privacy metrics.

    These metrics measure privacy by comparing statistical properties
    between real and synthetic datasets, without requiring access to
    the generative model.

    Inherits from MetricBase for the shared ``StatsStore`` cache. Reads two
    optional context keys:

    - ``context["holdout"]``: real rows not used to train the generator,
      with the same columns and preprocessing as the real table. It is the
      control for the membership and linkage attacks.
    - ``context["target"]``: the target column, default sensitive attribute
      and never a quasi-identifier.

    Subcategories:
        - Attribute Inference: MIA, Inference Attack
        - Reidentification: k-Anonymity, l-Diversity, t-Closeness, Record Linkage
        - Empirical Similarity: DCR, NNAA
    """

    purpose_tags: set = {"privacy", "dataset_based"}

    def _resolve_quasi_identifiers(
        self, explicit: list[str] | None, sensitive: list[str]
    ) -> tuple[list[str], list[str]]:
        """Return ``(quasi_identifiers, dropped)``; target and sensitive are never QIs."""
        common = self._common_columns()
        forbidden = set(sensitive)
        target = self._get_target()
        if target:
            forbidden.add(target)
        candidates = [c for c in explicit if c in common] if explicit else common
        qis = [c for c in candidates if c not in forbidden]
        dropped = [c for c in candidates if c in forbidden]
        return qis, dropped

    def __init__(self):
        BasePrivacyMetric.__init__(self)
        MetricBase.__init__(self)

    def fit(
        self,
        real_data: pd.DataFrame,
        synth_data: pd.DataFrame,
        context: dict[str, Any],
    ) -> DatasetPrivacyMetric:
        """Initialize with data, setting up both base classes."""
        BasePrivacyMetric.fit(self, real_data, synth_data, context)
        # Dataset metrics deliberately re-bind the full tables: subsampling the
        # real table would hide copies of the rows left out.
        self._setup(real_data, synth_data, context)
        self._fingerprint = None
        self._holdout_state: dict[str, Any] | None = None
        return self

    # ------------------------------------------------------------------
    # Context: holdout and target
    # ------------------------------------------------------------------

    def _shared_columns(self) -> list[str]:
        """Columns present in both the real and the synthetic tables (real order).

        One column per variable: the numeric ``<name>__ord`` shadow of an
        ordinal is dropped when its categorical twin is present (both used
        to enter as independent columns, so an ordinal weighed double in the
        distances and in the quasi-identifiers).
        """
        cols = [c for c in self._real_data.columns if c in self._synth_data.columns]
        shadows = ordinal_shadows(cols)
        return [c for c in cols if c not in shadows]

    def _resolve_holdout(self) -> dict[str, Any]:
        """Inspect ``context["holdout"]`` once per fit and warn about problems.

        A holdout that lacks some of the shared columns is still used, over
        the columns it has (the metrics then drop the others, see
        ``_common_columns``); it used to be discarded without notice, which
        silently switched every metric to its no-holdout substitute. A holdout
        with none of the shared columns, an empty one or a non-DataFrame is
        ignored, also with a warning. ``details["holdout"]`` records it.
        """
        if getattr(self, "_holdout_state", None) is not None:
            return self._holdout_state
        raw = self._context.get("holdout")
        state: dict[str, Any] = {
            "provided": raw is not None,
            "frame": None,
            "missing_columns": [],
            "ignored_reason": None,
        }
        if raw is None:
            pass
        elif not isinstance(raw, pd.DataFrame):
            state["ignored_reason"] = f"not a DataFrame ({type(raw).__name__})"
        elif len(raw) == 0:
            state["ignored_reason"] = "empty"
        else:
            shared = self._shared_columns()
            missing = [c for c in shared if c not in raw.columns]
            state["missing_columns"] = missing
            if shared and len(missing) == len(shared):
                state["ignored_reason"] = "it has none of the shared columns"
            else:
                state["frame"] = raw.reset_index(drop=True)
                if missing:
                    warnings.warn(
                        f"privacy: the holdout lacks shared column(s) {missing}; the "
                        "privacy metrics that use it drop those columns",
                        HoldoutColumnsWarning,
                        stacklevel=3,
                    )
        if state["ignored_reason"]:
            warnings.warn(
                f"privacy: the holdout is ignored ({state['ignored_reason']}); the "
                "privacy metrics fall back to their no-holdout reference",
                HoldoutColumnsWarning,
                stacklevel=3,
            )
        self._holdout_state = state
        return state

    def _ignore_holdout(self, reason: str) -> None:
        """Stop using the holdout in this metric (e.g. it lacks the sensitive column)."""
        state = self._resolve_holdout()
        if state["frame"] is None:
            return
        state["frame"] = None
        state["ignored_reason"] = reason
        warnings.warn(
            f"privacy.{self.name}: the holdout is ignored ({reason}); the metric falls "
            "back to its no-holdout reference",
            HoldoutColumnsWarning,
            stacklevel=3,
        )

    def _get_holdout(self) -> pd.DataFrame | None:
        """Return ``context["holdout"]`` (real rows NOT used to train the generator).

        ``None`` when there is no usable holdout (see ``_resolve_holdout``).
        """
        return self._resolve_holdout()["frame"]

    def _no_holdout_result(self, what: str) -> MetricResult:
        """Result of a metric that needs non-members and has no usable holdout.

        Not applicable when none was given; an error when one was given and
        could not be used.  The real→real leave-one-out substitute is not a
        control for a membership audit: it measures another release (the
        real table), so a generator farther from the real rows than they are
        from each other offset its copies (5–10 % of copies kept DCR and
        record linkage at 1.00).
        """
        message = (
            f"{what} needs non-members: pass real rows not used for training in context['holdout']"
        )
        reason = self._resolve_holdout()["ignored_reason"]
        if reason:
            return self._create_error_result(f"{message} (the holdout given was ignored: {reason})")
        return self._create_not_applicable_result(message)

    def _nn_distance_to_synth(
        self, frames: list[pd.DataFrame], columns: list[str]
    ) -> list[np.ndarray]:
        """Distance from every row of each frame to its closest synthetic record.

        Shared mixed-type space of the real table; rounded to 1e-9 so that
        floating-point noise around exact copies counts as a tie.
        """
        encoder = self._real_distance_space(columns)["encoder"]
        index = NearestNeighbors(n_neighbors=1).fit(encoder.transform(self._synth_data))
        return [np.round(index.kneighbors(encoder.transform(f))[0][:, 0], 9) for f in frames]

    def _local_scale(self, frames: list[pd.DataFrame], columns: list[str]) -> list[np.ndarray]:
        """Distance from every row of each frame to its closest *other* row of
        the pooled frames (members and non-members alike, so the scale does
        not tell them apart)."""
        encoder = self._real_distance_space(columns)["encoder"]
        mats = [encoder.transform(f) for f in frames]
        pool = sparse.vstack(mats, format="csr") if sparse.issparse(mats[0]) else np.vstack(mats)
        index = NearestNeighbors(n_neighbors=min(2, pool.shape[0])).fit(pool)
        out = []
        for m in mats:
            d = index.kneighbors(m, n_neighbors=min(2, pool.shape[0]))[0]
            out.append(np.round(d[:, -1], 9))
        return out

    def _holdout_details(self) -> dict[str, Any]:
        """``{"holdout": {...}}`` for ``details`` when a holdout was passed, else ``{}``."""
        state = self._resolve_holdout()
        if not state["provided"]:
            return {}
        return {
            "holdout": {
                "used": state["frame"] is not None,
                "missing_columns": list(state["missing_columns"]),
                "ignored_reason": state["ignored_reason"],
            }
        }

    def _create_error_result(self, error_msg: str) -> MetricResult:
        result = super()._create_error_result(error_msg)
        result.details.update(self._holdout_details())
        return result

    def _create_not_applicable_result(self, reason: str) -> MetricResult:
        result = super()._create_not_applicable_result(reason)
        result.details.update(self._holdout_details())
        return result

    def _common_columns(self) -> list[str]:
        """Columns present in the real, synthetic (and holdout, if used) tables."""
        cols = self._shared_columns()
        holdout = self._get_holdout()
        if holdout is not None:
            cols = [c for c in cols if c in holdout.columns]
        return cols

    def _get_target(self) -> str | None:
        """Return ``context["target"]`` when it names a column of both tables."""
        target = self._context.get("target")
        if not isinstance(target, str):
            return None
        shared = self._shared_columns()
        # A datetime target is cast to ``<name>__ts``.
        for candidate in (target, f"{target}__ts"):
            if candidate in shared:
                return candidate
        return None

    def _resolve_sensitive_columns(self, explicit: list[str] | None) -> list[str]:
        """Explicit sensitive columns, else ``[context["target"]]``, else ``[]``.

        The holdout is the control of the attack on these columns, so a holdout
        that lacks any of them is not used by the calling metric.
        """
        shared = set(self._shared_columns())
        if explicit:
            sensitive = [c for c in explicit if c in shared]
        else:
            target = self._get_target()
            sensitive = [target] if target else []
        holdout = self._get_holdout()
        if holdout is not None:
            absent = [c for c in sensitive if c not in holdout.columns]
            if absent:
                self._ignore_holdout(f"it lacks the sensitive column(s) {absent}")
        return sensitive

    # ------------------------------------------------------------------
    # Generalisation (quasi-identifiers, sensitive values)
    # ------------------------------------------------------------------

    # Numeric columns with at most this many distinct real values keep their
    # levels (each value snaps to the nearest real level); the rest are binned.
    _MAX_EXACT_NUMERIC_LEVELS: int = 10

    def _generalise(self, data: pd.DataFrame, columns: list[str], n_bins: int) -> pd.DataFrame:
        """Generalise *columns* with cut points fitted on the real table.

        Numeric: ``n_bins`` quantile bins of the real column or, for columns
        with few distinct real values, the nearest real level (so ``1.0`` and
        ``1`` or ``0.98`` fall together). Missing values get their own code.
        Other columns: their string value.
        """
        out = {}
        for col in columns:
            real_col = self._real_data[col]
            if not _is_numeric(real_col):
                out[col] = _as_category_strings(data[col])
                continue
            real_vals = pd.to_numeric(real_col, errors="coerce").dropna().to_numpy(float)
            levels = np.unique(real_vals)
            if len(levels) <= self._MAX_EXACT_NUMERIC_LEVELS:
                edges = (levels[:-1] + levels[1:]) / 2.0
            else:
                quantiles = np.linspace(0.0, 1.0, max(n_bins, 1) + 1)[1:-1]
                edges = np.unique(np.quantile(real_vals, quantiles))
            values = pd.to_numeric(data[col], errors="coerce").to_numpy(float)
            codes = np.searchsorted(edges, values, side="right").astype(np.int64)
            codes[np.isnan(values)] = -1
            out[col] = codes.astype(str)
        return pd.DataFrame(out, index=range(len(data)))

    # ------------------------------------------------------------------
    # Distance space shared by MIA, DCR, NNAA and record linkage
    # ------------------------------------------------------------------

    def _real_fingerprint(self) -> str:
        """Cheap content fingerprint of the (sampled) real table for cache keys."""
        if getattr(self, "_fingerprint", None) is None:
            hashed = pd.util.hash_pandas_object(self._real_data, index=False)
            self._fingerprint = f"{self._real_data.shape}:{int(hashed.sum())}"
        return self._fingerprint

    def _distance_columns(self, columns: list[str]) -> tuple[list[str], set[str]]:
        """Columns of the distance space and which of them are ordinal ranks.

        METIS represents an ordinal in two views, its level and its rank
        ``i/(K-1)`` (the ``<name>__ord`` column the caster adds); every
        computation uses the view that matches what it measures.  A distance
        measures how far apart two records are, so an ordinal enters by its
        rank only (unscaled, see ``MixedRecordEncoder``).  As a category, one
        level away cost as much as the opposite end: a generator that copied
        real records and moved their ordinals by one level looked private
        (DCR/NNAA/MIA/linkage 0.93-0.95 on rrhh and cardio).
        """
        out: list[str] = []
        ranks: set[str] = set()
        frames = [self._real_data, self._synth_data]
        for col in columns:
            rank = f"{col}{ORDINAL_SHADOW_SUFFIX}"
            if all(rank in f.columns for f in frames):
                out.append(rank)
                ranks.add(rank)
            else:
                out.append(col)
        return out, ranks

    def _real_distance_space(self, columns: list[str]) -> dict[str, Any]:
        """Encoder, encoded real matrix and real 1-NN index, cached in ``StatsStore``.

        DCR, NNAA, MIA and record linkage share the same encoder and index for
        a given real table and column set, so they are fitted once per
        evaluation instead of once per metric.
        """
        others = [self._get_holdout(), self._synth_data]
        tag = ":".join(_frame_fingerprint(t) for t in others)
        key = f"privacy:distance_space:{self._real_fingerprint()}:{tag}:{'|'.join(columns)}"

        def build() -> dict[str, Any]:
            distance_columns, ranks = self._distance_columns(columns)
            encoder = MixedRecordEncoder(
                self._real_data, distance_columns, unscaled=ranks, vocabulary_from=others
            )
            matrix = encoder.transform(self._real_data)
            index = NearestNeighbors(n_neighbors=min(2, matrix.shape[0])).fit(matrix)
            return {"encoder": encoder, "real_matrix": matrix, "real_index": index}

        return self._get_cached_or_compute(key, build)

    def _nn_distance_to_real(self, data: pd.DataFrame, columns: list[str]) -> np.ndarray:
        """Distance from every row of *data* to its closest real record."""
        space = self._real_distance_space(columns)
        distances, _ = space["real_index"].kneighbors(
            space["encoder"].transform(data), n_neighbors=1
        )
        return distances[:, 0]

    def _real_leave_one_out_distances(self, columns: list[str]) -> np.ndarray:
        """Distance from every real record to its closest *other* real record."""
        key = f"privacy:real_loo:{self._real_fingerprint()}:{'|'.join(columns)}"

        def build() -> np.ndarray:
            space = self._real_distance_space(columns)
            distances, _ = space["real_index"].kneighbors(space["real_matrix"], n_neighbors=2)
            return distances[:, 1]

        return self._get_cached_or_compute(key, build)

    def _reference_distances(self, columns: list[str]) -> tuple[np.ndarray, str]:
        """Reference distribution of distances-to-closest-real-record.

        With a holdout, it is holdout→real: what a fresh sample from the same
        population looks like from the real table. Without it, the documented
        substitute is real→real leave-one-out (each real record against the
        rest), which plays the same role at the same sample size.
        """
        holdout = self._get_holdout()
        if holdout is not None:
            return self._nn_distance_to_real(holdout, columns), "holdout"
        return self._real_leave_one_out_distances(columns), "real_leave_one_out"


# =============================================================================
# Attribute Inference Metrics
# =============================================================================


class AttackMetric(DatasetPrivacyMetric):
    """
    Base class of the audited privacy attacks.

    Each attack scores how much closer training records sit to the synthetic
    table than holdout records do, audited in the tail (ε lower bound, score
    e^-ε).  The three measure the same signal and form one subcategory,
    ``privacy.dataset_based.attacks`` (2026-10-19).

    Includes:
        - MIA (membership inference)
        - Inference Attack (attribute inference, nearest neighbour)
        - Record Linkage
    """

    purpose_tags: set = {"privacy", "dataset_based", "attacks"}


# =============================================================================
# Syntactic privacy models (k-anonymity, l-diversity, t-closeness)
# =============================================================================


class SyntacticMetric(DatasetPrivacyMetric):
    """
    Base class of the syntactic privacy models (subcategory
    ``privacy.dataset_based.syntactic``).

    k-anonymity, l-diversity and t-closeness are evaluated as attacks on the
    *released synthetic table*, relative to a control:

    - victims = real records (members of the generator's training set);
    - the attacker knows a victim's quasi-identifiers (QIs), looks up the
      matching equivalence class in the synthetic table and succeeds if that
      class violates the property (k, l or t) for the victim;
    - the same attack against control victims gives the baseline: holdout
      records against the same synthetic table when ``context["holdout"]``
      exists, otherwise real records against the real table without
      themselves (leave-one-out: what releasing *other* real records would
      disclose);
    - score = 1 − excess risk over the baseline (see ``excess_risk_score``).

    So a synthetic table only loses points for disclosing *its training
    records* more than an independent sample would, which is what the
    plain synthetic-only statistics could not tell (with 400 rows they gave
    ≈0 to any generator, holdout included).

    Quasi-identifiers are resolved identically for the three metrics:
    explicit ``quasi_identifiers`` or, by default, every shared column except
    the target and the sensitive columns. The target is never a QI. Numeric
    QIs are generalised into ``n_bins`` quantile bins of the real data.

    Includes:
        - k-Anonymity
        - l-Diversity
        - t-Closeness
    """

    purpose_tags: set = {"privacy", "dataset_based", "syntactic"}

    # Sensitive categoricals with more levels keep the most frequent ones and
    # pool the rest, so the class × value table stays small.
    _MAX_SENSITIVE_LEVELS: int = 50

    def _qi_keys(self, data: pd.DataFrame, qis: list[str], n_bins: int) -> np.ndarray:
        """One uint64 key per row identifying its generalised QI combination."""
        generalised = self._generalise(data, qis, n_bins)
        return pd.util.hash_pandas_object(generalised, index=False).to_numpy()

    def _sensitive_codes(
        self, frames: list[pd.DataFrame], column: str, n_bins: int
    ) -> tuple[list[np.ndarray], int, np.ndarray | None]:
        """Integer codes of the (generalised) sensitive column, shared across *frames*.

        Also returns the position of each code on the column's order (NaN for
        codes outside it, such as missing values), or ``None`` for a nominal
        column. Numeric columns are ordered by their bins; an ordinal by the
        rank of its ``<name>__ord`` shadow in the real table.
        """
        values = [self._generalise(f, [column], n_bins)[column].to_numpy() for f in frames]
        real_values = self._generalise(self._real_data, [column], n_bins)[column]
        keep = set(real_values.value_counts().index[: self._MAX_SENSITIVE_LEVELS - 1])
        if real_values.nunique() > self._MAX_SENSITIVE_LEVELS:
            values = [np.where(np.isin(v, list(keep)), v, "__other__") for v in values]
        codes, uniques = pd.factorize(np.concatenate(values))
        out, start = [], 0
        for v in values:
            out.append(codes[start : start + len(v)])
            start += len(v)
        return out, len(uniques), self._sensitive_order(column, uniques)

    def _sensitive_order(self, column: str, uniques: np.ndarray) -> np.ndarray | None:
        """Order position of each generalised value (see ``_sensitive_codes``)."""
        if _is_numeric(self._real_data[column]):
            # ``_generalise`` codes numeric values as their bin index, -1 if missing.
            pos = pd.to_numeric(pd.Series(uniques), errors="coerce").to_numpy(float)
            pos[pos < 0] = np.nan
            return pos
        shadow = column + ORDINAL_SHADOW_SUFFIX
        if shadow not in self._real_data.columns:
            return None
        ranks = (
            pd.to_numeric(self._real_data[shadow], errors="coerce")
            .groupby(_as_category_strings(self._real_data[column]))
            .mean()
        )
        return pd.Series(uniques).map(ranks).to_numpy(float)

    def _control(self) -> tuple[pd.DataFrame, pd.DataFrame, bool, str]:
        """``(release, victims, leave_one_out, kind)`` for the baseline attack."""
        holdout = self._get_holdout()
        if holdout is not None:
            return self._synth_data, holdout, False, "holdout"
        return self._real_data, self._real_data, True, "real_leave_one_out"

    @staticmethod
    def _class_disclosure(
        release_keys: np.ndarray,
        release_sens: np.ndarray,
        victim_keys: np.ndarray,
        victim_sens: np.ndarray,
        *,
        n_values: int,
        leave_one_out: bool,
        order: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Per-victim view of the release equivalence class matching its QIs.

        Returns, for every victim: ``size`` of the matching class (0 if none),
        ``own`` count of the victim's sensitive value in it, ``top`` count of
        its most frequent value, ``distinct`` sensitive values, ``tvd`` between
        the class and the release's global sensitive distribution, ``emd``
        between them (``_emd``, with the value *order* from
        ``_sensitive_codes``), and ``p_own``/``g_own``, the share of the
        victim's value in the class and globally. With ``leave_one_out`` the
        victim's own row is removed from its class (the release then contains
        the victim).
        """
        n_victims = len(victim_keys)
        classes, release_class = np.unique(release_keys, return_inverse=True)
        counts = np.zeros((len(classes), n_values))
        np.add.at(counts, (release_class, release_sens), 1.0)

        pos = np.clip(np.searchsorted(classes, victim_keys), 0, max(len(classes) - 1, 0))
        matched = classes[pos] == victim_keys if len(classes) else np.zeros(n_victims, bool)
        rows = np.zeros((n_victims, n_values))
        rows[matched] = counts[pos[matched]]
        idx = np.arange(n_victims)
        if leave_one_out:
            rows[idx[matched], victim_sens[matched]] -= 1.0

        size = rows.sum(axis=1)
        global_dist = np.bincount(release_sens, minlength=n_values) / max(len(release_sens), 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            shares = np.where(size[:, None] > 0, rows / size[:, None], 0.0)
        tvd = np.where(size > 0, 0.5 * np.abs(shares - global_dist).sum(axis=1), 0.0)
        emd = tvd if order is None else np.where(size > 0, _emd(shares, global_dist, order), 0.0)
        return {
            "size": size,
            "own": rows[idx, victim_sens],
            "top": rows.max(axis=1) if n_values else np.zeros(n_victims),
            "distinct": (rows > 0).sum(axis=1),
            "tvd": tvd,
            "emd": emd,
            "p_own": shares[idx, victim_sens],
            "g_own": global_dist[victim_sens],
        }

    def _attack_rates(
        self,
        qis: list[str],
        sensitive: str | None,
        n_bins: int,
        event,
    ) -> dict[str, Any]:
        """Success rate of *event* on training victims and on control victims;
        the score is ``1 − excess risk`` and the ε audit is a diagnostic.

        *event* maps the dict returned by ``_class_disclosure`` to a boolean
        array (one entry per victim).  The callers require a holdout
        (``_no_holdout_result``); the leave-one-out control is kept only for
        direct calls.
        """
        release_c, victims_c, loo, kind = self._control()
        frames = [self._synth_data, self._real_data, release_c, victims_c]
        order = None
        if sensitive is None:
            sens = [np.zeros(len(f), dtype=np.int64) for f in frames]
            n_values = 1
        else:
            sens, n_values, order = self._sensitive_codes(frames, sensitive, n_bins)
        keys = [self._qi_keys(f, qis, n_bins) for f in frames]

        train = self._class_disclosure(
            keys[0], sens[0], keys[1], sens[1], n_values=n_values, leave_one_out=False, order=order
        )
        control = self._class_disclosure(
            keys[2], sens[2], keys[3], sens[3], n_values=n_values, leave_one_out=loo, order=order
        )
        hit_train = np.asarray(event(train), dtype=float)
        hit_control = np.asarray(event(control), dtype=float)
        p_train, p_control = float(np.mean(hit_train)), float(np.mean(hit_control))
        privacy, risk = excess_risk_score(p_train, p_control)
        # The class-based models (k, l, t) keep the average excess rate: their
        # events live on coarse generalised classes, and the ε audit of a
        # binary event was erratic (adult: 0.20 with 10 % of copies, 1.00
        # with 25 %).  The ε bound stays in ``details``.
        audit = audit_epsilon(hit_train, hit_control)
        return {
            "privacy_score": privacy,
            "epsilon_lower_bound": audit["epsilon"],
            "risk": risk,
            "excess_risk_signed": signed_excess_risk(p_train, p_control),
            "attack_rate_training": p_train,
            "attack_rate_control": p_control,
            "control": kind,
            "n_training_victims": len(keys[1]),
            "n_control_victims": len(keys[3]),
            "ground_distance": "equal" if order is None else "ordered",
            "train_view": train,
        }


# =============================================================================
# Empirical Similarity Metrics
# =============================================================================


class EmpiricalSimilarityMetric(DatasetPrivacyMetric):
    """
    Base class for empirical similarity privacy metrics.

    These metrics measure privacy by computing distances between
    synthetic and real records.

    Includes:
        - DCR (Distance to Closest Record)
        - NNAA (Nearest Neighbor Adversarial Accuracy)
    """

    purpose_tags: set = {"privacy", "dataset_based", "empirical_similarity"}


# =============================================================================
# Mechanism-based Privacy Metrics
# =============================================================================


class MechanismPrivacyMetric(BasePrivacyMetric):
    """
    Base class for mechanism-based privacy metrics.

    These metrics require access to the generative model to verify
    privacy guarantees provided by the generation mechanism.

    The generative model, when available, is passed via
    ``context["generative_model"]``.

    Configuration:
        - epsilon: privacy budget declared by the user. It takes precedence
          over the one the model reports; without either the metric has no
          guarantee to assess (there is no default budget).

    Includes:
        - Differential Privacy verification
    """

    purpose_tags: set = {"privacy", "mechanism_based"}

    def __init__(self, epsilon: float | None = None):
        """
        Initialize mechanism-based metric.

        Args:
            epsilon: Declared privacy budget. ``None`` = take it from the model.
        """
        super().__init__()
        self._epsilon = epsilon
        self._generative_model: Any | None = None

    def _on_fit(self) -> None:
        """Extract the generative model from the context."""
        self._generative_model = self._context.get("generative_model")

    @property
    def epsilon(self) -> float | None:
        """The declared privacy budget (``None`` if not configured)."""
        return self._epsilon
