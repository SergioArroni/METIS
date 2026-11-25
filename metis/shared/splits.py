"""The one way METIS splits a real table into training, holdout and unseen parts.

The benchmark splits each real table per seed, and the calibration splits it
once per iteration to build its references.  Both used to do it differently:
the benchmark with a split stratified by the target, the calibration with a
plain permutation.  A reference is only the right anchor for an evaluation if
it is built the way the evaluated tables were, so both now call
:func:`split_real` (2026-11-11).

* ``split_real`` splits a whole table in the fractions given, 60/20/20 by
  default, stratified by a classification target when every class has at
  least three rows.
* ``split_parts`` gives parts of a fixed size, ``3m / m / m`` by default,
  which is what the calibration needs to evaluate its references at the
  size of the evaluated tables.  ``calibration.split`` changes the
  proportions when the evaluated tables were split otherwise (the defect
  bank of the paper evaluates training, holdout and synthetic tables of the
  same size, so it calibrates with thirds).
* The calibration draws its splits with seeds offset by
  ``CALIBRATION_SEED_OFFSET``, so that its references never reuse the split
  of an evaluation seed: with the same function and the same seeds, the
  ``real_data`` baseline of seed 42 would be one of the realizations that
  set its own anchor.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

DEFAULT_SPLIT = (0.6, 0.2, 0.2)
CALIBRATION_SEED_OFFSET = 1000
MIN_ROWS_PER_CLASS = 3


def split_fractions(config: dict[str, Any] | None, section: str = "calibration") -> tuple[float, float, float]:
    """The fractions of the split, as the benchmark or the calibration reads them.

    ``benchmark.split`` and ``calibration.split`` are two keys for the same
    decision.  Each side reads its own and, when it is not set, the other
    one, so setting one of them moves both.  Both are set only when the
    evaluated tables are not split as the benchmark does it.
    """
    config = config or {}
    other = "benchmark" if section == "calibration" else "calibration"
    for name in (section, other):
        value = (config.get(name) or {}).get("split") if isinstance(config.get(name), dict) else None
        if value:
            if isinstance(value, str) or len(value) != 3:
                raise ValueError(f"{name}.split needs three fractions, got {value!r}")
            return tuple(float(f) for f in value)  # type: ignore[return-value]
    return DEFAULT_SPLIT


def config_split(config: dict[str, Any] | None) -> tuple[str | None, tuple[float, float, float]]:
    """``(stratify_by, fractions)`` of the calibration splits, from a METIS config."""
    config = config or {}
    data_cfg = config.get("data") if isinstance(config.get("data"), dict) else {}
    return stratify_target(data_cfg), split_fractions(config, "calibration")


def stratify_target(data_cfg: dict[str, Any] | None) -> str | None:
    """The column to stratify by: the target, unless the task is a regression."""
    data_cfg = data_cfg or {}
    target = data_cfg.get("target")
    if not target or str(data_cfg.get("task_type") or "").lower() == "regression":
        return None
    return str(target)


def _labels(real: pd.DataFrame, stratify_by: str | None) -> pd.Series | None:
    if stratify_by is None or stratify_by not in real.columns:
        return None
    column = real[stratify_by]
    # A missing target is a class of its own (pandas 3 keeps NaN through ``astype(str)``).
    labels = column.astype(object).where(column.notna(), "nan").astype(str)
    counts = labels.value_counts()
    if len(counts) > 1 and counts.min() >= MIN_ROWS_PER_CLASS:
        return labels
    return None


def _cut(frame: pd.DataFrame, train_size: float | int, seed: int, labels: pd.Series | None):
    """``train_test_split`` of *frame*, stratified by *labels* when the sizes allow it.

    A stratified cut needs two rows of every class and parts with at least as
    many rows as classes.  A class with 3 or 4 rows can leave a single row
    outside the training part, and a table whose size is not a multiple of the
    parts can leave fewer rows aside than classes.  The cut is then a plain
    one instead of an error.
    """
    from sklearn.model_selection import train_test_split

    if labels is not None:
        try:
            return train_test_split(frame, train_size=train_size, random_state=seed, stratify=labels.loc[frame.index])
        except ValueError:
            pass
    return train_test_split(frame, train_size=train_size, random_state=seed)


def split_real(
    real: pd.DataFrame,
    seed: int,
    fractions: tuple[float, float, float] = DEFAULT_SPLIT,
    stratify_by: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """``(train, holdout, unseen)``: disjoint parts of *real* for one seed.

    Stratified by *stratify_by* (a classification target) when every class has
    at least 3 rows.  The generators see ``train`` only; METIS evaluates the
    synthetic table against ``train`` with ``holdout`` as the never-seen
    reference; ``unseen`` is what ``real_data`` returns.
    """
    f_train, f_hold, f_unseen = fractions
    if min(fractions) <= 0 or abs(f_train + f_hold + f_unseen - 1.0) > 1e-9:
        raise ValueError(f"benchmark.split must be three positive fractions summing to 1: {fractions}")
    real = real.reset_index(drop=True)  # the labels are looked up by index, which must be unique
    labels = _labels(real, stratify_by)
    train, rest = _cut(real, f_train, seed, labels)
    holdout, unseen = _cut(rest, f_hold / (f_hold + f_unseen), seed, labels)
    return (
        train.reset_index(drop=True),
        holdout.reset_index(drop=True),
        unseen.reset_index(drop=True),
    )


def split_parts(
    real: pd.DataFrame,
    part_size: int,
    seed: int,
    stratify_by: str | None = None,
    fractions: tuple[float, float, float] = DEFAULT_SPLIT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """``(train, holdout, unseen)`` in the proportions of *fractions*, with ``m`` unseen rows.

    ``m = min(part_size, n · f_unseen)``.  With the default 60/20/20 the parts
    have ``3m``, ``m`` and ``m`` rows.  Draws the ``m / f_unseen`` rows it needs
    (stratified like the split) when the table is larger, then splits them as
    :func:`split_real` does, with integer sizes so that no rounding of the
    fractions moves a row from one part to another.
    """
    f_train, f_hold, f_unseen = fractions
    if min(fractions) <= 0 or abs(f_train + f_hold + f_unseen - 1.0) > 1e-9:
        raise ValueError(f"calibration.split must be three positive fractions summing to 1: {fractions}")
    real = real.reset_index(drop=True)  # the labels are looked up by index, which must be unique
    n = len(real)
    m = min(int(part_size), int(n * f_unseen + 1e-9))
    n_train, n_hold = round(m * f_train / f_unseen), round(m * f_hold / f_unseen)
    if m < 1 or n_train < 1 or n_hold < 1:
        raise ValueError(
            f"Calibration needs at least one row in every part: {n} rows and parts of {part_size} "
            f"with fractions {fractions} give {n_train}/{n_hold}/{m}"
        )
    total = n_train + n_hold + m
    if total < n:
        real, _ = _cut(real, total, seed, _labels(real, stratify_by))
    labels = _labels(real, stratify_by)
    train, rest = _cut(real, n_train, seed, labels)
    holdout, unseen = _cut(rest, n_hold, seed, labels)
    return (
        train.reset_index(drop=True),
        holdout.reset_index(drop=True),
        unseen.reset_index(drop=True),
    )
