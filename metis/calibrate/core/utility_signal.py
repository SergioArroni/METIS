"""Whether the real table has signal for the utility metrics, decided once.

The utility score is undefined on a table whose features do not predict the
target beyond sampling noise: TTR gain ≤ ``UTILITY_SIGNAL_MULTIPLE``·√2·σ (module
docstring of ``ml_efficiency_metric``).  Inside one evaluation σ comes from
five half-samples, and a standard deviation from five values is itself
uncertain by ~35 % (1/√(2·4)), so on a table near the threshold the decision
flipped with the seed: rrhh was not applicable on 4 of the 5 seeds of the
2026-10-10 benchmark, and a generator's utility depended on which seeds
counted.

The calibration measures it once per real table: ``N_SPLITS`` splits of the
real table (60/20/20, as the calibration iterations and the benchmark), with
the training part at the size of the unseen part, the size every evaluation
compares at (``equal_sizes``); ``HALF_SAMPLES_PER_SPLIT`` half-samples on each.
σ is the pooled within-split spread (16 degrees of freedom, ~18 % uncertainty)
and the gain the mean over the 20 TTR fits.  The decision is stored with the
bounds and every evaluation that uses them follows it; the anchors are reused
only by evaluations of that same size.
"""

import logging
from typing import Any

import pandas as pd

from metis.calibrate.strategies.base import calibration_seed, tercile_split
from metis.shared.splits import DEFAULT_SPLIT
from metis.infrastructure.metrics.utility.ml_efficiency_metric import pooled_utility_signal

N_SPLITS = 4
HALF_SAMPLES_PER_SPLIT = 5
# The probe runs this metric: the others share its TTR and half-samples.
PROBE_METRIC = "utility.ml_efficiency"


def measure_utility_signal(
    evaluator: Any,
    real_data: pd.DataFrame,
    sample_size: int,
    base_seed: int,
    logger: logging.Logger | None = None,
    stratify_by: str | None = None,
    fractions: tuple[float, float, float] | None = None,
) -> dict[str, Any] | None:
    """The signal decision of *real_data*, or None when it cannot be measured.

    *evaluator* is an ``InMemoryEvaluator`` on the calibration's config.
    None when the config has no utility metric, or no split could be
    measured (no target, a single class...): the evaluations then decide on
    their own, as without calibration.
    """
    logger = logger or logging.getLogger(__name__)
    if not evaluator.metrics_of(("utility",)):
        return None
    probes: list[dict[str, Any]] = []
    skipped: list[str] = []
    rows = 0
    for i in range(N_SPLITS):
        seed = calibration_seed(base_seed, i)
        train, holdout, unseen = tercile_split(real_data, sample_size, seed, stratify_by, fractions or DEFAULT_SPLIT)
        # Evaluations compare real and synthetic at the unseen size (equal_sizes).
        train = train.sample(n=min(len(unseen), len(train)), random_state=seed).reset_index(drop=True)
        rows = len(train)
        (result,) = evaluator.evaluate_results(
            train,
            train,
            seed,
            holdout=holdout,
            metrics=[PROBE_METRIC],
            ml_efficiency_config={"scale_only": True, "n_splits": HALF_SAMPLES_PER_SPLIT},
        )
        if "scale" in result.details:
            probes.append(result.details)
        else:
            skipped.append(str(result.details.get("reason") or result.details.get("error")))
    signal = pooled_utility_signal(probes)
    if signal is None:
        logger.warning("Utility signal not measured: %s", "; ".join(skipped) or "no pairs")
        return None
    signal.update(seeds=[calibration_seed(base_seed, i) for i in range(N_SPLITS)], rows_per_part=rows)
    if skipped:
        signal["skipped_splits"] = skipped
    logger.info(
        "Utility signal: %s (TTR gain %.4f, threshold %.4f, %d half-samples)",
        "applicable" if signal["applicable"] else "not applicable",
        signal["ttr_gain"],
        signal["threshold"],
        signal["n_half_samples"],
    )
    return signal
