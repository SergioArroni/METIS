"""Programmatic SDK for METIS.

Provides ``Evaluator`` and ``evaluate_from_config`` for library consumers
who want to call METIS from Python code rather than the CLI.

Example::

    from metis.interface.sdk import Evaluator

    config = {
        "data": {"schema": {"age": "continuous", "sex": "nominal"}, "target": "sex"},
        "metrics": ["fidelity.tvd", "privacy.dcr"],
        "evaluation": {
            # constructor arguments per metric, checked against its signature
            "metric_params": {"privacy.dcr": {"percentile_threshold": 1.0}},
        },
    }
    ev = Evaluator()  # caches the preprocessed real table
    # whether the target has signal, decided once for this real table
    signal = ev.measure_utility_signal(real, config, holdout=real_holdout)
    for synth in candidates:
        summary = ev.evaluate(real, synth, config, holdout=real_holdout, utility_signal=signal)
"""

from __future__ import annotations

import copy
import hashlib
import json
import warnings
from collections import OrderedDict
from typing import Any

import pandas as pd

from ..application.orchestrator import Orchestrator
from ..application.pipeline.preprocessor import ReferenceContext
from ..domain.entities import RunSummary
from ..infrastructure.metrics.privacy.privacy_base import MissingHoldoutWarning
from ..infrastructure.metrics.utility.ml_efficiency_metric import pooled_utility_signal
from ..infrastructure.runtime.config import validate_dataframe_config


def reference_key(real: pd.DataFrame, config: dict[str, Any]) -> str:
    """Content hash of the real table plus the ``data`` section of *config*.

    Two calls with equal keys preprocess the real table identically, so the
    preprocessed reference can be reused.
    """
    h = hashlib.sha256()
    h.update(json.dumps(list(map(str, real.columns))).encode())
    h.update(json.dumps([str(t) for t in real.dtypes]).encode())
    h.update(pd.util.hash_pandas_object(real, index=True).to_numpy().tobytes())
    h.update(json.dumps(config.get("data", {}), sort_keys=True, default=str).encode())
    return h.hexdigest()


# Half-samples of ``Evaluator.measure_utility_signal``: σ uncertain by ~16 %
# instead of ~35 % with the evaluation's five.
UTILITY_SIGNAL_HALF_SAMPLES = 20


class Evaluator:
    """High-level API for running evaluations programmatically.

    Args:
        cache_reference: Reuse the preprocessed real table (ID stripping,
            NaN removal, caster fitted on the real rows) across ``evaluate``
            calls with the same real table and ``data`` config.  The key is
            a content hash, so modifying the real table in place invalidates
            it.  Results are identical with and without the cache.
        max_cached_references: Size of the LRU cache.
        strict_config: Raise on unknown config keys instead of warning.
    """

    def __init__(
        self,
        *,
        cache_reference: bool = True,
        max_cached_references: int = 4,
        strict_config: bool = False,
    ) -> None:
        self._orchestrator = Orchestrator()
        self._cache_reference = cache_reference
        self._max_cached = max(1, int(max_cached_references))
        self._strict = strict_config
        self._references: OrderedDict[str, ReferenceContext] = OrderedDict()

    def run_from_config(self, config_path: str) -> RunSummary:
        """Run the full pipeline from a YAML config file."""
        return self._orchestrator.run(config_path)

    def evaluate(
        self,
        real: pd.DataFrame,
        synth: pd.DataFrame,
        config: dict[str, Any],
        seed: int | None = None,
        *,
        holdout: pd.DataFrame | None = None,
        utility_signal: dict[str, Any] | None = None,
    ) -> RunSummary:
        """Evaluate pre-loaded DataFrames.

        ``seed=None`` uses the config's seed (``evaluation.seed`` →
        ``reproducibility.seed`` → 42); an explicit value overrides it.

        ``holdout`` is an optional table of real rows the generator never
        saw.  It is preprocessed exactly like *real* (same fitted caster,
        same columns) and reaches the metrics as ``context["holdout"]``.

        ``utility_signal`` is the result of :meth:`measure_utility_signal`
        for this *real* table: the utility metrics follow it instead of
        deciding on each call whether the target has signal.
        """
        config_warnings = validate_dataframe_config(config, strict=self._strict)
        reference = None
        if self._cache_reference:
            try:
                reference = self._reference_for(real, config)
            except TypeError:  # unhashable cell values: evaluate without cache
                reference = None
        summary = self._orchestrator.evaluate_dataframes(
            real,
            synth,
            config,
            seed,
            holdout=holdout,
            reference=reference,
            utility_signal=utility_signal,
        )
        if config_warnings:
            summary.artifacts.setdefault("warnings", []).extend(config_warnings)
        if holdout is None and any(r.family == "privacy" for r in summary.results):
            message = (
                "privacy measured without a holdout: the dataset-based privacy metrics "
                "do not apply; pass holdout= (real rows the generator never saw)"
            )
            warnings.warn(message, MissingHoldoutWarning, stacklevel=2)
            summary.artifacts.setdefault("warnings", []).append(message)
        return summary

    def measure_utility_signal(
        self,
        real: pd.DataFrame,
        config: dict[str, Any],
        seed: int | None = None,
        *,
        holdout: pd.DataFrame | None = None,
        n_half_samples: int = UTILITY_SIGNAL_HALF_SAMPLES,
    ) -> dict[str, Any] | None:
        """Whether *real* has signal for the utility metrics, measured once.

        Same rule as each evaluation (TTR gain > 3·√2·σ), with σ from
        *n_half_samples* half-samples instead of the evaluation's five, on
        the same real table, holdout and model budget the evaluations use.
        It also fixes the anchors of the utility scale.  Pass the result
        as ``utility_signal=`` to every :meth:`evaluate` of this table: they
        then train only the synthetic strategies.  None when *config* has no target or nothing could be
        measured (the evaluations then decide on their own).
        """
        if not (config.get("data") or {}).get("target"):
            return None
        probe = copy.deepcopy(config)
        probe["metrics"] = ["utility.ml_efficiency"]
        evaluation = probe.setdefault("evaluation", {})
        evaluation["ml_efficiency_config"] = {
            **(evaluation.get("ml_efficiency_config") or {}),
            "scale_only": True,
            "n_splits": int(n_half_samples),
        }
        reference = None
        if self._cache_reference:
            try:
                reference = self._reference_for(real, probe)
            except TypeError:
                reference = None
        summary = self._orchestrator.evaluate_dataframes(
            real, real, probe, seed, holdout=holdout, reference=reference
        )
        (result,) = summary.results
        if "scale" not in result.details:
            return None
        signal = pooled_utility_signal([result.details])
        if signal is not None:
            signal["source"] = "measure_utility_signal"
        return signal

    def clear_cache(self) -> None:
        """Forget every cached reference."""
        self._references.clear()

    def _reference_for(self, real: pd.DataFrame, config: dict[str, Any]) -> ReferenceContext:
        key = reference_key(real, config)
        ref = self._references.get(key)
        if ref is not None:
            self._references.move_to_end(key)
            return ref
        ref = self._orchestrator.prepare_reference(real, copy.deepcopy(config))
        self._references[key] = ref
        while len(self._references) > self._max_cached:
            self._references.popitem(last=False)
        return ref


def evaluate_from_config(config_path: str) -> RunSummary:
    """Convenience function — run evaluation from a config path."""
    return Evaluator().run_from_config(config_path)
