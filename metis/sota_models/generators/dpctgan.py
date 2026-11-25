"""DP-CTGAN (Differentially Private Conditional Tabular GAN) generator."""

import warnings

import numpy as np
import pandas as pd

from .base import BaseGenerator
from .gan_utils import DEFAULT_HIGH_CARDINALITY_THRESHOLD, filter_high_cardinality

# Value of a high-cardinality column whose domain is not declared public.
SUPPRESSED = "SUPPRESSED"

# ── Optional dependency check ────────────────────────────────────────────────
try:
    from snsynth import Synthesizer as SNSynthesizer

    SMARTNOISE_AVAILABLE = True
except ImportError:
    SMARTNOISE_AVAILABLE = False


class DPCTGANGenerator(BaseGenerator):
    """
    DP-CTGAN — Differentially Private Conditional Tabular GAN.

    Wraps the SmartNoise Synthesizers library (``smartnoise-synth``)
    implementation of DPCTGAN, which applies differentially-private
    stochastic gradient descent (DP-SGD) to a CTGAN architecture.

    The key parameter is ``epsilon`` (ε), the total privacy budget.
    Lower ε ⇒ stronger privacy guarantees but lower data quality.

    Privacy accounting (``self.privacy_accounting`` after ``fit``) — the whole
    pipeline spends at most ``epsilon``:

    * Continuous bounds: from ``bounds`` (public, declared in the config) at no
      cost; for columns without declared bounds, smartnoise infers them with
      ``preprocessor_eps`` (a share of the budget it subtracts from training).
      The previous version read the *exact* min/max of the real table with
      ``preprocessor_eps=0``: not private, and outside the declared ε.
    * High-cardinality categorical columns (dropped from the GAN to avoid OOM)
      are sampled from a Laplace-noised histogram (a share of ε) over a domain
      declared public in ``public_domains``; without a declared domain the
      column is filled with the constant ``SUPPRESSED`` (the labels themselves,
      e.g. host names, are private).  The previous version bootstrapped the
      *real* values (in Airbnb, real ``name``/``host_name``/``license``) with
      no DP at all.
    * The rest is DP-SGD on the discriminator.

    Robustness: rows with missing values are dropped before training (a NaN in
    a batch made the sigmoid output NaN and ``BCELoss`` fail — the "divergence"
    seen in telco on 2 of 3 seeds, whose ``TotalCharges`` has blank cells), and
    ``batch_size`` is capped to an even number not larger than the table.

    Requires: smartnoise-synth (``pip install smartnoise-synth``)
    """

    def __init__(
        self,
        name: str = "DP-CTGAN",
        epsilon: float = 1.0,
        epochs: int = 300,
        batch_size: int = 500,
        sigma: float = 1.0,
        embedding_dim: int = 128,
        generator_dim: tuple[int, int] = (256, 256),
        discriminator_dim: tuple[int, int] = (256, 256),
        high_cardinality_threshold: int = DEFAULT_HIGH_CARDINALITY_THRESHOLD,
        disabled_dp: bool = False,
        random_state: int | None = None,
        bounds: dict[str, tuple[float, float]] | None = None,
        preprocessor_eps_share: float = 0.1,
        histogram_eps_share: float = 0.1,
        public_domains: dict[str, list] | None = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.epsilon = epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.sigma = sigma
        self.embedding_dim = embedding_dim
        self.generator_dim = generator_dim
        self.discriminator_dim = discriminator_dim
        self.high_cardinality_threshold = high_cardinality_threshold
        self.disabled_dp = disabled_dp
        self.random_state = random_state
        self.bounds = dict(bounds or {})
        self.preprocessor_eps_share = preprocessor_eps_share
        self.histogram_eps_share = histogram_eps_share
        self.public_domains = dict(public_domains or {})
        self.privacy_accounting: dict = {}
        self._dp_histograms: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        # Fitted state
        self._model = None
        self._real_data = None
        self._all_columns: list[str] = []
        self._categorical_columns: list[str] = []
        self._continuous_columns: list[str] = []
        self._dropped_cols: list[str] = []

    def fit(
        self,
        real_data: pd.DataFrame,
        categorical_columns: list[str] | None = None,
        ordinal_columns: dict[str, list] | None = None,
        continuous_columns: list[str] | None = None,
    ) -> None:
        """Fit DP-CTGAN to real data with differential privacy guarantees."""
        if not SMARTNOISE_AVAILABLE:
            raise ImportError(
                "smartnoise-synth is required for DPCTGANGenerator. "
                "Install with: pip install smartnoise-synth"
            )

        # The real table is not kept: nothing after fit may read real values.
        self._all_columns = real_data.columns.tolist()
        self._categorical_columns = list(categorical_columns or [])
        ordinal_columns = ordinal_columns or {}
        self._continuous_columns = list(continuous_columns or [])

        # Treat ordinal columns as categorical for smartnoise
        for col in ordinal_columns:
            if col not in self._categorical_columns:
                self._categorical_columns.append(col)

        # Infer unspecified columns
        specified = set(self._categorical_columns) | set(self._continuous_columns)
        for col in self._all_columns:
            if col not in specified:
                if pd.api.types.is_numeric_dtype(real_data[col]):
                    self._continuous_columns.append(col)
                else:
                    self._categorical_columns.append(col)

        # Numeric columns as numbers (blank strings become NaN) and complete
        # rows only: a NaN reaching the discriminator breaks training.
        data = real_data.copy()
        for col in self._continuous_columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        data = data.dropna().reset_index(drop=True)
        if len(data) < 4:
            raise ValueError("DP-CTGAN needs at least 4 complete rows")

        # Mirror the OOM protection already used by CTGAN/TVAE: very
        # high-cardinality categoricals explode the one-hot dimensionality
        # and make Opacus allocate huge per-sample gradient tensors.
        train_data, kept_cat_cols, self._dropped_cols = filter_high_cardinality(
            data,
            self._categorical_columns,
            threshold=self.high_cardinality_threshold,
        )
        continuous_kept = [c for c in self._continuous_columns if c not in self._dropped_cols]
        self._categorical_columns = kept_cat_cols
        self._continuous_columns = continuous_kept

        rng = np.random.default_rng(self.random_state)
        if self.random_state is not None:
            np.random.seed(self.random_state)
            try:
                import torch

                torch.manual_seed(self.random_state)
            except ImportError:
                pass

        # Budget: histograms of the dropped columns, then (inside smartnoise)
        # bound inference, then DP-SGD.
        with_domain = [c for c in self._dropped_cols if c in self.public_domains]
        eps_hist = self.histogram_eps_share * self.epsilon if with_domain else 0.0
        self._dp_histograms = {}
        for col in self._dropped_cols:
            if col not in self.public_domains:
                self._dp_histograms[col] = (np.array([SUPPRESSED]), np.array([1.0]))
                continue
            values = data[col].astype(str)
            domain = np.array([str(v) for v in self.public_domains[col]])
            counts = values.value_counts().reindex(domain).fillna(0).to_numpy(float)
            # Laplace mechanism, sensitivity 1 per histogram, the share split over columns.
            noisy = counts + rng.laplace(0.0, len(with_domain) / eps_hist, len(domain))
            noisy = np.clip(noisy, 0.0, None)
            if noisy.sum() <= 0:
                noisy = np.ones(len(domain))
            self._dp_histograms[col] = (domain, noisy / noisy.sum())
        eps_model = self.epsilon - eps_hist

        from snsynth.transform import MinMaxTransformer

        constraints = {}
        for col in self._continuous_columns:
            if col in self.bounds:
                lower, upper = self.bounds[col]
                constraints[col] = MinMaxTransformer(
                    lower=float(lower), upper=float(upper), negative=True
                )
        needs_inference = any(c not in constraints for c in self._continuous_columns)
        preprocessor_eps = self.preprocessor_eps_share * eps_model if needs_inference else 0.0

        # Opacus needs an even batch no larger than the table.
        batch_size = min(self.batch_size, len(train_data))
        batch_size -= batch_size % 2
        self.privacy_accounting = {
            "epsilon_total": self.epsilon,
            "epsilon_histograms": eps_hist,
            "epsilon_bounds_inference": preprocessor_eps,
            "epsilon_dp_sgd": eps_model - preprocessor_eps,
            "public_bounds": sorted(constraints),
            "inferred_bounds": sorted(c for c in self._continuous_columns if c not in constraints),
            "dp_histogram_columns": with_domain,
            "suppressed_columns": [c for c in self._dropped_cols if c not in with_domain],
            "batch_size": batch_size,
            "rows_used": len(train_data),
        }

        self._model = SNSynthesizer.create(
            "dpctgan",
            epsilon=eps_model,
            epochs=self.epochs,
            batch_size=batch_size,
            sigma=self.sigma,
            embedding_dim=self.embedding_dim,
            generator_dim=self.generator_dim,
            discriminator_dim=self.discriminator_dim,
            disabled_dp=self.disabled_dp,
            verbose=False,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self._model.fit(
                    train_data,
                    categorical_columns=self._categorical_columns,
                    continuous_columns=self._continuous_columns,
                    preprocessor_eps=preprocessor_eps,
                    transformer=constraints,
                )
            except ValueError as exc:
                if "bounds" not in str(exc):
                    raise
                raise ValueError(
                    f"DP-CTGAN could not infer private bounds for {self.privacy_accounting['inferred_bounds']} "
                    f"with ε={preprocessor_eps:.3g} on {len(train_data)} rows; declare public "
                    "bounds in the generator params (bounds: {column: [low, high]})"
                ) from exc

        self._is_fitted = True

    def generate(self, n_samples: int) -> pd.DataFrame:
        """Generate synthetic data with DP guarantees."""
        if not self._is_fitted or self._model is None:
            raise RuntimeError(f"{self.name} must be fitted before generating data")

        if self.random_state is not None:
            np.random.seed(self.random_state)
            try:
                import torch

                torch.manual_seed(self.random_state)
            except ImportError:
                pass

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = self._model.sample(n_samples)

        # smartnoise returns a numpy array or DataFrame depending on version
        if isinstance(raw, np.ndarray):
            synthetic = pd.DataFrame(
                raw,
                columns=[c for c in self._all_columns if c not in self._dropped_cols],
            )
        else:
            synthetic = pd.DataFrame(raw)
            if list(synthetic.columns) != self._all_columns:
                synthetic.columns = [c for c in self._all_columns if c not in self._dropped_cols]

        rng = np.random.default_rng(self.random_state)
        for col, (domain, probs) in self._dp_histograms.items():
            # Never real values row by row: a draw from the DP histogram.
            synthetic[col] = rng.choice(domain, size=n_samples, p=probs)

        # Ensure column order matches original data
        final_cols = [c for c in self._all_columns if c in synthetic.columns]
        return synthetic[final_cols]
