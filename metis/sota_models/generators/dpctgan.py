"""DP-CTGAN (Differentially Private Conditional Tabular GAN) generator."""

import warnings

import numpy as np
import pandas as pd

from .base import BaseGenerator
from .gan_utils import DEFAULT_HIGH_CARDINALITY_THRESHOLD, filter_high_cardinality

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

        self._real_data = real_data.copy()
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

        # Mirror the OOM protection already used by CTGAN/TVAE: very
        # high-cardinality categoricals explode the one-hot dimensionality
        # and make Opacus allocate huge per-sample gradient tensors.
        train_data, kept_cat_cols, self._dropped_cols = filter_high_cardinality(
            real_data.copy(),
            self._categorical_columns,
            threshold=self.high_cardinality_threshold,
        )
        continuous_kept = [c for c in self._continuous_columns if c not in self._dropped_cols]

        self._categorical_columns = kept_cat_cols
        self._continuous_columns = continuous_kept

        if self.random_state is not None:
            np.random.seed(self.random_state)
            try:
                import torch

                torch.manual_seed(self.random_state)
            except ImportError:
                pass

        # Build explicit bounds for continuous columns so the DP
        # preprocessor does not need to spend any epsilon on approx_bounds.
        # This keeps the full privacy budget for DP-SGD training.
        from snsynth.transform import MinMaxTransformer

        constraints = {}
        for col in self._continuous_columns:
            constraints[col] = MinMaxTransformer(
                lower=float(train_data[col].min()),
                upper=float(train_data[col].max()),
                negative=True,
            )

        self._model = SNSynthesizer.create(
            "dpctgan",
            epsilon=self.epsilon,
            epochs=self.epochs,
            batch_size=self.batch_size,
            sigma=self.sigma,
            embedding_dim=self.embedding_dim,
            generator_dim=self.generator_dim,
            discriminator_dim=self.discriminator_dim,
            disabled_dp=self.disabled_dp,
            verbose=False,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(
                train_data,
                categorical_columns=self._categorical_columns,
                continuous_columns=self._continuous_columns,
                preprocessor_eps=0.0,
                transformer=constraints,
            )

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

        if self._dropped_cols and self._real_data is not None:
            for col in self._dropped_cols:
                if col in self._real_data.columns:
                    synthetic[col] = (
                        self._real_data[col]
                        .sample(n=n_samples, replace=True, random_state=self.random_state)
                        .values
                    )

        # Ensure column order matches original data
        final_cols = [c for c in self._all_columns if c in synthetic.columns]
        return synthetic[final_cols]
