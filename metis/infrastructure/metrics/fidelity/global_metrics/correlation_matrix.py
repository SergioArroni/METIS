"""
Correlation Matrix Similarity metric.

Compares the correlation structure between real and synthetic datasets
using multiple methods:
- Frobenius norm of difference
- Spectral norm (largest eigenvalue difference)
- RV coefficient (multivariate correlation)
"""

import numpy as np

from ...registry import register
from ..fidelity_base import GlobalFidelityMetric


@register("fidelity.correlation_matrix")
class CorrelationMatrixMetric(GlobalFidelityMetric):
    """
    Measures similarity between correlation matrices of real and synthetic data.

    This metric evaluates how well the synthetic data preserves the
    correlation structure of the original data using:
    1. Frobenius norm: Element-wise difference
    2. Spectral norm: Eigenvalue-based comparison
    3. RV coefficient: Multivariate structural similarity

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> real = pd.DataFrame(np.random.randn(1000, 5), columns=["a", "b", "c", "d", "e"])
        >>> synth = pd.DataFrame(np.random.randn(1000, 5), columns=["a", "b", "c", "d", "e"])
        >>> metric = CorrelationMatrixMetric()
        >>> metric.fit(real, synth)
        >>> result = metric.compute()
        >>> print(f"Correlation Similarity: {result.value:.4f}")
    """

    name: str = "correlation_matrix"

    def __init__(self, method: str = "combined"):
        """
        Initialize the metric.

        Args:
            method: Comparison method ("frobenius", "spectral", "rv", "combined")
        """
        super().__init__()
        self.method = method

    def _compute_global(self) -> tuple[float, float, dict]:
        """
        Compute correlation matrix similarity.

        Returns:
            tuple of (raw_value, normalized_value, details)
        """
        # Get numeric columns
        real_num = self._real_data.select_dtypes(include=[np.number])
        synth_num = self._synth_data.select_dtypes(include=[np.number])
        common_cols = list(set(real_num.columns) & set(synth_num.columns))

        if len(common_cols) < 2:
            raise ValueError("Need at least 2 common numeric columns")

        # Compute correlation matrices
        real_corr = real_num[common_cols].corr().values
        synth_corr = synth_num[common_cols].corr().values

        # Handle NaN values
        real_corr = np.nan_to_num(real_corr, nan=0.0)
        synth_corr = np.nan_to_num(synth_corr, nan=0.0)

        # Compute metrics
        frobenius_score = self._frobenius_similarity(real_corr, synth_corr)
        spectral_score = self._spectral_similarity(real_corr, synth_corr)
        rv_score = self._rv_coefficient(real_corr, synth_corr)

        if self.method == "frobenius":
            score = frobenius_score
        elif self.method == "spectral":
            score = spectral_score
        elif self.method == "rv":
            score = rv_score
        else:  # combined
            score = 0.4 * frobenius_score + 0.3 * spectral_score + 0.3 * rv_score

        details = {
            "frobenius_score": frobenius_score,
            "spectral_score": spectral_score,
            "rv_coefficient": rv_score,
            "n_columns": len(common_cols),
            "matrix_size": f"{len(common_cols)}x{len(common_cols)}",
        }

        return 1.0 - score, score, details

    def _frobenius_similarity(self, real_corr: np.ndarray, synth_corr: np.ndarray) -> float:
        """
        Compute similarity based on Frobenius norm.

        Frobenius norm = sqrt(sum of squared element differences)
        Max possible = sqrt(2 * n^2) for correlations in [-1, 1]
        """
        diff = real_corr - synth_corr
        frobenius_norm = np.linalg.norm(diff, "fro")

        # Normalize: max difference is 2 per element (from -1 to 1)
        n = real_corr.shape[0]
        max_norm = np.sqrt(4 * n * n)  # sqrt(2^2 * n^2)

        similarity = 1.0 - (frobenius_norm / max_norm)
        return float(np.clip(similarity, 0, 1))

    def _spectral_similarity(self, real_corr: np.ndarray, synth_corr: np.ndarray) -> float:
        """
        Compute similarity based on spectral (operator) norm.

        Spectral norm = largest singular value of difference matrix
        """
        diff = real_corr - synth_corr
        spectral_norm = np.linalg.norm(diff, 2)

        # Max spectral norm for correlation difference is ~2
        max_spectral = 2.0

        similarity = 1.0 - (spectral_norm / max_spectral)
        return float(np.clip(similarity, 0, 1))

    def _rv_coefficient(self, real_corr: np.ndarray, synth_corr: np.ndarray) -> float:
        """
        Compute RV coefficient (multivariate correlation coefficient).

        RV = trace(A @ B.T) / sqrt(trace(A @ A.T) * trace(B @ B.T))
        """
        # Center matrices (remove diagonal)
        real_centered = real_corr - np.eye(real_corr.shape[0])
        synth_centered = synth_corr - np.eye(synth_corr.shape[0])

        # Compute traces
        trace_ab = np.trace(real_centered @ synth_centered.T)
        trace_aa = np.trace(real_centered @ real_centered.T)
        trace_bb = np.trace(synth_centered @ synth_centered.T)

        if trace_aa * trace_bb <= 0:
            return 0.0

        rv = trace_ab / np.sqrt(trace_aa * trace_bb)
        return float(np.clip(rv, 0, 1))
