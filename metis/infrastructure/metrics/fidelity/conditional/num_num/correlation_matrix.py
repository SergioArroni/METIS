"""
Correlation Matrix Similarity metric.

Compares the correlation structure between real and synthetic datasets
using multiple methods:
- Frobenius norm of difference
- Spectral norm (largest eigenvalue difference)
- RV coefficient (multivariate correlation)
"""

import numpy as np
from scipy.stats import chi2

from ....registry import register
from ...fidelity_base import GlobalFidelityMetric


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

    def __init__(self, method: str = "combined", rv_alpha: float = 0.05):
        """
        Initialize the metric.

        Args:
            method: Comparison method ("frobenius", "spectral", "rv", "combined")
            rv_alpha: Level of Bartlett's sphericity test that decides whether a
                table has correlation structure (RV undefined without it).
        """
        super().__init__()
        self.method = method
        self.rv_alpha = rv_alpha

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
        rv_score, rv_details = self._rv_coefficient(
            real_corr,
            synth_corr,
            self._pairwise_counts(real_num[common_cols]),
            self._pairwise_counts(synth_num[common_cols]),
            alpha=self.rv_alpha,
            return_details=True,
        )

        if self.method == "frobenius":
            score = frobenius_score
        elif self.method == "spectral":
            score = spectral_score
        elif self.method == "rv":
            score = rv_score
        elif np.isnan(rv_score):
            # RV undefined (no detectable structure in the real table): the
            # question "is the synthetic structure the same?" is answered by
            # the two distances alone, with their weights renormalized.
            score = (0.4 * frobenius_score + 0.3 * spectral_score) / 0.7
        else:  # combined
            score = 0.4 * frobenius_score + 0.3 * spectral_score + 0.3 * rv_score

        details = {
            "frobenius_score": frobenius_score,
            "spectral_score": spectral_score,
            "rv_coefficient": rv_score,
            "rv_method": "rv2_adjusted_bartlett_gated",
            **rv_details,
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

        Known limit (audit, 2026-10-19; not corrected): the operator norm of a
        sample correlation matrix's error grows with the number of columns and
        shrinks with n, so two samples of the same real table score 0.67 with
        100 rows and 0.95 with 4000.  The per-metric calibration absorbs it when
        it is measured at the evaluation's size (its upper anchor is real
        against real at that size); see docs/auditoria §15.
        """
        diff = real_corr - synth_corr
        spectral_norm = np.linalg.norm(diff, 2)

        # Max spectral norm for correlation difference is ~2
        max_spectral = 2.0

        similarity = 1.0 - (spectral_norm / max_spectral)
        return float(np.clip(similarity, 0, 1))

    @staticmethod
    def _pairwise_counts(df) -> np.ndarray:
        """Number of rows where each pair of columns is observed (as ``df.corr``)."""
        observed = df.notna().to_numpy(dtype=float)
        return observed.T @ observed

    @staticmethod
    def _sphericity_pvalue(corr: np.ndarray, n: float) -> float:
        """p-value of Bartlett's (1951) sphericity test (H0: correlation matrix = I).

        ``-(n - 1 - (2k + 5) / 6) ln|R| ~ chi^2_{k(k-1)/2}`` under H0 for
        multivariate normal data. A non-positive determinant (singular or, with
        pairwise deletion, not positive definite) is treated as structure
        (p = 0). Returns NaN when ``n`` is too small for the approximation.
        The diagonal is set to 1 (a constant column has NaN correlations,
        mapped to 0 upstream, and must not read as a singular matrix).
        """
        k = corr.shape[0]
        corr = corr.copy()
        np.fill_diagonal(corr, 1.0)
        sign, logdet = np.linalg.slogdet(corr)
        if sign <= 0:
            return 0.0
        if np.isinf(n):  # infinite sample: any correlation is structure
            return 0.0 if logdet < 0 else 1.0
        factor = n - 1 - (2 * k + 5) / 6
        if factor <= 0:
            return float("nan")
        return float(chi2.sf(-factor * logdet, k * (k - 1) / 2))

    @staticmethod
    def _rv_coefficient(
        real_corr: np.ndarray,
        synth_corr: np.ndarray,
        n_real: np.ndarray | float | None = None,
        n_synth: np.ndarray | float | None = None,
        *,
        alpha: float = 0.05,
        return_details: bool = False,
    ) -> float | tuple[float, dict]:
        """
        RV-type similarity of the correlation structure of two tables.

        It answers "do both tables have the same *pattern* of correlations?"
        (a cosine between the off-diagonal correlations: invariant to an overall
        weakening of every correlation, which the Frobenius and spectral scores
        do measure). Built only from published pieces:

        * **Off-diagonal entries only**, as in the modified RV / RV2 of Smilde
          et al. (2009): the unit diagonal of a correlation matrix carries no
          information and pulls the plain RV towards 1.
        * **Adjusted squared correlations in the norms**, as in the adjusted RV
          of Mayer, Lorent & Horgan (2011): ``r2_adj = 1 - (n-1)/(n-2) (1 - r^2)``
          removes the upward bias of ``r^2`` (``E[r^2] = 1/(n-1)`` when rho = 0).
          The cross term ``sum a_ij b_ij`` needs no correction: real and synthetic
          are independent samples, so it is unbiased for ``sum rho^a rho^b``.
        * **The cosine is only defined when there is structure.** If the real
          correlation matrix is not distinguishable from the identity by
          Bartlett's (1951) sphericity test at ``alpha`` (the conventional 0.05),
          the pattern is undefined (0/0) and NaN is returned; the combined score
          then uses Frobenius and spectral only. If the real table has structure
          and the synthetic one does not (same test), the pattern is lost: 0.

        This replaces an earlier, unpublished variant that added a noise-level
        ridge to numerator and norms so that "no structure vs no structure"
        scored ~1 (with a large variance, 0.87 +/- 0.16).

        References:
            Smilde, A. K., Kiers, H. A. L., Bijlsma, S., Rubingh, C. M. & van Erk,
            M. J. (2009). Matrix correlations for high-dimensional data: the
            modified RV-coefficient. Bioinformatics, 25(3), 401-405.
            Mayer, C.-D., Lorent, J. & Horgan, G. W. (2011). Exploratory analysis
            of multiple omics datasets using the adjusted RV coefficient.
            Statistical Applications in Genetics and Molecular Biology, 10(1), 14.
            Bartlett, M. S. (1951). The effect of standardization on a chi-square
            approximation in factor analysis. Biometrika, 38(3/4), 337-344.
        """
        k = real_corr.shape[0]
        iu = np.triu_indices(k, 1)
        a = real_corr[iu]
        b = synth_corr[iu]

        def _counts(n: np.ndarray | float | None) -> np.ndarray:
            if n is None:
                return np.full(a.size, np.inf)
            arr = np.asarray(n, dtype=float)
            return arr[iu] if arr.ndim == 2 else np.full(a.size, float(arr))

        n_a, n_b = _counts(n_real), _counts(n_synth)

        def _done(value: float, **info) -> float | tuple[float, dict]:
            return (value, info) if return_details else value

        if a.size == 0:
            return _done(float("nan"), rv_defined=False, rv_reason="fewer than 2 columns")

        # Smallest pairwise count: the conservative n for the sphericity test.
        p_real = CorrelationMatrixMetric._sphericity_pvalue(real_corr, float(n_a.min()))
        p_synth = CorrelationMatrixMetric._sphericity_pvalue(synth_corr, float(n_b.min()))
        info = {"sphericity_p_real": p_real, "sphericity_p_synth": p_synth}
        if np.isnan(p_real) or p_real >= alpha:
            return _done(
                float("nan"),
                rv_defined=False,
                rv_reason="no detectable correlation structure in the real table",
                **info,
            )
        if np.isnan(p_synth) or p_synth >= alpha:
            return _done(0.0, rv_defined=True, rv_reason="no structure in synthetic", **info)

        def _adj_energy(x: np.ndarray, n: np.ndarray) -> float:
            with np.errstate(divide="ignore", invalid="ignore"):
                factor = np.where(np.isfinite(n), (n - 1) / np.maximum(n - 2, 1e-12), 1.0)
            return float(np.sum(1 - factor * (1 - x * x)))

        s_aa = _adj_energy(a, n_a)
        s_bb = _adj_energy(b, n_b)
        if s_aa <= 0 or s_bb <= 0:
            return _done(0.0, rv_defined=True, rv_reason="non-positive adjusted energy", **info)
        rv = float(np.sum(a * b)) / np.sqrt(s_aa * s_bb)
        return _done(float(np.clip(rv, 0, 1)), rv_defined=True, **info)
