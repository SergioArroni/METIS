# Normalization

## Overview

METIS normalizes all raw metric values to **[0, 1]** where **1 = best quality**. This enables:

- Comparison across metrics with different scales
- Aggregation into composite scores
- Cross-dataset comparability

## Normalization Types

Each metric has a normalization type that determines how the raw value is transformed:

| Type | Raw interpretation | Formula |
|------|-------------------|---------|
| **Distance** | Lower = better | `1 - clip((raw - lower) / (upper - lower))` |
| **Similarity** | Higher = better | `clip((raw - lower) / (upper - lower))` |
| **Bounded [0,1]** | Already in [0,1] | `clip(raw)` or calibrated |

## Metric Normalization Map

All 48 metrics are pre-classified:

### Distance metrics (lower raw = better synthetic quality)

- KS, Wasserstein, Anderson-Darling, Hellinger, KDE-ISE
- Delta Mean, Delta Median, Delta IQR, Delta MAD, Cohen's d
- TVD, JS, KL, PSI, Entropy Delta, Gini Delta
- Pearson delta, Spearman delta, MI delta, dCor delta
- Eta², Point-Biserial, Kruskal ε²
- Cramér's V delta, Theil's U delta, χ² statistic delta
- Correlation Matrix, MMD, Energy Distance
- DCR (inverted), NNAA, MIA, Record Linkage, Inference Attack

### Similarity metrics (higher raw = better)

- Outliers Coverage
- TSTR, TRTS, TTS, TTRS, ML Efficiency
- k-Anonymity, l-Diversity, t-Closeness
- Differential Privacy, Delta Exceedance

## Stochastic Dominance Aggregation

After normalization, scores are aggregated using **First-Order Stochastic Dominance (FSD)**:

1. Per-column scores are computed for each metric
2. FSD determines if one distribution of scores dominates another
3. Family scores (fidelity, utility, privacy) are computed
4. A composite score combines all three dimensions

This is more robust than simple averaging because it respects the ordinal structure of scores.
