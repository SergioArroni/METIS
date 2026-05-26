# Calibration

## The Problem

Raw metric values are not comparable:

- KS statistic of 0.15 — is that good or bad?
- Wasserstein distance of 23.4 — relative to what?
- DCR of 0.8 — compared to the best achievable?

Different metrics have different scales, ranges, and interpretations.

## The Solution: Empirical Bounds

METIS estimates **data-specific bounds** for each metric:

| Bound | How it's computed | Interpretation |
|-------|------------------|----------------|
| **Upper** (best) | Real vs. Real (split-half) | Best achievable quality |
| **Lower** (worst) | Real vs. Uniform Noise | Worst expected quality |

## Normalization Formula

For distance metrics (lower raw = better):

$$
\text{score} = 1 - \frac{\text{raw} - \text{lower}}{\text{upper} - \text{lower}}
$$

For similarity metrics (higher raw = better):

$$
\text{score} = \frac{\text{raw} - \text{lower}}{\text{upper} - \text{lower}}
$$

Result is clipped to [0, 1].

## Why Split-Half?

The upper bound uses the real data split in half:

1. Split real data into two halves (stratified by target if available)
2. Compute metric between the two halves
3. Repeat N times with different random splits
4. Take the median as the upper bound

This represents the **best a synthetic dataset could achieve** — matching the real distribution as closely as two halves of the real data match each other.

## Caching

Calibration is expensive, so results are cached:

- Cache key = `{data_fingerprint}_{config_fingerprint}_{params_hash}`
- Automatically invalidated when data or config changes
- Stored as JSON in `metis/calibrate/cache/`

## Aggregator Tuning

When `tune_aggregators: true`, METIS uses Optuna to optimize aggregation weights:

- Maximizes separation between upper-bound scores and lower-bound scores
- Ensures the aggregation function discriminates well between good and bad synthetic data
- Weights are cached alongside bounds
