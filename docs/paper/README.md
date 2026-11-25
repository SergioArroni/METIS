# METIS paper: material in the repository

Material referenced from the METIS paper that is not in its appendices.

- [`metric_definitions.pdf`](metric_definitions.pdf) (Spanish: [`metric_definitions_es.pdf`](metric_definitions_es.pdf)):
  the definitions that the appendices of the paper do not give, with what METIS computes in fidelity
  differently from the usual definition (except the bias corrections of Appendix A), the syntactic
  privacy models, the empirical similarity metrics and the mechanism-based privacy.

- The code of the two benchmarks of the paper is in [`benchmark/`](../../benchmark/README.md):
  [`models/`](../../benchmark/models/README.md) has the benchmark of twelve generators (Sections 6.1
  and 6.2), with its scores per seed and per metric, and [`validators/`](../../benchmark/validators/README.md)
  the bank of known defects and the comparison of METIS with SDMetrics, SynthEval and synthcity
  (Section 6.3 and Appendix E).

## Aggregation operators

Let x = (x_1, ..., x_n) in [0,1]^n be the harmonized scores aggregated at a node of the hierarchy.
METIS uses `mean` at L1-L2 and `ssd` with lambda = 3 at L3-L4.

- `mean`: arithmetic mean, (1/n) * sum_i x_i.
- `median`: sample median of the sorted vector; the mean of the two central values when n is even.
- `trimmed_mean_10`: symmetric mean trimmed by 10% per tail; with k = floor(0.1 n) it averages
  x_(k+1), ..., x_(n-k), and equals the mean when k = 0. Available as an option.
- `ssd`: second-order stochastic dominance aggregator, the exponential certainty equivalent
  A(x) = -(1/lambda) * log((1/n) * sum_i exp(-lambda * x_i)), with risk aversion lambda > 0.
  It equals the mean as lambda -> 0 and approaches the minimum as lambda grows. The
  implementation bounds the result to [min_i x_i, mean(x)] for numerical stability.
