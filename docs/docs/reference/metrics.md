# All Metrics (48)

METIS provides 48 metrics across three dimensions.

## Fidelity — Global (4)

| ID | Description | Type |
|----|-------------|------|
| `fidelity.correlation_matrix` | Frobenius distance between correlation matrices | Global |
| `fidelity.mmd` | Maximum Mean Discrepancy (kernel-based) | Global |
| `fidelity.energy_distance` | Energy distance between distributions | Global |
| `fidelity.outliers_coverage` | Coverage of real-data outlier regions | Global |

## Fidelity — Marginal: Tails (6)

| ID | Description | Per-column |
|----|-------------|-----------|
| `fidelity.ks` | Kolmogorov-Smirnov two-sample test | Numeric |
| `fidelity.wasserstein` | Wasserstein (earth-mover) distance | Numeric |
| `fidelity.anderson_darling` | Anderson-Darling test statistic | Numeric |
| `fidelity.hellinger` | Hellinger distance | Numeric |
| `fidelity.kde_ise` | Integrated squared error of KDE | Numeric |
| `fidelity.delta_exceedance` | Exceedance probability delta | Numeric |

## Fidelity — Marginal: Location & Scale (5)

| ID | Description | Per-column |
|----|-------------|-----------|
| `fidelity.delta_mean` | Absolute difference in means | Numeric |
| `fidelity.delta_median` | Absolute difference in medians | Numeric |
| `fidelity.delta_iqr` | Absolute difference in IQR | Numeric |
| `fidelity.delta_mad` | Absolute difference in MAD | Numeric |
| `fidelity.cohens_d` | Cohen's d effect size | Numeric |

## Fidelity — Marginal: Coverage (6)

| ID | Description | Per-column |
|----|-------------|-----------|
| `fidelity.tvd` | Total Variation Distance | Categorical |
| `fidelity.js` | Jensen-Shannon divergence | Categorical |
| `fidelity.kl` | Kullback-Leibler divergence | Categorical |
| `fidelity.psi` | Population Stability Index | Categorical |
| `fidelity.entropy_delta` | Entropy difference | Categorical |
| `fidelity.gini_delta` | Gini impurity difference | Categorical |

## Fidelity — Conditional: num↔num (4)

| ID | Description | Per-pair |
|----|-------------|---------|
| `fidelity.pearson` | Pearson correlation delta | Numeric×Numeric |
| `fidelity.spearman` | Spearman rank correlation delta | Numeric×Numeric |
| `fidelity.mi` | Mutual information delta | Numeric×Numeric |
| `fidelity.dcor` | Distance correlation delta | Numeric×Numeric |

## Fidelity — Conditional: num↔cat (3)

| ID | Description | Per-pair |
|----|-------------|---------|
| `fidelity.eta_squared` | Eta-squared (ANOVA effect size) delta | Numeric×Categorical |
| `fidelity.point_biserial` | Point-biserial correlation delta | Numeric×Categorical |
| `fidelity.kruskal_epsilon` | Kruskal-Wallis epsilon² delta | Numeric×Categorical |

## Fidelity — Conditional: cat↔cat (3)

| ID | Description | Per-pair |
|----|-------------|---------|
| `fidelity.cramers_v` | Cramér's V delta | Categorical×Categorical |
| `fidelity.theils_u` | Theil's U (uncertainty coefficient) delta | Categorical×Categorical |
| `fidelity.chi2_stat` | Chi-squared statistic delta | Categorical×Categorical |

## Utility (5)

| ID | Description |
|----|-------------|
| `utility.tstr` | Train-on-Synthetic, Test-on-Real |
| `utility.trts` | Train-on-Real, Test-on-Synthetic |
| `utility.tts` | Train-on-Test-Synthetic |
| `utility.ttrs` | Train-on-Test-Real+Synthetic |
| `utility.ml_efficiency` | Aggregated ML efficiency score |

## Privacy (9)

| ID | Description |
|----|-------------|
| `privacy.dcr` | Distance to Closest Record |
| `privacy.nnaa` | Nearest Neighbor Adversarial Accuracy |
| `privacy.mia` | Membership Inference Attack |
| `privacy.inference_attack` | Attribute inference attack |
| `privacy.record_linkage` | Record linkage attack |
| `privacy.k_anonymity` | k-Anonymity preservation |
| `privacy.l_diversity` | l-Diversity preservation |
| `privacy.t_closeness` | t-Closeness preservation |
| `privacy.differential_privacy` | Differential privacy estimation |
