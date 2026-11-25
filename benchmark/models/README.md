# Benchmark of generators

Material of the benchmark of twelve generators on Telco, Cardio, HR and Airbnb (Sections 6.1 and
6.2 of the METIS paper). The scripts are the ones that produced the results of the paper, kept as
they were run; their paths point to the machine where they ran (`/home/ubuntu/...`).

## How it is run

1. **Synthetic tables.** `metis.sota_models.comparison.orchestrator.BenchmarkOrchestrator`
   (`run_benchmark`) splits each real table per seed (42 to 46) into training, control holdout and
   unseen parts, 60/20/20 and stratified by the target (`metis.shared.splits.split_real`), trains
   each generator on the training part and saves its synthetic table, of the size of the unseen part.
   `real_data` returns the unseen part.
2. **Calibration and evaluation.** `evalua_bench_metis.py cal <dataset>` calibrates METIS once per
   real table, with the same split function and calibration seeds that never coincide with the
   seeds of the benchmark. `evalua_bench_metis.py eval <dataset> <out.jsonl> [i/n]` then evaluates
   the saved synthetic tables against their training part, with their holdout, and stores every
   score of the hierarchy and every metric.
3. **Tables of the paper.** `reagrega_bench.py` builds the scores per seed (`benchmark_by_seed.csv`),
   the median (IQR) per dataset and generator, the Friedman tests and the harmonization ablation, and
   `filas_latex.py` the rows of the tables of the paper.

## Files

| File | Content |
|---|---|
| `benchmark_by_seed.csv` | Calibrated (`fidelity`, `utility`, `privacy`, `composite`) and raw (`*_raw`) scores of every seed |
| `metrics_by_generator.csv` | Every metric per dataset and generator, as the median and IQR over the five seeds of its harmonized score before calibration; `NA` marks a metric that does not apply |
| `evalua_bench_metis.py` | Calibration and evaluation of the saved synthetic tables |
| `reagrega_bench.py` | Scores per seed, medians, Friedman and Nemenyi |
| `filas_latex.py` | Rows of the benchmark tables of the paper |

The comparison of METIS with SDMetrics, SynthEval and synthcity on these same synthetic tables is
in [`../validators`](../validators/README.md) (`evalua_bench.py`, `analiza_e5.py`).

## Raw benchmark scores

Scores per dimension before calibration, as median (IQR) over the five seeds (42 to 46).
`benchmark_by_seed.csv` has every seed, with the calibrated (`fidelity`, `utility`, `privacy`,
`composite`) and the raw (`*_raw`) scores. `metrics_by_generator.csv` has every metric per dataset
and generator, as the median and IQR over the five seeds of its harmonized score before calibration;
`NA` marks a metric that does not apply.

| Dataset | Generator | Fidelity raw | Utility raw | Privacy raw |
|---|---|---|---|---|
| Telco | real_data | 0.9891 (0.0077) | 1.0000 (0.0000) | 0.9957 (0.0020) |
| Telco | uniform_noise | 0.4846 (0.0002) | 0.0088 (0.0543) | 1.0000 (0.0004) |
| Telco | bootstrap | 0.9897 (0.0019) | 0.9987 (0.0059) | 0.3601 (0.0005) |
| Telco | smotenc | 0.9413 (0.0148) | 0.9919 (0.0032) | 0.5147 (0.0178) |
| Telco | delete_zero | 0.4516 (0.0051) | 0.3348 (0.0189) | 0.7139 (0.0222) |
| Telco | delete_mean | 0.5154 (0.0012) | 0.3845 (0.0461) | 0.6967 (0.0105) |
| Telco | bn | 0.7880 (0.0244) | 0.9504 (0.0137) | 0.9924 (0.0022) |
| Telco | cart | 0.9885 (0.0161) | 1.0000 (0.0000) | 0.6928 (0.0071) |
| Telco | gaussian_copula | 0.6548 (0.0159) | 0.7474 (0.1903) | 0.9977 (0.0015) |
| Telco | ctgan | 0.6043 (0.0070) | 0.9805 (0.0161) | 0.9986 (0.0028) |
| Telco | tvae | 0.5968 (0.0075) | 0.2137 (0.0076) | 0.9915 (0.0127) |
| Telco | dpctgan | 0.4309 (0.0117) | 0.0541 (0.1113) | 0.9998 (0.0001) |
| Cardio | real_data | 0.9897 (0.0099) | 0.9964 (0.0058) | 0.9983 (0.0006) |
| Cardio | uniform_noise | 0.3264 (0.0016) | 0.0000 (0.0192) | 0.9994 (0.0007) |
| Cardio | bootstrap | 0.9935 (0.0143) | 0.9727 (0.0038) | 0.3940 (0.0026) |
| Cardio | smotenc | 0.7420 (0.0205) | 0.9085 (0.0632) | 0.4424 (0.0023) |
| Cardio | delete_zero | 0.4144 (0.0022) | 0.2127 (0.0005) | 0.6978 (0.0142) |
| Cardio | delete_mean | 0.5552 (0.0013) | 0.2133 (0.0008) | 0.6794 (0.0025) |
| Cardio | bn | 0.9183 (0.0165) | 0.9750 (0.0214) | 0.9986 (0.0032) |
| Cardio | cart | 0.9926 (0.0099) | 0.9902 (0.0172) | 0.9781 (0.0094) |
| Cardio | gaussian_copula | 0.5965 (0.0172) | 0.5699 (0.0372) | 0.9979 (0.0024) |
| Cardio | ctgan | 0.6350 (0.0238) | 0.6559 (0.1118) | 0.9991 (0.0015) |
| Cardio | tvae | 0.6317 (0.0057) | 0.2587 (0.1178) | 0.9958 (0.0087) |
| Cardio | dpctgan | 0.4074 (0.0132) | 0.1122 (0.1289) | 0.9968 (0.0162) |
| HR | real_data | 0.9687 (0.0206) | 1.0000 (0.0000) | 0.9910 (0.0079) |
| HR | uniform_noise | 0.4871 (0.0033) | 0.0000 (0.0000) | 0.9981 (0.0012) |
| HR | bootstrap | 0.9715 (0.0024) | 1.0000 (0.0000) | 0.4069 (0.0031) |
| HR | smotenc | 0.7672 (0.0168) | 1.0000 (0.0000) | 0.4588 (0.0139) |
| HR | delete_zero | 0.4438 (0.0039) | 0.2301 (0.4429) | 0.6994 (0.0005) |
| HR | delete_mean | 0.5360 (0.0052) | 1.0000 (0.0014) | 0.6083 (0.0137) |
| HR | bn | 0.9505 (0.0091) | 1.0000 (0.0000) | 0.9898 (0.0095) |
| HR | cart | 0.9635 (0.0060) | 1.0000 (0.0000) | 0.8891 (0.0648) |
| HR | gaussian_copula | 0.6051 (0.0067) | 0.0000 (0.4424) | 0.9917 (0.0202) |
| HR | ctgan | 0.5841 (0.0331) | 0.0000 (0.0000) | 0.9988 (0.0028) |
| HR | tvae | 0.5277 (0.0080) | 0.8269 (0.1368) | 0.9625 (0.0189) |
| HR | dpctgan | 0.4554 (0.0037) | 0.0000 (0.0000) | 0.9966 (0.0062) |
| Airbnb | real_data | 0.9687 (0.0049) | 1.0000 (0.0035) | 1.0000 (0.0021) |
| Airbnb | uniform_noise | 0.3044 (0.0008) | 0.0000 (0.0000) | 0.7680 (0.0052) |
| Airbnb | bootstrap | 0.9277 (0.0070) | 1.0000 (0.0000) | 0.3577 (0.0018) |
| Airbnb | smotenc | 0.6140 (0.0064) | 0.8551 (0.0506) | 0.4987 (0.0071) |
| Airbnb | delete_zero | 0.3694 (0.0016) | 0.0000 (0.0000) | 0.6799 (0.0049) |
| Airbnb | delete_mean | 0.5034 (0.0010) | 0.0000 (0.0000) | 0.6542 (0.0063) |
| Airbnb | bn | 0.6099 (0.0029) | 0.6438 (0.0450) | 0.9691 (0.0085) |
| Airbnb | cart | 0.7845 (0.0105) | 0.8371 (0.0045) | 0.8562 (0.0266) |
| Airbnb | gaussian_copula | 0.5608 (0.0084) | 0.4167 (0.2170) | 0.9951 (0.0203) |
| Airbnb | ctgan | 0.5967 (0.0253) | 0.4925 (0.0452) | 0.9992 (0.0014) |
| Airbnb | tvae | 0.5916 (0.0195) | 0.5943 (0.1763) | 0.9988 (0.0003) |
| Airbnb | dpctgan | 0.2785 (0.0187) | 0.0000 (0.0000) | 1.0000 (0.0003) |
