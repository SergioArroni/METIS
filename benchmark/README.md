# Benchmarks of the METIS paper

- [`models/`](models/README.md): the benchmark of twelve generators on Telco, Cardio, HR and Airbnb
  (Sections 6.1 and 6.2), with its scores per seed and per metric and the code that evaluates it.
  The generators themselves are in the package, `metis.sota_models`.
- [`validators/`](validators/README.md): the bank of known defects and the comparison of METIS with
  SDMetrics, SynthEval and synthcity (Section 6.3 and Appendix E), with the code that builds the
  bank and evaluates it, and the two studies of METIS that reuse the bank (Sections 4.3 and 5.2).

The scripts are kept as they were run, and their paths point to the machine where they ran.
The datasets are not redistributed here (see `data/`).
