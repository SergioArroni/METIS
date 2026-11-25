# Benchmark of validators

Code that builds the bank of tables with known defects of the METIS paper (Section 6.3 and
Appendix E) and evaluates it with METIS, SDMetrics, SynthEval and synthcity. These are the scripts
that produced the results of the paper, kept as they were run. Their paths point to the machine
where they ran (`/home/ubuntu/...`) and have to be adapted to run them elsewhere.

## How the bank is built

1. **Split.** Each real table (Telco, Cardio, HR and Adult, up to 8,000 rows) is split once, with
   seed 0 and stratified by the target, into A (a quarter, the training table the synthetic one is
   compared with), H (a quarter, the control holdout) and B (the rest, a reserve).
2. **Clean tables.** Five clean tables per dataset are samples of B of the size of A, stratified
   by the target and without replacement. They are real rows that are not in A, so they play a
   perfect generator.
3. **Defects.** Each defect is applied to each clean table, and each defect has the dimension
   it should lower (fidelity, utility, privacy or validity). A defect with no column to act on in
   a dataset, such as ordinal defects in Telco, is not generated there. The list of defects and
   what each does is in the docstrings of `gen_casos.py`, `gen_casos_het.py` and `gen_casos_e12.py`,
   and in Appendix E of the paper.
4. **Calibration.** METIS is calibrated once per dataset on A, H and as many rows of B as A has
   (written by `gen_casos_het.py`), with parts of the size of A (`calibration.split` in thirds),
   the size of what is evaluated.

## How detection is measured

Every validator is run on the five clean tables and on every defective table. For each published
score and each defect,

    z_k = (mean of the 5 clean scores − score of the defect applied to clean table k) / sd of the 5 clean scores

with a floor of 0.005 for the standard deviation, and the defect counts as detected when the median
of z_k over the five samples is at least 3. The criterion is relative to the noise of the clean
tables of each dataset, so no threshold is carried over between datasets or validators. Each
validator is measured with its overall score (METIS composite, SDMetrics Quality, the mean of all
SynthEval or synthcity metrics), with its score of the dimension the defect should lower when it
publishes one, and with the most favourable of all the aggregate scores it publishes.

## Scripts

| Script | What it does |
|---|---|
| `gen_casos.py` | Split, clean tables and the first defects (shifts, scales, tails, broken relations, labels, copies) |
| `gen_casos_het.py` | Heterogeneity and validity defects; writes B and the table used for calibration |
| `gen_casos_e12.py` | Ordinal distance and dose ladders (copies, labels, shifts) |
| `gen_estudio.py` | Defects for the study of alternative metrics (Appendix F) |
| `gen_privutil.py`, `gen_matices.py` | Privacy and utility bank, and local nuances (Appendix E.2) |
| `a_csv.py` | CSV copies for the synthcity environment |
| `metis_banco_v3.py` | METIS (and METIS without types, `PLANO=1`) on a dataset of the bank |
| `evalua.py` | SDMetrics, SynthEval or synthcity on a dataset of the bank |
| `metricas_nuevas.py` | Candidate metrics of Appendix F |
| `reagrega.py`, `reagrega_banco.py` | Re-aggregation of stored METIS results with a given taxonomy |
| `analiza2.py`, `analiza_sc.py` | Detection tables of the paper |
| `desglose_defecto.py` | z of one defect at every level of the METIS hierarchy, down to each metric |
| `evalua_bench.py`, `e5_synthcity.py`, `analiza_e5.py` | SDMetrics, SynthEval and synthcity on the synthetic tables of the generator benchmark, and their ranks against METIS (Table 7) |
| `factorial_lambda.py` | The 4,096 combinations of aggregation operators (Section 4.2, Table 2) |
| `ablacion_51.py` | Sensitivity to the harmonization families and their scale (Section 4.3) |
| `reorganiza.py`, `analiza_metricas.py` | Grouping of the metrics and usefulness of each one (Appendix D) |
| `analiza_aviso.py`, `analiza_dilucion.py` | Per-metric warning and dilution of single-metric defects (Section 4.3) |
| `analiza_matices.py`, `analiza_pu.py`, `analiza_estudio.py` | Local nuances, privacy and utility bank, and alternative metrics (Appendices E.2, D and F) |
| `ejecuta_todo.sh` | The whole run of both benchmarks: generation, calibrations and a queue of single-thread jobs |

`ejecuta_todo.sh` documents the order of the run. It also calls a few one-off helpers of that
machine that are not kept here (`strat_parches.py`, which copied the scripts with their paths,
`efecto_etiquetas.py` and `fuerza_trio.py`, two side checks), and it expects
`../models/evalua_bench_metis.py` next to the other scripts.

## Studies of METIS on the bank

Two studies reuse the cases of the bank to fix choices of METIS itself.

- `estudio_operadores/`: the aggregation operators of each level of the hierarchy, with ordinal
  defects and injected leaks (Section 4.3). `resultado_2026-11-18.txt` has the result with the
  current METIS.
- `estudio_tipos/`: defects that only break what the type of a column declares, evaluated with the
  full schema and with simpler ones (Section 5.2), and the counts of the bank by dimension and
  without the validity cases. `tipos_resultado.txt` and `sin_validez_recuento.txt` have the results.
- `escalera_tipos.py`: detections of the bank for each level of the type schema.

SDMetrics 0.32 and SynthEval 1.7.3 run in one environment (numpy 2) and synthcity 0.2.12 in another
(numpy 1), which reads the CSV copies.
