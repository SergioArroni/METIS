"""Evalúa las sintéticas del benchmark v2 de METIS con SDMetrics o SynthEval.

Uso: evalua_bench.py <sdmetrics|syntheval> <dataset> <salida.jsonl> [max_sinteticas]

Mismas particiones que METIS (~/metis-v2b-runs/partes/<ds>/seed<s>_{train,holdout}.csv) y las
mismas sintéticas (las rutas de full/<ds>/<gen>/benchmark_results.json).  El entrenamiento y el
holdout van completos: submuestrearlos ocultaría las filas de origen de las copias.  Si se da
max_sinteticas, la sintética se submuestrea a ese número de filas (semilla de la ejecución), que
no quita filas copiadas, solo las diluye por igual.

Variables de entorno:
  SHARD=i/n      solo las ejecuciones i, i+n, i+2n... (para repartir un dataset entre núcleos)
  SUB_REAL=filas entrenamiento y holdout submuestreados a ese tamaño (semilla de la ejecución).
                 Solo para SynthEval en cardio: con las 42 000 filas completas pasa de 28 GB y el
                 sistema lo mata.  Es una desviación del protocolo y se marca en cada registro.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
VAL, DS, OUT = sys.argv[1], sys.argv[2], Path(sys.argv[3])
MAX_SYN = int(sys.argv[4]) if len(sys.argv) > 4 else None
SHARD = tuple(int(v) for v in os.environ.get("SHARD", "0/1").split("/"))
SUB_REAL = int(os.environ["SUB_REAL"]) if os.environ.get("SUB_REAL") else None
BASE = Path("/home/ubuntu/metis-v2b-runs")
man = json.load(open(BASE / "partes" / "manifest.json"))[DS]
target, task = man["target"], man["task_type"]


def typ(s):
    return s if isinstance(s, str) else s.get("type")


schema = {c: s for c, s in man["schema"].items() if typ(s) != "id"}
cat_cols = [c for c, s in schema.items() if typ(s) in ("nominal", "ordinal")]


def clean(df):
    out = df[[c for c in schema if c in df.columns]].copy()
    for c in out.columns:
        if c in cat_cols:
            out[c] = out[c].astype(str)
        elif typ(schema[c]) == "continuous" and isinstance(schema[c], dict) and schema[c].get("datetime"):
            out[c] = pd.to_datetime(out[c], errors="coerce").astype("int64").where(out[c].notna()) / 1e9
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    return out


runs = []
for f in sorted(glob.glob(str(BASE / "full" / DS / "*" / "benchmark_results.json"))):
    if "/_calibracion/" in f:
        continue
    for gen, seeds in json.load(open(f)).items():
        for seed, p in seeds.items():
            if p.get("synthetic_path"):
                runs.append((gen, int(seed), p["synthetic_path"]))
runs = runs[SHARD[0]::SHARD[1]]

done = set()
if OUT.exists():
    for line in OUT.read_text().splitlines():
        r = json.loads(line)
        done.add((r["generador"], r["semilla"]))

if VAL == "sdmetrics":
    from sdmetrics.reports.single_table import QualityReport
    from sdmetrics.single_table import DCROverfittingProtection

    md = {"columns": {c: {"sdtype": "categorical" if c in cat_cols else "numerical"} for c in schema}}

    def score(train, hold, syn, seed):
        rep = QualityReport()
        rep.generate(train, syn, md, verbose=False)
        dcr = DCROverfittingProtection.compute(train, syn, hold, {"tables": {"t": md}}, "t",
                                               num_rows_subsample=min(len(syn), 2000), num_iterations=3)
        return {"quality": rep.get_score(), "dcr_overfitting": dcr}

elif VAL == "syntheval":
    from syntheval import SynthEval

    def score(train, hold, syn, seed):
        np.random.seed(seed)
        se = SynthEval(train, holdout_dataframe=hold, cat_cols=cat_cols, verbose=False, enable_plots=False,
                       console="off", missing_directive="drop", show_warnings=False)
        preset = "full_eval" if task == "classification" else "fast_eval"
        res = se.evaluate(syn, analysis_target=target, presets_file=preset)
        res = res[np.isfinite(pd.to_numeric(res["n_val"], errors="coerce"))]
        out = {f"m:{r.metric}": float(r.n_val) for r in res.itertuples()}
        for dim, g in res.groupby("dim"):
            out[dim] = float(g["n_val"].astype(float).mean())
        out["all"] = float(res["n_val"].astype(float).mean())
        return out

if VAL == "synthcity":
    # Mismas métricas y misma agregación que en el banco (evalua.py): media por categoría y en conjunto
    # de las métricas en [0, 1], orientadas para que más sea mejor.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from synthcity.metrics import Metrics
    from synthcity.plugins.core.dataloader import GenericDataLoader

    SC_METRICS = {
        "sanity": ["data_mismatch", "common_rows_proportion", "nearest_syn_neighbor_distance",
                   "close_values_probability", "distant_values_probability"],
        "stats": ["jensenshannon_dist", "chi_squared_test", "feature_corr", "inv_kl_divergence", "ks_test",
                  "max_mean_discrepancy", "wasserstein_dist", "prdc", "alpha_precision"],
        "performance": ["xgb"],
        "detection": ["detection_xgb"],
        "privacy": ["delta-presence", "k-anonymization", "k-map", "distinct l-diversity", "identifiability_score"],
    }

    def score(train, hold, syn, seed):
        fill = lambda d: d.fillna(d.median(numeric_only=True))
        res = Metrics.evaluate(GenericDataLoader(fill(train), target_column=target),
                               GenericDataLoader(fill(syn), target_column=target), metrics=SC_METRICS,
                               task_type=task, random_state=seed, use_cache=False)
        out, by_cat = {}, {}
        for name, row in res.iterrows():
            v = float(row["mean"])
            if not np.isfinite(v):
                continue
            out[f"m:{name}"] = v
            if 0.0 <= v <= 1.0:
                by_cat.setdefault(name.split(".")[0], []).append(v if row["direction"] == "maximize" else 1.0 - v)
        for cat, vals in by_cat.items():
            out[cat] = float(np.mean(vals))
        allv = [x for vals in by_cat.values() for x in vals]
        out["all"] = float(np.mean(allv)) if allv else float("nan")
        return out

for gen, seed, path in runs:
    if (gen, seed) in done:
        continue
    t0 = time.perf_counter()
    try:
        train = clean(pd.read_csv(BASE / "partes" / DS / f"seed{seed}_train.csv", low_memory=False))
        hold = clean(pd.read_csv(BASE / "partes" / DS / f"seed{seed}_holdout.csv", low_memory=False))
        syn = clean(pd.read_csv(path, low_memory=False))
        if SUB_REAL:
            train = train.sample(min(SUB_REAL, len(train)), random_state=seed).reset_index(drop=True)
            hold = hold.sample(min(SUB_REAL, len(hold)), random_state=seed).reset_index(drop=True)
        if MAX_SYN and len(syn) > MAX_SYN:
            syn = syn.sample(MAX_SYN, random_state=seed).reset_index(drop=True)
        rec, err = score(train, hold, syn, seed), None
    except Exception as exc:  # noqa: BLE001
        rec, err = {}, f"{type(exc).__name__}: {exc}"[:300]
    rec.update({"validador": VAL, "dataset": DS, "generador": gen, "semilla": seed, "error": err, "sub_real": SUB_REAL,
                "segundos": round(time.perf_counter() - t0, 1)})
    with OUT.open("a") as fh:
        fh.write(json.dumps(rec, default=float) + "\n")
    print(VAL, DS, gen, seed, rec.get("segundos"), err or "", flush=True)
