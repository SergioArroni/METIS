"""Evalúa el banco de defectos con un validador.

Uso: evalua.py <metis|sdmetrics|syntheval|synthcity> <dir_casos>/<dataset> <salida.jsonl> [k_desde k_hasta]

Cada validador puntúa cada caso con las puntuaciones que él mismo publica, sin reescalarlas:

  metis      fidelity, utility, privacy y composite (holdout H, objetivo de la tabla)
  sdmetrics  quality (QualityReport: Column Shapes + Column Pair Trends, la nota de SDV) y
             dcr_overfitting (DCROverfittingProtection, con H como validación)
  syntheval  preset full_eval: media de n_val por dimensión (utility, privacy) y de todas
  synthcity  ver la rama correspondiente

Reanudable: cada caso se añade a la salida en cuanto termina.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
VAL, CASES, OUT = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
meta = json.load(open(CASES / "meta.json"))
K_FROM = int(sys.argv[4]) if len(sys.argv) > 4 else 0
K_TO = int(sys.argv[5]) if len(sys.argv) > 5 else meta["n_good"]
target, cat_cols = meta["target"], meta["cat_cols"]
# Los identificadores no describen la población: METIS los excluye y los demás no deben verlos.
IDS = [c for c, s in meta["schema"].items() if (s if isinstance(s, str) else s.get("type")) == "id"]
schema = {c: s for c, s in meta["schema"].items() if c not in IDS}


def load(name):
    if VAL == "synthcity":
        return pd.read_csv(CASES / f"{name}.csv.gz", dtype={c: str for c in cat_cols},
                           keep_default_na=False, na_values={c: [""] for c in schema if c not in cat_cols})
    return pd.read_pickle(CASES / f"{name}.pkl").drop(columns=IDS)


A, H = load("A"), load("H")


def clean(df):
    """Tipos explícitos para los validadores que los infieren: numéricas a float, categóricas a str."""
    out = df.copy()
    for c in out.columns:
        if c in cat_cols:
            out[c] = out[c].astype(str)
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    return out


if VAL == "metis":
    from metis.interface.sdk import Evaluator

    ev = Evaluator()
    conf = {"data": {"schema": schema, "target": target, "task_type": "classification"},
            "metrics": ["fidelity", "utility", "privacy"], "evaluation": {"n_runs": 1}}

    def score(synth, k):
        s = ev.evaluate(A, synth, conf, seed=k, holdout=H)
        a = s.aggregates
        return {"fidelity": a.get("fidelity_score"), "utility": a.get("utility_score"),
                "privacy": a.get("privacy_score"), "composite": a.get("composite_score")}

elif VAL == "sdmetrics":
    from sdmetrics.reports.single_table import DiagnosticReport, QualityReport
    from sdmetrics.single_table import DCROverfittingProtection

    md = {"columns": {c: {"sdtype": "categorical" if c in cat_cols else "numerical"} for c in A.columns}}
    A_c, H_c = clean(A), clean(H)

    def score(synth, k):
        s = clean(synth)
        rep = QualityReport()
        rep.generate(A_c, s, md, verbose=False)
        props = rep.get_properties().set_index("Property")["Score"].to_dict()
        dcr = DCROverfittingProtection.compute(real_training_data=A_c, synthetic_data=s,
                                               real_validation_data=H_c, metadata={"tables": {"t": md}},
                                               table_name="t")
        diag = DiagnosticReport()
        diag.generate(A_c, s, md, verbose=False)
        dprops = diag.get_properties().set_index("Property")["Score"].to_dict()
        return {"quality": rep.get_score(), "column_shapes": props.get("Column Shapes"),
                "column_pair_trends": props.get("Column Pair Trends"), "dcr_overfitting": dcr,
                "diagnostic": diag.get_score(), "data_validity": dprops.get("Data Validity")}

elif VAL == "syntheval":
    from syntheval import SynthEval

    A_c, H_c = clean(A), clean(H)
    se = SynthEval(A_c, holdout_dataframe=H_c, cat_cols=cat_cols, verbose=False, enable_plots=False,
                   console="off", missing_directive="drop", show_warnings=False)

    def score(synth, k):
        np.random.seed(k)
        res = se.evaluate(clean(synth), analysis_target=target, presets_file="full_eval")
        res = res[np.isfinite(pd.to_numeric(res["n_val"], errors="coerce"))]
        out = {f"m:{r.metric}": float(r.n_val) for r in res.itertuples()}
        for dim, g in res.groupby("dim"):
            out[dim] = float(g["n_val"].astype(float).mean())
        out["all"] = float(res["n_val"].astype(float).mean())
        return out

elif VAL == "synthcity":
    # synthcity no publica una nota agregada: cada métrica sale con su dirección (maximize/minimize).
    # Se guarda cada métrica y, como con SynthEval, la media por categoría y en conjunto de las que
    # están en [0, 1], orientadas para que más sea mejor (1 - v si la dirección es minimize).
    import os

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

    def _num(df):
        # categóricas como texto: synthcity las codifica por su cuenta; huecos numéricos a la mediana
        out = clean(df)
        return out.fillna(out.median(numeric_only=True))

    A_n = _num(A)
    X_gt = GenericDataLoader(A_n, target_column=target)

    def score(synth, k):
        s = _num(synth)
        X_syn = GenericDataLoader(s, target_column=target)
        res = Metrics.evaluate(X_gt, X_syn, metrics=SC_METRICS, task_type="classification",
                               random_state=k, use_cache=False)
        out, by_cat = {}, {}
        for name, row in res.iterrows():
            v = float(row["mean"])
            if not np.isfinite(v):
                continue
            out[f"m:{name}"] = v
            if 0.0 <= v <= 1.0:
                o = v if row["direction"] == "maximize" else 1.0 - v
                by_cat.setdefault(name.split(".")[0], []).append(o)
        for cat, vals in by_cat.items():
            out[cat] = float(np.mean(vals))
        allv = [x for vals in by_cat.values() for x in vals]
        out["all"] = float(np.mean(allv)) if allv else float("nan")
        return out

else:
    raise SystemExit(f"validador desconocido: {VAL}")

done = set()
if OUT.exists():
    for line in OUT.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["dataset"] == meta["dataset"]:
            done.add((r["muestra"], r["defecto"]))

for k in range(K_FROM, K_TO):
    for name in meta["family"]:
        if (k, name) in done:
            continue
        t0 = time.perf_counter()
        try:
            rec = score(load(f"k{k}__{name}"), k)
            err = None
        except Exception as exc:  # noqa: BLE001
            rec, err = {}, f"{type(exc).__name__}: {exc}"[:300]
        rec.update({"validador": VAL, "dataset": meta["dataset"], "muestra": k, "defecto": name,
                    "familia_defecto": meta["family"][name], "error": err,
                    "segundos": round(time.perf_counter() - t0, 1)})
        with OUT.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=float) + "\n")
    print(VAL, meta["dataset"], "muestra", k, "hecha", flush=True)
