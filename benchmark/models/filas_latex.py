"""Filas LaTeX de las tablas del benchmark (Tabla 2, tabla en bruto y tablas por métrica) con la taxonomía instalada.

Uso: filas_latex.py <bench_tax3/por_semilla.csv> <dir_res_bench2> <salida.json>
"""
import glob
import json
import sys

import numpy as np
import pandas as pd

from metis.domain.taxonomy import expand_metric_ids

CSV, RES, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
DS = [("telco", "Telco"), ("cardio", "Cardio"), ("rrhh", "RRHH"), ("airbnb", "Airbnb"), ("adult", "Adult")]
GEN = ["real_data", "uniform_noise", "bootstrap", "smotenc", "delete_zero", "delete_mean", "bn", "cart",
       "gaussian_copula", "ctgan", "tvae", "dpctgan"]
tex = lambda s: s.replace("_", r"\_")


def mi(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if not len(v):
        return r"\textit{NA}"
    return f"{np.median(v):.4f} ({np.percentile(v, 75) - np.percentile(v, 25):.4f})"


df = pd.read_csv(CSV)
out = {"norm": [], "raw": [], "metricas": {}}
for ds, name in DS:
    for g in GEN:
        d = df[(df.dataset == ds) & (df.generator == g)]
        out["norm"].append(f"    {name} & {tex(g)} & " + " & ".join(mi(d[c]) for c in ("fidelity", "utility", "privacy", "composite")) + r" \\")
        out["raw"].append(f"    {name} & {tex(g)} & " + " & ".join(mi(d[c + "_raw"]) for c in ("fidelity", "utility", "privacy")) + r" \\")
    out["norm"].append("    \\midrule")
    out["raw"].append("    \\midrule")
out["norm"] = out["norm"][:-1]
out["raw"] = out["raw"][:-1]

recs = []
for f in sorted(glob.glob(f"{RES}/m_*.jsonl")):
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        if not r.get("error"):
            recs.append(r)
ORDER = {"fidelity": expand_metric_ids(["fidelity"]), "utility": ["utility.tstr", "utility.log_loss"],
         "privacy": expand_metric_ids(["privacy"])}
for ds, _ in DS:
    for fam, ms in ORDER.items():
        rows = []
        for m in ms:
            vals = [mi([r["metricas"].get(m) for r in recs if r["dataset"] == ds and r["generador"] == g]) for g in GEN]
            if all(v == r"\textit{NA}" for v in vals) and fam != "fidelity":
                continue
            rows.append(f"    \\texttt{{{tex(m)}}} & " + " & ".join(vals) + r" \\")
        out["metricas"][f"{ds}_{fam}"] = rows
json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
print("escrito", OUT, {k: len(v) for k, v in out["metricas"].items()})
