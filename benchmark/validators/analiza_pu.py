"""¿Miden lo mismo las métricas de cada subcategoría de privacidad y de utilidad?  Y ¿qué detecta cada una?

Uso: analiza_pu.py <dir_res_pu> <dir_res_bench2> <dir_cache_banco4> <salida.md>

Dos fuentes: el banco de privacidad y utilidad (buenas + 11 defectos de privacidad y 8 de utilidad por tabla, con
TTS, TRTS y TTRS además de TSTR y log-loss) y las 240 sintéticas del benchmark (12 generadores).
  ρ        de Spearman entre métricas sobre todas las tablas de un dataset, mediana entre datasets
  grupos   ρ interno (pares del grupo) y externo (sus métricas con las de la misma familia fuera del grupo)
  detección  defectos detectados (z >= 3 frente a las 5 buenas, mediana de 5 muestras) por cada métrica y
           por la nota de su familia (reagregada con la taxonomía instalada, solo TSTR y log-loss en utilidad);
           buenas marcadas (z >= 3 frente a las otras cuatro)
"""
from __future__ import annotations

import glob
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import reagrega as ra
from metis.calibrate.core.bounds import CalibrationBounds

PU, BENCH, CACHE, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
GROUPS = {
    "privacy.attacks": ["privacy.mia", "privacy.inference_attack", "privacy.record_linkage"],
    "privacy.syntactic": ["privacy.k_anonymity", "privacy.l_diversity", "privacy.t_closeness"],
    "privacy.empirical_similarity": ["privacy.dcr", "privacy.nnaa"],
    "utility.standalone": ["utility.tstr", "utility.tts", "utility.trts", "utility.ttrs"],
    "utility.probabilistic": ["utility.log_loss"],
}
METRICS = [m for ms in GROUPS.values() for m in ms]
EXTRA = {"utility.tts", "utility.trts", "utility.ttrs", "fidelity.missingness"}


def read(pattern):
    out = []
    for f in sorted(glob.glob(str(pattern))):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if not r.get("error"):
                out.append(r)
    return out


pu = read(PU / "m_*.jsonl")
seen, PUR = set(), []
for r in pu:
    k = (r["dataset"], r["muestra"], r["defecto"])
    if k not in seen:
        seen.add(k)
        PUR.append(r)
bench = read(BENCH / "m_*.jsonl")


def frame(recs, key):
    return pd.DataFrame([{**{c: r[c] for c in key}, **{m: (r.get("metricas") or {}).get(m) for m in METRICS}}
                         for r in recs])


A = frame(PUR, ["dataset", "muestra", "defecto", "familia_defecto"])
Bn = frame(bench, ["dataset", "generador", "semilla"])


def rho_table(df):
    acc = {}
    for ds, d in df.groupby("dataset"):
        for a, b in combinations(METRICS, 2):
            x = d[[a, b]].dropna().astype(float)
            if len(x) > 10 and x[a].nunique() > 2 and x[b].nunique() > 2:
                r = spearmanr(x[a], x[b])[0]
                if np.isfinite(r):
                    acc.setdefault(frozenset((a, b)), []).append(r)
    return {k: float(np.median(v)) for k, v in acc.items() if len(v) >= 2}


L = [__doc__.split("Uso:")[0].strip(), ""]
w = L.append
f2 = lambda v: f"{np.median(v):.2f}" if v else "—"
for name, df in (("banco de privacidad y utilidad", A), ("benchmark (240 sintéticas)", Bn)):
    R = rho_table(df)
    w(f"\n## Coherencia de los grupos: {name}\n")
    w("| grupo | métricas | ρ interno | ρ externo (misma familia) |")
    w("|---|---|---|---|")
    for g, ms in GROUPS.items():
        fam = g.split(".")[0]
        others = [m for gg, mm in GROUPS.items() if gg.startswith(fam) and gg != g for m in mm]
        inner = [R[frozenset(p)] for p in combinations(ms, 2) if frozenset(p) in R]
        outer = [R[frozenset((a, o))] for a in ms for o in others if frozenset((a, o)) in R]
        w(f"| {g} | {', '.join(m.split('.')[1] for m in ms)} | {f2(inner)} | {f2(outer)} |")
    w("\nρ entre pares:\n")
    for fam in ("privacy", "utility"):
        ms = [m for m in METRICS if m.startswith(fam)]
        w("| | " + " | ".join(m.split(".")[1] for m in ms) + " |")
        w("|---|" + "---|" * len(ms))
        for a in ms:
            w(f"| {a.split('.')[1]} | " + " | ".join("1" if a == b else (f"{R[frozenset((a, b))]:.2f}" if frozenset((a, b)) in R else "—") for b in ms) + " |")
        w("")

# detección en el banco de privacidad y utilidad
BOUNDS = {}
for ds in sorted(A.dataset.unique()):
    files = [f for f in sorted(CACHE.glob(f"bounds_banco4_{ds}_*.json"), key=lambda p: p.stat().st_mtime) if "_plano_" not in f.name]
    BOUNDS[ds] = CalibrationBounds.load(str(files[-1]))
notes = []
for r in PUR:
    rr = {**r, "metricas": {k: v for k, v in (r.get("metricas") or {}).items() if k not in EXTRA}}
    a = ra.aggregate(rr, BOUNDS[r["dataset"]])
    notes.append({"dataset": r["dataset"], "muestra": r["muestra"], "defecto": r["defecto"],
                  "nota_privacidad": a["privacy"], "nota_utilidad": a["utility"], "compuesto": a["composite"]})
A = A.merge(pd.DataFrame(notes), on=["dataset", "muestra", "defecto"])
FAM = {(r["dataset"], r["defecto"]): r.get("familia_defecto") for r in PUR if r["defecto"] != "bueno"}
cols = METRICS + ["nota_privacidad", "nota_utilidad", "compuesto"]
Z = {c: ra.z_table(A, c) for c in cols}
w("\n## Detección en el banco de privacidad y utilidad (z ≥ 3)\n")
defs = sorted(set(A.defecto) - {"bueno"})
fams = {d: next((FAM[k] for k in FAM if k[1] == d), "") for d in defs}
w("| defecto | familia | " + " | ".join(c.split(".")[-1] for c in cols) + " |")
w("|---|---|" + "---|" * len(cols))
for d in sorted(defs, key=lambda d: (fams[d], d)):
    cells = []
    for c in cols:
        n = sum(Z[c].get((ds, d), -np.inf) >= ra.ZD for ds in sorted(A.dataset.unique()))
        tot = sum((ds, d) in Z[c] for ds in sorted(A.dataset.unique()))
        cells.append(f"{n}/{tot}")
    w(f"| {d} | {fams[d]} | " + " | ".join(cells) + " |")
w("\nBuenas marcadas (z ≥ 3 frente a las otras cuatro):\n")
for c in cols:
    fp, n = ra.goods_fp(A, [c])
    w(f"- {c}: {fp}/{n}")
open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
print("escrito", OUT)
