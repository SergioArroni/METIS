"""E5: los tres validadores sobre las 240 sintéticas del benchmark v2.

Uso: analiza_e5.py <dir_res> <por_semilla_metis.csv> <salida.md>

Por dataset: nota media por generador (5 semillas) de cada validador, el orden que da cada uno,
la concordancia entre órdenes (Spearman) y dónde quedan las dos referencias: real_data (filas
reales no vistas, el techo honesto) y bootstrap (remuestreo del entrenamiento, copia filas).
Copiar se premia si bootstrap queda por encima de real_data.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

RES, METIS_CSV, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])

rows = []
for f in sorted(glob.glob(str(RES / "e5_*.jsonl"))):
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        if not r.get("error"):
            rows.append({k: v for k, v in r.items() if not k.startswith("m:")})
oth = pd.DataFrame(rows)
met = pd.read_csv(METIS_CSV).rename(columns={"generator": "generador", "seed": "semilla"})
met["validador"] = "metis"

SCORES = {"metis": ["composite", "fidelity", "privacy"], "sdmetrics": ["quality", "dcr_overfitting"],
          "syntheval": ["all", "u", "p"],
          "synthcity": ["all", "stats", "privacy"]}
GENERAL = {"metis": "composite", "sdmetrics": "quality", "syntheval": "all", "synthcity": "all"}
frames = [met[["dataset", "generador", "semilla", "validador"] + SCORES["metis"]]]
for v in ("sdmetrics", "syntheval", "synthcity"):
    d = oth[oth.validador == v]
    if len(d):
        frames.append(d[["dataset", "generador", "semilla", "validador"] + [c for c in SCORES[v] if c in d.columns]
                        + (["sub_real"] if "sub_real" in d.columns else [])])
df = pd.concat(frames, ignore_index=True)

L = []
w = L.append
w("# E5: validadores sobre las sintéticas del benchmark v2\n")
if "sub_real" in df.columns and df["sub_real"].notna().any():
    w("SynthEval en cardio con entrenamiento, holdout y sintética submuestreados a 10 000 filas (completo no cabe en 30 GB).\n")
summary = []
for ds in sorted(df.dataset.unique()):
    d = df[df.dataset == ds]
    tab = {}
    for v, cols in SCORES.items():
        x = d[d.validador == v]
        for c in cols:
            if c in x.columns and x[c].notna().any():
                tab[f"{v}:{c}"] = x.groupby("generador")[c].mean()
    t = pd.DataFrame(tab)
    gen_cols = [f"{v}:{GENERAL[v]}" for v in SCORES if f"{v}:{GENERAL[v]}" in t.columns]
    t = t.sort_values(gen_cols[0], ascending=False)
    w(f"\n## {ds}\n")
    w("| generador | " + " | ".join(t.columns) + " |")
    w("|---|" + "---|" * len(t.columns))
    for g, r in t.iterrows():
        w(f"| {g} | " + " | ".join("—" if pd.isna(x) else f"{x:.3f}" for x in r) + " |")
    w("\nPuesto de cada referencia según la nota general (1 = mejor) y Spearman entre órdenes:\n")
    w("| validador | n generadores | puesto real_data | puesto bootstrap | bootstrap > real_data |")
    w("|---|---|---|---|---|")
    for c in gen_cols:
        s = t[c].dropna()
        rk = s.rank(ascending=False)
        rb, rr = rk.get("bootstrap"), rk.get("real_data")
        prem = None if rb is None or rr is None else bool(s["bootstrap"] > s["real_data"])
        w(f"| {c} | {len(s)} | {rr if rr is not None else '—'} | {rb if rb is not None else '—'} | "
          f"{'—' if prem is None else ('sí' if prem else 'no')} |")
        summary.append((ds, c, rr, rb, prem))
    w("")
    for i, a in enumerate(gen_cols):
        for b in gen_cols[i + 1:]:
            both = t[[a, b]].dropna()
            if len(both) >= 4:
                rho, p = spearmanr(both[a], both[b])
                w(f"- Spearman {a} vs {b}: ρ = {rho:.2f} (p = {p:.3g}, n = {len(both)})")

w("\n## Resumen: ¿se premia copiar? (bootstrap por encima de real_data en la nota general)\n")
w("| dataset | validador | puesto real_data | puesto bootstrap | premia copiar |")
w("|---|---|---|---|---|")
for ds, c, rr, rb, prem in summary:
    w(f"| {ds} | {c} | {rr} | {rb} | {'—' if prem is None else ('sí' if prem else 'no')} |")
OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
print("escrito", OUT)
