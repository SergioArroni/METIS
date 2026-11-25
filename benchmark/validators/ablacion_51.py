"""Ablación de la armonización (§5.1 del paper) sobre el banco de defectos, sin volver a evaluar.

Uso: ablacion_51.py <res/metis.jsonl> <dir_cache> <salida.md>

Variantes de la armonización de las métricas de fidelidad (privacidad y utilidad no se armonizan):
  metis         la de METIS: una transformación por familia (N1 1 − d, N2 exp(−d/s), N3 similitud, N4 1/(1 + |d|))
  inversa       1/(1 + d) para todas las discrepancias
  exponencial   exp(−d) para todas (s = 1)
  lineal        max(0, 1 − d) para todas
  s*k           N2 con su escala s multiplicada por k (k = 0,25; 0,5; 2; 4)
La discrepancia d se recupera invirtiendo la armonización de METIS; las iteraciones de la calibración pasan por
la misma transformación y las anclas y exclusiones se recalculan.  Se informa de las detecciones del banco
(z >= 3, mediana de 5 muestras), de las buenas marcadas y del ρ de Spearman del compuesto con el de METIS.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import reagrega as ra

RES, CACHE, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
recs = ra.load_records([RES])
DATASETS = sorted({r["dataset"] for r in recs})
BOUNDS = {ds: ra.load_bounds(CACHE, ds) for ds in DATASETS}
FAM = {(r["dataset"], r["defecto"]): r.get("familia_defecto") for r in recs if r["defecto"] != "bueno"}
VARIANTS = ["metis", "inversa", "exponencial", "lineal", "s*0.25", "s*0.5", "s*2", "s*4"]


def run(variant):
    fn = ra.harmonize(variant)
    nb = {ds: ra.rebound(BOUNDS[ds], fn) for ds in DATASETS}
    rows = []
    for r in recs:
        a = ra.aggregate(r, nb[r["dataset"]], fn)
        rows.append({**{k: r[k] for k in ra.KEY}, **{k: a[k] for k in ra.NOTES}, "stored": r.get("composite")})
    return pd.DataFrame(rows)


L = [__doc__.split("Uso:")[0].strip(), ""]
w = L.append
res = {v: run(v) for v in VARIANTS}
base = res["metis"]
err = float(np.nanmax(np.abs(base.composite.astype(float) - base.stored.astype(float))))
w(f"Control: la reagregación de METIS reproduce el compuesto guardado con un error máximo de {err:.2e}.\n")
w(f"Familias de los defectos: {sorted(set(FAM.values()), key=str)}\n")
w("| variante | general | alguna | fidelidad (nota de fidelidad) | utilidad | privacidad | validez | buenas marcadas | ρ con METIS (mín. por dataset) |")
w("|---|---|---|---|---|---|---|---|---|")
for v in VARIANTS:
    d = res[v]
    c = ra.counts(d, FAM)
    rho = min(spearmanr(d[d.dataset == ds].composite, base[base.dataset == ds].composite)[0] for ds in DATASETS)
    w(f"| {v} | {c['general']}/{c['n']} | {c['alguna']}/{c['n']} | {c['fid_fidelidad']}/{c['n_fid']} | "
      f"{c['utility_general']}/{c['utility_alguna']} | {c['privacy_general']}/{c['privacy_alguna']} | "
      f"{c['validez_general']}/{c['validez_alguna']} | {c['fp']}/{c['n_buenas']} | {rho:.3f} |")
w("\nUtilidad, privacidad y validez: general / alguna.\n")

w("## Defectos que cambian de estado frente a METIS (nota general)\n")
zb = ra.z_table(base, "composite")
for v in VARIANTS[1:]:
    zv = ra.z_table(res[v], "composite")
    gan = sorted(k for k in zb if zb[k] < ra.ZD <= zv.get(k, -np.inf))
    pierde = sorted(k for k in zb if zv.get(k, -np.inf) < ra.ZD <= zb[k])
    f = lambda ks: "; ".join(f"{a} {b}" for a, b in ks) or "—"
    w(f"- **{v}**: gana {len(gan)} ({f(gan)}); pierde {len(pierde)} ({f(pierde)})")

w("\n## Ruido de las buenas: media ± sd del compuesto\n")
w("| variante | " + " | ".join(DATASETS) + " |")
w("|---|" + "---|" * len(DATASETS))
for v in VARIANTS:
    d = res[v]
    cells = [f"{d[(d.dataset == ds) & (d.defecto == 'bueno')].composite.mean():.3f} ± "
             f"{d[(d.dataset == ds) & (d.defecto == 'bueno')].composite.std(ddof=1):.3f}" for ds in DATASETS]
    w(f"| {v} | " + " | ".join(cells) + " |")
open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
print("escrito", OUT)
