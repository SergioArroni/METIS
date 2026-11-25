"""Detecciones de synthcity en el banco 2, con el mismo criterio que el resto de validadores.

Uso: analiza_sc.py <synthcity.jsonl> <salida.md>
z = (media de las 5 buenas − nota) / max(sd de las buenas, 0,005), mediana sobre las 5 muestras; detectado si z >= 3.
Nota general: «all» (media de las métricas en [0, 1], orientadas). Más favorable: la mayor z entre «all» y las
medias por categoría (sanity, stats, performance, detection, privacy). Buenas marcadas: z >= 3 de una buena frente
a las otras cuatro con alguna nota.
"""
import json
import sys

import numpy as np
import pandas as pd

SD, ZD = 0.005, 3.0
NOTES = ["all", "sanity", "stats", "performance", "detection", "privacy"]
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
err = [r for r in rows if r.get("error")]
df = pd.DataFrame([r for r in rows if not r.get("error")])
fam = {(r["dataset"], r["defecto"]): r["familia_defecto"] for r in rows if r["defecto"] != "bueno"}


def ztab(col):
    out = {}
    for ds, d in df.groupby("dataset"):
        g = d[d.defecto == "bueno"][col].astype(float).dropna()
        if len(g) < 3:
            continue
        mu, sd = g.mean(), max(g.std(ddof=1), SD)
        for n, x in d[d.defecto != "bueno"].groupby("defecto"):
            v = x[col].astype(float).dropna()
            if len(v):
                out[(ds, n)] = float(np.median((mu - v) / sd))
    return out


zg = ztab("all")
zb = {}
for c in NOTES:
    if c in df:
        for k, z in ztab(c).items():
            zb[k] = max(zb.get(k, -np.inf), z)

fp = n = 0
for ds, d in df.groupby("dataset"):
    g = d[d.defecto == "bueno"]
    for i in range(len(g)):
        n += 1
        hit = False
        for c in NOTES:
            if c not in g:
                continue
            v = g[c].astype(float).to_numpy()
            rest = np.delete(v, i)
            rest = rest[~np.isnan(rest)]
            if len(rest) >= 3 and not np.isnan(v[i]) and (rest.mean() - v[i]) / max(rest.std(ddof=1), SD) >= ZD:
                hit = True
        fp += hit

L = [__doc__.split("Uso:")[0].strip(), "", f"Registros: {len(rows)}, con error: {len(err)}.", "",
     "| familia (casos) | general | más favorable |", "|---|---|---|"]
tot_g = tot_b = tot_n = 0
for f in ("fidelity", "utility", "privacy", "validez"):
    ks = [k for k, v in fam.items() if v == f]
    g = sum(zg.get(k, -np.inf) >= ZD for k in ks)
    b = sum(zb.get(k, -np.inf) >= ZD for k in ks)
    tot_g, tot_b, tot_n = tot_g + g, tot_b + b, tot_n + len(ks)
    L.append(f"| {f} ({len(ks)}) | {g} | {b} |")
L.append(f"| total ({tot_n}) | {tot_g} | {tot_b} |")
L += ["", f"Buenas marcadas: {fp} de {n}.", "", "## z por defecto (general / más favorable)", ""]
for k in sorted(fam):
    L.append(f"- {k[0]} {k[1]} ({fam[k]}): {zg.get(k, float('nan')):.1f} / {zb.get(k, float('nan')):.1f}")
if err:
    L += ["", "## Errores", ""] + [f"- {r['dataset']} {r['muestra']} {r['defecto']}: {r['error']}" for r in err[:30]]
open(sys.argv[2], "w", encoding="utf-8").write("\n".join(L) + "\n")
print("escrito", sys.argv[2])
