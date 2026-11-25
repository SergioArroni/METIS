"""¿Pierden las vistas los matices locales?  METIS v3 sobre defectos de alta cardinalidad, categorías raras
y colas pesadas (gen_matices.py), frente a las buenas del banco 2.

Uso: analiza_matices.py <res_banco2/metis.jsonl> <res_matices/*.jsonl glob> <dir_casos_matices> <dir_cache> <salida.md>

Por defecto: z (mediana de 5 muestras, frente a las 5 buenas) del compuesto, la fidelidad, la nota más
favorable (compuesto, familias y validez), la subcategoría que más cae y la métrica que más cae; y la z
del aviso fuera de la nota (§24: métrica calibrada K desviaciones bajo otra muestra real, K = 4).
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RES, PAT, CASES, CACHE, OUT = sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5])
SD_FLOOR, ZD, KW = 0.005, 3.0, 4.0
KEY = ["dataset", "muestra", "defecto"]
NOTES = ["composite", "fidelity", "utility", "privacy", "validez"]


def read(files):
    rows = []
    for f in files:
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if not r.get("error") and r.get("validador", "metis") == "metis":
                d = {k: r[k] for k in KEY}
                d.update({s: r.get(s) for s in NOTES})
                d.update({f"m:{k}": v for k, v in (r.get("metricas") or {}).items()})
                d.update({f"h:{k}": v for k, v in (r.get("jerarquia") or {}).items()})
                rows.append(d)
    return rows


base = [r for r in read([RES]) if r["defecto"] == "bueno"]
new = read(sorted(glob.glob(PAT)))
DF = pd.DataFrame(base + new).drop_duplicates(KEY)
MC = [c for c in DF.columns if c.startswith("m:")]
SUB = [c for c in DF.columns if c.startswith("h:") and c.count(".") == 2]


def bounds_of(ds):
    files = [f for f in sorted(CACHE.glob(f"bounds_banco3_{ds}_calibration_*.json"), key=lambda p: p.stat().st_mtime)
             if "_plano_" not in f.name]
    b = json.load(open(files[-1], encoding="utf-8"))["bounds"]
    return {m: e for m, e in b.items() if e.get("calibratable") and e.get("family") in ("fidelity", "privacy")}


def zcol(d, col):
    g = d[d.defecto == "bueno"][col].astype(float).dropna()
    if len(g) < 3:
        return {}
    mu, sd = g.mean(), max(g.std(ddof=1), SD_FLOOR)
    return {n: float(np.median((mu - x[col].astype(float).dropna()) / sd))
            for n, x in d[d.defecto != "bueno"].groupby("defecto") if x[col].notna().any()}


def best(d, cols):
    out = {}
    for c in cols:
        for n, z in zcol(d, c).items():
            if z > out.get(n, (-np.inf, ""))[0]:
                out[n] = (z, c[2:])
    return out


L = [__doc__.split("Uso:")[0].strip(), ""]
w = L.append
f1 = lambda z: "—" if z is None or not np.isfinite(z) else (f"**{z:.1f}**" if z >= ZD else f"{z:.1f}")
w("| dataset | columnas afectadas | defecto | compuesto | fidelidad | más favorable | subcategoría (z) | métrica (z) | aviso z (métrica) |")
w("|---|---|---|---|---|---|---|---|---|")
for ds in sorted(DF.dataset.unique()):
    d = DF[DF.dataset == ds]
    info = json.load(open(CASES / ds / "meta.json"))["matices"]
    zc, zf = zcol(d, "composite"), zcol(d, "fidelity")
    zn = {}
    for c in NOTES:
        for n, z in zcol(d, c).items():
            zn[n] = max(zn.get(n, -np.inf), z)
    zs, zm = best(d, SUB), best(d, MC)
    b = bounds_of(ds)
    for n in sorted(set(d.defecto) - {"bueno"}):
        x = d[d.defecto == n]
        ws = []
        for _, r in x.iterrows():
            zz = {m[2:]: (e["upper_mean"] - r[m]) / max(e["upper_sd"], SD_FLOOR)
                  for m, e in ((f"m:{k}", e) for k, e in b.items()) if m in r and pd.notna(r[m])}
            ws.append(max(zz.items(), key=lambda t: t[1]) if zz else ("—", np.nan))
        wz = float(np.median([t[1] for t in ws]))
        wm = pd.Series([t[0] for t in ws]).mode().iloc[0]
        grp = ("alta_cardinalidad" if n.startswith("card") else "cola_pesada" if n.startswith("cola")
               else "raras_5_una" if n in ("raras_5_una",) else "raras_1")
        cols = ", ".join(f"{k} ({v})" for k, v in (info.get(grp) or {}).items())
        w(f"| {ds} | {cols} | {n} | {f1(zc.get(n))} | {f1(zf.get(n))} | {f1(zn.get(n))} | "
          f"{f1(zs.get(n, (np.nan,))[0])} ({zs.get(n, (0, '—'))[1]}) | {f1(zm.get(n, (np.nan,))[0])} ({zm.get(n, (0, '—'))[1]}) | "
          f"{'**' + format(wz, '.1f') + '**' if wz >= KW else format(wz, '.1f')} ({wm}) |")
w("\nColumnas: alta cardinalidad = nominal (niveles); raras = nominal (niveles raros o masa rara); "
  "cola pesada = numérica (curtosis en exceso).  Negrita: z ≥ 3 (notas y métricas) o z ≥ 4 (aviso).")
OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
print("escrito", OUT)
