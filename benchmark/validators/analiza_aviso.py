"""Opción B: un aviso aparte de la nota cuando una métrica sale de su banda calibrada.

Uso: analiza_aviso.py <dir_res> <dir_res_estudio> <dir_cache> <salida.md>

El aviso solo usa lo que METIS ya tiene en producción, la calibración de la propia tabla real (no
las tablas buenas del banco).  Para cada métrica calibrable de fidelidad y privacidad, su ancla
superior son las iteraciones de un tercio real no visto frente al tercio de entrenamiento
(``upper_mean``, ``upper_sd``).  Una tabla recibe aviso en la métrica m si

    z_m = (upper_mean − valor) / max(upper_sd, SD_FLOOR) ≥ K

es decir, si la métrica queda K desviaciones por debajo de lo que da otra muestra real.

Se cuenta, para K = 3, 4, 5, 6:
  buenas avisadas   tablas buenas (de 20) con algún aviso
  defectos avisados  defectos cuyo aviso salta en la mayoría de las 5 muestras (mediana de max z_m ≥ K)
  recuperados       defectos que ninguna nota ve (z ≥ 3 frente a las buenas) y el aviso sí
y qué métricas avisan en las tablas buenas.
"""

from __future__ import annotations

import glob
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

RES, REST, CACHE, OUT = (Path(a) for a in sys.argv[1:5])
SD_FLOOR, ZD = 0.005, 3.0
KS = (3.0, 4.0, 5.0, 6.0)
KEY = ["dataset", "muestra", "defecto"]
NOTES = ["composite", "fidelity", "utility", "privacy", "validez"]


def read(pattern):
    rows = []
    for f in sorted(glob.glob(str(pattern))):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if not r.get("error") and r.get("validador", "metis") == "metis":
                rows.append(r)
    return rows


recs = []
for r in read(RES / "metis.jsonl") + read(REST / "metis_est_*.jsonl"):
    d = {k: r[k] for k in KEY}
    d.update({s: r.get(s) for s in NOTES})
    d["metricas"] = r.get("metricas") or {}
    recs.append(d)
DF = pd.DataFrame(recs).drop_duplicates(KEY).reset_index(drop=True)
DATASETS = sorted(DF.dataset.unique())


def bounds_of(ds):
    files = sorted(CACHE.glob(f"bounds_banco3_{ds}_calibration_*.json"), key=lambda p: p.stat().st_mtime)
    files = [f for f in files if "_plano_" not in f.name]
    assert files, f"sin calibración para {ds}"
    b = json.load(open(files[-1], encoding="utf-8"))
    return files[-1].name, {m: e for m, e in b["bounds"].items()
                            if e.get("calibratable") and e.get("family") in ("fidelity", "privacy")}


BOUNDS = {ds: bounds_of(ds) for ds in DATASETS}


def zw(row):
    """{métrica: z_m} de una tabla."""
    b = BOUNDS[row.dataset][1]
    return {m: (e["upper_mean"] - v) / max(e["upper_sd"], SD_FLOOR)
            for m, e in b.items() if (v := row.metricas.get(m)) is not None and np.isfinite(v)}


DF["z"] = [zw(r) for r in DF.itertuples()]
DF["zmax"] = [max(z.values(), default=-np.inf) for z in DF.z]


def z_notes(ds):
    """{defecto: mayor z (frente a las buenas, mediana de las muestras) entre las notas}."""
    d = DF[DF.dataset == ds]
    out = {}
    for col in NOTES:
        g = d[d.defecto == "bueno"][col].astype(float).dropna()
        if len(g) < 3:
            continue
        mu, sd = g.mean(), max(g.std(ddof=1), SD_FLOOR)
        for n, x in d[d.defecto != "bueno"].groupby("defecto"):
            if x[col].notna().any():
                out[n] = max(out.get(n, -np.inf), float(np.median((mu - x[col].astype(float).dropna()) / sd)))
    return out


ZN = {ds: z_notes(ds) for ds in DATASETS}

L = [__doc__.split("Uso:")[0].strip(), ""]
w = L.append
w("Calibraciones: " + ", ".join(f"{ds} `{BOUNDS[ds][0]}` ({len(BOUNDS[ds][1])} métricas)" for ds in DATASETS) + "\n")
w("## Resumen por K\n")
w("| K | buenas avisadas | defectos avisados | defectos vistos por nota o aviso | recuperados |")
w("|---|---|---|---|---|")
detail = {}
for K in KS:
    goods = DF[DF.defecto == "bueno"]
    n_goods = int((goods.zmax >= K).sum())
    n_def = n_any = 0
    rec = []
    for ds in DATASETS:
        d = DF[(DF.dataset == ds) & (DF.defecto != "bueno")]
        for n, x in d.groupby("defecto"):
            av = float(np.median(x.zmax)) >= K
            nota = ZN[ds].get(n, -np.inf) >= ZD
            n_def += av
            n_any += av or nota
            if av and not nota:
                top = Counter(max(z, key=z.get) for z in x.z if z and max(z.values()) >= K).most_common(1)
                rec.append(f"{ds} {n} ({top[0][0] if top else '?'}, z {np.median(x.zmax):.1f})")
    n_tot = DF[DF.defecto != "bueno"].groupby(["dataset", "defecto"]).ngroups
    w(f"| {K:g} | {n_goods}/{len(goods)} | {n_def}/{n_tot} | {n_any}/{n_tot} | {len(rec)} |")
    detail[K] = rec

for K in KS:
    w(f"\n### K = {K:g}\n")
    w("Recuperados (ninguna nota los ve, el aviso sí): " + ("; ".join(detail[K]) or "ninguno") + ".\n")
    goods = DF[(DF.defecto == "bueno") & (DF.zmax >= K)]
    if len(goods):
        w("Tablas buenas avisadas:\n")
        for r in goods.itertuples():
            fired = sorted(((m, z) for m, z in r.z.items() if z >= K), key=lambda t: -t[1])
            w(f"- {r.dataset} muestra {r.muestra}: " + ", ".join(f"{m} ({z:.1f})" for m, z in fired))

w("\n## z del aviso en las tablas buenas (máximo por tabla)\n")
w("| dataset | " + " | ".join(f"m{k}" for k in sorted(DF.muestra.unique())) + " |")
w("|---|" + "---|" * DF.muestra.nunique())
for ds in DATASETS:
    g = DF[(DF.dataset == ds) & (DF.defecto == "bueno")].set_index("muestra").zmax
    w(f"| {ds} | " + " | ".join(f"{g.get(k, float('nan')):.1f}" for k in sorted(DF.muestra.unique())) + " |")
OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
print("escrito", OUT)
