"""¿Es útil cada métrica de METIS y tiene sentido agregarla con las de su grupo?  METIS v3, banco 2
más los defectos del estudio (140 defectos, 20 tablas buenas, 4 datasets).

Grupo L1 de una métrica: su subcategoría, o su categoría si la categoría no tiene subcategorías
(global, detection, mechanism_based, probabilistic).  Es el primer nivel en que METIS promedia.

Por métrica (regla fijada antes de ver los resultados, 3 nov):
  detecta      defectos con z ≥ 3 (frente a las 5 buenas del dataset, mediana de las 5 muestras)
  única        defectos que solo ella detecta entre todas las métricas de METIS
  única grupo  defectos que detecta y ningún otro miembro de su grupo
  FP           tablas buenas marcadas (z ≥ 3 frente a las otras cuatro)
  ρ grupo      mediana de su ρ de Spearman con los demás miembros del grupo (mediana entre datasets,
               sobre todas las tablas del dataset)
  mejor fuera  el grupo ajeno con mayor ρ medio con ella, y ese ρ
  max ρ        la métrica más parecida (cualquier grupo) y su ρ
Veredicto:
  sin señal    no detecta nada (puede seguir sirviendo de control, pero no aporta al banco)
  ruidosa      marca alguna tabla buena
  redundante   ρ ≥ 0,9 con otra métrica y ninguna detección única
  mal ubicada  el mejor grupo ajeno supera su ρ de grupo en más de 0,1
  útil         el resto
Por grupo:
  ρ interno    mediana de los ρ entre pares del grupo; ρ externo, de sus miembros con el resto
  vistos       defectos que detecta algún miembro; nota, cuántos de ellos detecta la nota del grupo
               (la dilución en el primer nivel)
  coherente    ρ interno ≥ 0,5 y mayor que el externo
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

from metis.domain.taxonomy import get_metric_hierarchy

RES, REST, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
SD_FLOOR, ZD = 0.005, 3.0
KEY = ["dataset", "muestra", "defecto"]


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
    d.update({f"m:{k}": v for k, v in (r.get("metricas") or {}).items()})
    d.update({f"h:{k}": v for k, v in (r.get("jerarquia") or {}).items()})
    recs.append(d)
DF = pd.DataFrame(recs).drop_duplicates(KEY).reset_index(drop=True)
DATASETS = sorted(DF.dataset.unique())
METRICS = sorted(c[2:] for c in DF.columns if c.startswith("m:") and DF[c].notna().any())


def group_of(m):
    h = get_metric_hierarchy(m)
    return ".".join(x for x in (h["family"], h["category"], h.get("subcategory")) if x)


GROUP = {m: group_of(m) for m in METRICS}
GROUPS = {}
for m, g in GROUP.items():
    GROUPS.setdefault(g, []).append(m)


def z_of(ds, col):
    d = DF[DF.dataset == ds]
    if col not in d or d[col].isna().all():
        return {}
    g = d[d.defecto == "bueno"][col].astype(float).dropna()
    if len(g) < 3:
        return {}
    mu, sd = g.mean(), max(g.std(ddof=1), SD_FLOOR)
    return {n: float(np.median((mu - x[col].astype(float).dropna()) / sd))
            for n, x in d[d.defecto != "bueno"].groupby("defecto") if x[col].notna().any()}


def detected(col):
    return {(ds, n) for ds in DATASETS for n, z in z_of(ds, col).items() if z >= ZD}


def goods_fp(col):
    fp = n = 0
    for ds in DATASETS:
        g = DF[(DF.dataset == ds) & (DF.defecto == "bueno")][col].astype(float).dropna().to_numpy()
        if len(g) < 4:
            continue
        n += len(g)
        fp += sum((np.delete(g, i).mean() - g[i]) / max(np.delete(g, i).std(ddof=1), SD_FLOOR) >= ZD
                  for i in range(len(g)))
    return int(fp), n


DET = {m: detected(f"m:{m}") for m in METRICS}
FP = {m: goods_fp(f"m:{m}") for m in METRICS}

# ρ de Spearman por dataset, mediana entre datasets (al menos 2 con datos).
_rho = {}
for ds in DATASETS:
    d = DF[DF.dataset == ds]
    for a, b in combinations(METRICS, 2):
        x = d[[f"m:{a}", f"m:{b}"]].dropna().astype(float)
        if len(x) > 20 and x.iloc[:, 0].nunique() > 2 and x.iloc[:, 1].nunique() > 2:
            r = spearmanr(x.iloc[:, 0], x.iloc[:, 1])[0]
            if np.isfinite(r):
                _rho.setdefault(frozenset((a, b)), []).append(r)
RHO = {k: float(np.median(v)) for k, v in _rho.items() if len(v) >= 2}


def rho(a, b):
    return RHO.get(frozenset((a, b)), np.nan)


def mean_rho(m, others):
    v = [rho(m, o) for o in others if o != m and np.isfinite(rho(m, o))]
    return float(np.median(v)) if v else np.nan


L = [__doc__.strip(), ""]
w = L.append
f2 = lambda x: "—" if x is None or not np.isfinite(x) else f"{x:.2f}"
VERD = {}
w("## Por métrica\n")
w("| grupo | métrica | detecta | única | única grupo | FP | ρ grupo | mejor fuera (ρ) | max ρ (métrica) | veredicto |")
w("|---|---|---|---|---|---|---|---|---|---|")
for g in sorted(GROUPS):
    for m in sorted(GROUPS[g]):
        others_all = set().union(*(DET[o] for o in METRICS if o != m))
        sib = [o for o in GROUPS[g] if o != m]
        sib_det = set().union(*(DET[o] for o in sib)) if sib else set()
        uniq, uniq_g = DET[m] - others_all, DET[m] - sib_det
        rg = mean_rho(m, sib)
        outs = {h: mean_rho(m, ms) for h, ms in GROUPS.items() if h != g}
        outs = {h: v for h, v in outs.items() if np.isfinite(v)}
        bo = max(outs, key=outs.get) if outs else None
        cand = {o: rho(m, o) for o in METRICS if o != m and np.isfinite(rho(m, o))}
        mo = max(cand, key=lambda o: abs(cand[o])) if cand else None
        fp, ng = FP[m]
        if not DET[m]:
            v = "sin señal"
        elif fp:
            v = "ruidosa"
        elif mo and abs(cand[mo]) >= 0.9 and not uniq:
            v = "redundante"
        elif sib and bo and np.isfinite(rg) and outs[bo] > rg + 0.1:
            v = "mal ubicada"
        else:
            v = "útil"
        VERD[m] = v
        w(f"| {g} | {m.split('.', 1)[1]} | {len(DET[m])} | {len(uniq)} | {len(uniq_g) if sib else '—'} | {fp}/{ng} | "
          f"{f2(rg) if sib else '—'} | {(bo + ' (' + f2(outs[bo]) + ')') if bo else '—'} | "
          f"{(mo.split('.', 1)[1] + ' (' + f2(cand[mo]) + ')') if mo else '—'} | {v} |")

w("\n## Por grupo (primer nivel de agregación)\n")
w("| grupo | métricas | ρ interno | ρ externo | vistos por algún miembro | vistos por la nota del grupo | coherente |")
w("|---|---|---|---|---|---|---|")
for g in sorted(GROUPS):
    ms = GROUPS[g]
    inner = [rho(a, b) for a, b in combinations(ms, 2) if np.isfinite(rho(a, b))]
    outer = [rho(a, o) for a in ms for o in METRICS if o not in ms and np.isfinite(rho(a, o))]
    ri = float(np.median(inner)) if inner else np.nan
    ro = float(np.median(outer)) if outer else np.nan
    seen = set().union(*(DET[m] for m in ms))
    hcol = f"h:{g}"
    note = detected(hcol) if hcol in DF else set()
    coh = "—" if len(ms) < 2 else ("sí" if np.isfinite(ri) and ri >= 0.5 and ri > ro else "no")
    w(f"| {g} | {len(ms)} | {f2(ri)} | {f2(ro)} | {len(seen)} | "
      f"{len(note & seen) if hcol in DF else '—'} | {coh} |")

w("\n## ρ entre los miembros de cada grupo\n")
for g in sorted(GROUPS):
    ms = sorted(GROUPS[g])
    if len(ms) < 2:
        continue
    short = [m.split(".", 1)[1] for m in ms]
    w(f"\n### {g}\n")
    w("| | " + " | ".join(short) + " |")
    w("|---|" + "---|" * len(ms))
    for a, sa in zip(ms, short):
        w(f"| {sa} | " + " | ".join("1" if a == b else f2(rho(a, b)) for b in ms) + " |")

w("\n## Resumen de veredictos\n")
for v in ("útil", "redundante", "mal ubicada", "ruidosa", "sin señal"):
    ms = [m for m in METRICS if VERD[m] == v]
    w(f"- **{v}** ({len(ms)}): " + (", ".join(ms) or "—"))
w(f"\nMétricas sin valores en el banco (no configuradas o no aplicables): "
  + (", ".join(sorted(set(c[2:] for c in DF.columns if c.startswith('m:')) - set(METRICS))) or "ninguna"))
OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
print("escrito", OUT)
