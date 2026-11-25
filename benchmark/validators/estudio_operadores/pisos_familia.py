"""Agregación por pisos con media, mediana, SSD y FSD (FSD ≡ media en METIS: control).

Uso: pisos_familia.py <dir jsonl> <familia> <defectos_excluidos,...> [piso1/piso2 fijos]

Reagrega la familia desde el valor de cada métrica guardado en los estudios.  Para fidelidad se
fijan los pisos 1 y 2 en media (decisión del 2026-10-20) y se prueba el piso 3; para privacidad se
prueban todas las combinaciones de los pisos 1 y 2 (el 3 tiene una sola categoría aplicable).
Por defecto: z = (media buenas − media defecto) / sd(buenas); detectado si z ≥ 3.
"""

import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from metis.domain.taxonomy import get_metric_hierarchy
from metis.shared.aggregation_registry import get_aggregation_function

d, FAM = Path(sys.argv[1]), sys.argv[2]
EXCL = set(filter(None, sys.argv[3].split(","))) if len(sys.argv) > 3 else set()
FIXED = sys.argv[4].split("/") if len(sys.argv) > 4 else None
RULES = {n: get_aggregation_function(n, **({"risk_aversion": 3.0} if n == "ssd" else {}))
         for n in ("mean", "median", "ssd")}

rows = []
for f in sorted(d.glob("*.jsonl")):
    rows += [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x]
tree = {}
for r in rows:
    for mid in r["metricas"]:
        if mid.startswith(FAM + ".") and mid not in tree:
            h = get_metric_hierarchy(mid)
            tree[mid] = (h["category"], h["subcategory"] or "_direct")


def agg(metricas, l1, l2, l3):
    g = defaultdict(lambda: defaultdict(list))
    for mid, v in metricas.items():
        if mid in tree and v is not None and np.isfinite(v):
            c, s = tree[mid]
            g[c][s].append(v)
    if not g:
        return np.nan
    cats = [float(RULES[l2]([float(RULES[l1](v)) for v in subs.values()])) for subs in g.values()]
    return float(RULES[l3](cats))


if FIXED:
    combos = [(FIXED[0], FIXED[1], l3) for l3 in RULES]
else:
    combos = [(a, b, "ssd") for a, b in itertools.product(RULES, RULES)]
keys = ["/".join(c) for c in combos]
for r in rows:
    for c, k in zip(combos, keys, strict=True):
        r[k] = agg(r["metricas"], *c)
df = pd.DataFrame([{k: v for k, v in r.items() if k != "metricas"} for r in rows])
ref = f"{FAM}_score" if f"{FAM}_score" in df else FAM
if "median/median/ssd" in keys and FAM in df:
    print("control mediana/mediana/ssd vs METIS:", float((df["median/median/ssd"] - df[FAM]).abs().max()))
if FIXED and "mean/mean/ssd" in keys and FAM in df:
    print("control media/media/ssd vs METIS (v2):", float((df["mean/mean/ssd"] - df[FAM]).abs().max()))

out = []
grp = ["dataset", "tratamiento"] if "tratamiento" in df else ["dataset"]
for key_, g in df.groupby(grp):
    b = g[g.defecto == "bueno"]
    for de, gd in g[(g.defecto != "bueno") & ~g.defecto.isin(EXCL)].groupby("defecto"):
        rec = dict(zip(grp, key_ if isinstance(key_, tuple) else (key_,), strict=True)) | {"defecto": de}
        for k in keys:
            sd = b[k].std(ddof=1)
            rec[k] = (b[k].mean() - gd[k].mean()) / sd if sd > 1e-12 else np.nan
            rec[k + "|delta"] = b[k].mean() - gd[k].mean()
        out.append(rec)
res = pd.DataFrame(out)
pd.set_option("display.width", 250, "display.max_rows", 300, "display.max_columns", 40)
noise = df[df.defecto == "bueno"].groupby(grp)[keys].std().median()
level = df[df.defecto == "bueno"][keys].mean()
print(f"\ncasos: {len(res)}")
print(pd.DataFrame({
    "detectados_z>=3": (res[keys] >= 3).mean().round(3),
    "no_detectados_z<2": (res[keys] < 2).sum(),
    "z_mediano": res[keys].median().round(1),
    "ruido_buenas_sd": noise.round(4),
    "nota_media_buenas": level.round(3),
}))
res.to_csv(d / f"pisos_{FAM}.csv", index=False)
