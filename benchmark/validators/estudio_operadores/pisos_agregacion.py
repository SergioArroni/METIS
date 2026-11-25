"""¿Qué regla en los pisos 1 (métricas → subcategoría) y 2 (subcategorías → categoría)?

Uso: pisos_agregacion.py <dir con los jsonl del estudio de ordinales>

Reagrega la familia de fidelidad desde el valor de cada métrica (guardado en el estudio) con
mediana, media o SSD en cada piso (el piso 3, categorías → familia, sigue SSD).  Control:
mediana/mediana reproduce la nota de METIS.  Para cada combinación: proporción de defectos que
la nota de fidelidad detecta (z ≥ 3), cuántos no (z < 2) y el ruido de las sintéticas buenas.
casi_copia queda fuera: es un defecto de privacidad, no de fidelidad.
"""

import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from metis.application.aggregator import get_aggregation_function
from metis.domain.taxonomy import get_metric_hierarchy

ssd = get_aggregation_function("ssd", risk_aversion=5.0)
RULES = {"mediana": lambda v: float(np.median(v)), "media": lambda v: float(np.mean(v)),
         "ssd": lambda v: float(ssd(list(v)))}

rows = []
for f in sorted(Path(sys.argv[1]).glob("*.jsonl")):
    rows += [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x]

tree = {}
for r in rows:
    for mid in r["metricas"]:
        if mid.startswith("fidelity.") and mid not in tree:
            h = get_metric_hierarchy(mid)
            tree[mid] = (h["category"], h["subcategory"] or "_direct")


def familia(metricas, l1, l2):
    groups = defaultdict(lambda: defaultdict(list))
    for mid, v in metricas.items():
        if mid in tree and v is not None and np.isfinite(v):
            cat, sub = tree[mid]
            groups[cat][sub].append(v)
    cats = []
    for cat, subs in groups.items():
        sub_scores = [RULES[l1](v) for v in subs.values()]
        cats.append(RULES[l2](sub_scores))
    return float(ssd(cats))


combos = list(itertools.product(RULES, RULES))
for l1, l2 in combos:
    key = f"{l1}/{l2}"
    for r in rows:
        r[key] = familia(r["metricas"], l1, l2)
df = pd.DataFrame([{k: v for k, v in r.items() if k != "metricas"} for r in rows])
print("control mediana/mediana vs METIS: |dif| máx =", float((df["mediana/mediana"] - df["fidelity"]).abs().max()))

out = []
for (ds, tr), g in df.groupby(["dataset", "tratamiento"]):
    b = g[g.defecto == "bueno"]
    for de, gd in g[(g.defecto != "bueno") & (g.defecto != "casi_copia")].groupby("defecto"):
        rec = {"dataset": ds, "tratamiento": tr, "defecto": de}
        for l1, l2 in combos:
            k = f"{l1}/{l2}"
            sd = b[k].std(ddof=1)
            rec[k] = (b[k].mean() - gd[k].mean()) / sd if sd > 1e-12 else np.nan
        out.append(rec)
res = pd.DataFrame(out)
keys = [f"{a}/{b}" for a, b in combos]
noise = df[df.defecto == "bueno"].groupby(["dataset", "tratamiento"])[keys].std().median()
summary = pd.DataFrame({
    "detectados_z>=3": (res[keys] >= 3).mean().round(3),
    "no_detectados_z<2": (res[keys] < 2).sum(),
    "z_mediano": res[keys].median().round(1),
    "ruido_buenas_sd": noise.round(4),
})
pd.set_option("display.width", 200, "display.max_rows", 200)
print(f"\ncasos: {len(res)} (3 datasets x 10 tratamientos x 8 defectos de fidelidad)")
print("piso1/piso2 (piso 3 = SSD siempre)")
print(summary)

print("\n=== tratamiento actual (doble_vista): z de la familia por defecto ===")
dv = res[res.tratamiento == "doble_vista"].set_index(["defecto", "dataset"])
print(dv[["mediana/mediana", "mediana/media", "media/media", "media/ssd", "ssd/ssd"]].round(1).sort_index())

print("\n=== ¿algún defecto empeora mucho al cambiar de mediana/mediana? (media/media − actual < −3) ===")
d = res.assign(cambio=res["media/media"] - res["mediana/mediana"])
print(d[d.cambio < -3][["dataset", "tratamiento", "defecto", "mediana/mediana", "media/media"]].round(1).to_string())
res.to_csv(Path(sys.argv[1]) / "pisos.csv", index=False)
