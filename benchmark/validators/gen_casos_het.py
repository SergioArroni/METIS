"""Defectos de heterogeneidad y de validez para el banco de validadores (amplía gen_casos.py).

Uso: gen_casos_het.py <dataset> <dir_casos>

Reconstruye la partición de gen_casos.py (misma semilla; comprueba que A y H coinciden con los
guardados), guarda B y la tabla completa (para calibrar METIS como el benchmark: sobre la tabla
entera, con tercios del tamaño de lo que se evalúa) y añade a cada buena:

  fidelidad  ordinal_orden_roto (los niveles de cada ordinal reetiquetados al azar: el orden se
             pierde), ordinal_desplazada (cada ordinal un nivel arriba), num_cat_rota (el bloque de
             numéricas barajado por filas a la vez: se conservan las relaciones entre numéricas y
             entre categóricas, se pierden las mixtas), num_num_rota (cada numérica barajada por
             separado, categóricas intactas), cat_cat_rota (cada categórica barajada por separado,
             numéricas intactas)
  validez    discreta_fuera_reticula (30 % de cada discreta movido medio paso), nivel_inventado
             (10 % de cada nominal con un nivel que no existe), fuera_de_rango (5 % de cada continua
             a tres veces su máximo)

Las ordinales y discretas solo existen donde el esquema las declara; un defecto sin columnas a
las que aplicarse no se crea.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.argv = [sys.argv[0], sys.argv[1], sys.argv[2], "0"]  # gen_casos con 0 buenas: solo la partición
exec(open(Path(__file__).with_name("gen_casos.py")).read().split("OUT.mkdir")[0])  # noqa: S102

CASES = Path(sys.argv[2]) / DS
assert pd.read_pickle(CASES / "A.pkl").equals(A) and pd.read_pickle(CASES / "H.pkl").equals(H)
B.to_pickle(CASES / "B.pkl")
CALIB.to_pickle(CASES / "full.pkl")  # A + H + |A| filas de B: tercios del tamaño de A
meta = json.load(open(CASES / "meta.json"))

ORD = [c for c, s in schema.items() if typ(s) == "ordinal" and c != TARGET]
NOMINAL = [c for c, s in schema.items() if typ(s) == "nominal" and c != TARGET]
CONT = [c for c, s in schema.items() if typ(s) == "continuous" and c != TARGET]
DISCRETE = [c for c, s in schema.items() if typ(s) == "discrete" and c != TARGET]


def levels(c):
    return [str(v) for v in schema[c]["levels"]]


def half_step(c):
    s = schema[c]
    if s.get("step"):
        return s["step"] / 2
    lv = sorted(float(v) for v in s["levels"])
    return min(np.diff(lv)) / 2 if len(lv) > 1 else 0.5


def het_defects(good, rng):
    d = {}
    if ORD:
        x = good.copy()
        for c in ORD:
            lv = levels(c)
            perm = dict(zip(lv, rng.permutation(lv)))
            x[c] = x[c].astype(str).map(lambda v, p=perm: p.get(v, v))
        d["ordinal_orden_roto"] = x
        x = good.copy()
        for c in ORD:
            lv = levels(c)
            pos = {v: i for i, v in enumerate(lv)}
            x[c] = x[c].astype(str).map(lambda v, lv=lv, pos=pos: lv[min(pos[v] + 1, len(lv) - 1)] if v in pos else v)
        d["ordinal_desplazada"] = x
    if NUM and CAT:
        x = good.copy()
        perm = rng.permutation(len(x))
        x[NUM] = x[NUM].iloc[perm].to_numpy()
        d["num_cat_rota"] = x
    if len(NUM) > 1:
        x = good.copy()
        for c in NUM:
            x[c] = rng.permutation(x[c].to_numpy())
        d["num_num_rota"] = x
    if len(CAT) > 1:
        x = good.copy()
        for c in CAT:
            x[c] = rng.permutation(x[c].to_numpy())
        d["cat_cat_rota"] = x
    if DISCRETE:
        x = good.copy()
        for c in DISCRETE:
            v = _num(x, c)
            m = rng.random(len(x)) < 0.3
            x[c] = np.where(m, v + half_step(c), v)
        d["discreta_fuera_reticula"] = x
    if NOMINAL:
        x = good.copy()
        for c in NOMINAL:
            m = rng.random(len(x)) < 0.1
            x[c] = x[c].astype(object)
            x.loc[m, c] = "__inventado__"
        d["nivel_inventado"] = x
    if CONT:
        x = good.copy()
        for c in CONT:
            v = _num(x, c)
            m = rng.random(len(x)) < 0.05
            x[c] = np.where(m, 3 * v.abs().max() + 1, v)
        d["fuera_de_rango"] = x
    return d


VALIDITY = {"discreta_fuera_reticula", "nivel_inventado", "fuera_de_rango"}
n_good = meta["n_good"]
for k in range(n_good):
    rng = np.random.default_rng(1000 + k)
    good = pd.read_pickle(CASES / f"k{k}__bueno.pkl")
    for name, df in het_defects(good, rng).items():
        assert list(df.columns) == list(A.columns), name
        df.to_pickle(CASES / f"k{k}__{name}.pkl")
        meta["family"][name] = "validez" if name in VALIDITY else "fidelity"
json.dump(meta, open(CASES / "meta.json", "w"), indent=1, default=str)
print(DS, "ordinales", len(ORD), "discretas", len(DISCRETE), "nominales", len(NOMINAL), "continuas", len(CONT),
      "->", sorted(k for k, v in meta["family"].items() if v in ("validez",) or k in
                   ("ordinal_orden_roto", "ordinal_desplazada", "num_cat_rota", "num_num_rota", "cat_cat_rota")))
