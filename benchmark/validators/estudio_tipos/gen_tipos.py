"""Defectos que solo rompen lo que declara el tipo de una columna (estudio de los tipos, 17 nov).

Uso: gen_tipos.py <dataset> <dir_casos2> <dir_salida> [cod]

Reconstruye la partición de gen_casos.py y escribe en <dir_salida>/<dataset>[_cod] las cinco buenas y, sobre cada una:

  ord_d<d>      el mismo 20 % de filas de cada ordinal pasa a un nivel a distancia d (las mismas filas y el mismo
                sentido en todas las distancias, salvo que el sentido se salga del rango)
  disc_pares    cada discreta se redondea hacia abajo a un múltiplo par de su paso (sigue en la retícula)
  disc_suave    cada discreta recibe -1, 0 o +1 pasos al azar, recortada a su rango (sigue en la retícula)

Con «cod», las nominales de texto con entre 3 y 50 niveles se recodifican con enteros en orden alfabético, como hace
un codificador de etiquetas, y los defectos son

  nom_vecino_20  20 % de las filas de cada nominal codificada pasa al código vecino
  nom_lejano_20  el mismo 20 % pasa al código más lejano a distancia >= K/2

Criterio fijado antes de ejecutar: los tipos influyen si la caída de la fidelidad con el esquema completo es al
menos el doble que con otro esquema, con el mismo signo en las cinco muestras, en dos o más pares defecto-tabla,
sin que aumente el ruido de las buenas; o, en la tabla recodificada, si el esquema de formato da caídas distintas
para vecino y lejano (razón >= 2 en las cinco muestras) y el completo no.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_DS, _CASOS2, _OUT = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
COD = len(sys.argv) > 4 and sys.argv[4] == "cod"
sys.argv = [sys.argv[0], _DS, str(_CASOS2), "0"]
exec(open("/home/ubuntu/strat/validadores/gen_casos.py").read().split("OUT.mkdir")[0])  # noqa: S102

SRC = _CASOS2 / DS
assert pd.read_pickle(SRC / "A.pkl").equals(A) and pd.read_pickle(SRC / "H.pkl").equals(H)
meta = json.load(open(SRC / "meta.json"))
NAME = f"{DS}_cod" if COD else DS
OUT = _OUT / NAME
OUT.mkdir(parents=True, exist_ok=True)
ORD = [c for c, s in schema.items() if typ(s) == "ordinal" and c != TARGET]
DIS = [c for c, s in schema.items() if typ(s) == "discrete" and c != TARGET]
NOMC = []
if COD:
    NOMC = [c for c, s in schema.items() if typ(s) == "nominal" and c != TARGET and A[c].dtype == object
            and 3 <= pd.concat([A[c], H[c], B[c]]).nunique() <= 50]
    for c in NOMC:
        codes = {v: i for i, v in enumerate(sorted(pd.concat([A[c], H[c], B[c]]).dropna().astype(str).unique()))}
        for df in (A, H, B):
            df[c] = df[c].astype(str).map(codes).astype("Int64").astype(float if df[c].isna().any() else int)
    CALIB = pd.concat([A, H, B], ignore_index=True)
    CALIB.to_pickle(OUT / "full.pkl")
else:
    assert pd.read_pickle(SRC / "full.pkl").equals(CALIB)
    CALIB.to_pickle(OUT / "full.pkl")
A.to_pickle(OUT / "A.pkl")
H.to_pickle(OUT / "H.pkl")


def move(good, cols, levels_of, seed, d=None, far=False):
    x = good.copy()
    for j, c in enumerate(cols):
        lv = levels_of(c)
        k = len(lv)
        idx = {str(v): i for i, v in enumerate(lv)}
        pos = x[c].astype(str).map(idx)
        rng = np.random.default_rng(seed + 1000 * j)
        mask = (rng.random(len(x)) < 0.2) & pos.notna().to_numpy()
        sign = rng.choice([-1, 1], len(x))
        rows = np.flatnonzero(mask)
        new = []
        for r in rows:
            i = int(pos.iloc[r])
            if far:
                cand = [t for t in range(k) if abs(t - i) >= k / 2] or [0 if i > (k - 1) / 2 else k - 1]
                t = max(cand, key=lambda t: abs(t - i))
            else:
                t = i + sign[r] * d
                if not 0 <= t < k:
                    t = i - sign[r] * d
                assert 0 <= t < k, (c, i, d, k)
            new.append(lv[t])
        col = x[c].astype(object)
        col.iloc[rows] = new
        x[c] = col.infer_objects()
        if x[c].dtype != good[c].dtype:
            try:
                x[c] = x[c].astype(good[c].dtype)
            except (TypeError, ValueError):
                pass
    return x


def disc(good, mode, seed):
    x = good.copy()
    rng = np.random.default_rng(seed)
    for c in DIS:
        s = schema[c]
        step = float(s.get("step") or 1)
        v = _num(x, c)
        if mode == "pares":
            new = 2 * step * np.floor(v / (2 * step))
        else:
            a = _num(A, c)
            new = (v + step * rng.integers(-1, 2, len(x))).clip(a.min(), a.max())
        if s.get("levels"):
            ok = new.isin([float(t) for t in s["levels"]])
            new = new.where(ok, v)
        new = new.where(v.notna(), v)
        x[c] = new.astype(good[c].dtype) if not new.isna().any() else new
    return x


K = max((len(schema[c]["levels"]) for c in ORD), default=0)
DMAX = min(-(-(len(schema[c]["levels"]) - 1) // 2) for c in ORD) if ORD else 0
DIST = [d for d in (1, 2, 4, 8) if d <= DMAX]
family = {"bueno": None}
for k in range(5):
    good = muestra_buena(B, len(A), k).reset_index(drop=True)
    if not COD:
        assert pd.read_pickle(SRC / f"k{k}__bueno.pkl").equals(good), k
    cases = {"bueno": good}
    seed = 5000 + 100 * k
    if COD:
        lev = lambda c: sorted(pd.concat([A[c], H[c], B[c]]).dropna().unique())  # noqa: E731
        cases["nom_vecino_20"] = move(good, NOMC, lev, seed, d=1)
        cases["nom_lejano_20"] = move(good, NOMC, lev, seed, far=True)
    else:
        for d in DIST:
            cases[f"ord_d{d}"] = move(good, ORD, lambda c: schema[c]["levels"], seed, d=d)
        if DIS:
            cases["disc_pares"] = disc(good, "pares", seed + 1)
            cases["disc_suave"] = disc(good, "suave", seed + 2)
    for name, df in cases.items():
        assert list(df.columns) == list(A.columns), name
        if name != "bueno":
            ch = {c: round(float((df[c].astype(str) != good[c].astype(str)).mean()), 3) for c in df.columns}
            ch = {c: v for c, v in ch.items() if v > 0}
            if k == 0:
                print(NAME, name, "columnas cambiadas", ch, flush=True)
        df.to_pickle(OUT / f"k{k}__{name}.pkl")
        family[name] = None if name == "bueno" else "fidelity"
meta.update({"dataset": NAME, "config_base": "rrhh" if DS == "adult" else DS, "family": family, "n_good": 5,
             "codificadas": NOMC, "distancias": DIST})
json.dump(meta, open(OUT / "meta.json", "w"), indent=1, default=str)
print(NAME, "ordinales", ORD, "K máx", K, "distancias", DIST, "discretas", len(DIS), "codificadas", NOMC, len(family), "casos por muestra")
