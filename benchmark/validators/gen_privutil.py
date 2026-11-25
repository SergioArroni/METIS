"""Defectos de privacidad y utilidad más variados que copiar filas, para medir la coherencia de sus métricas.

Uso: gen_privutil.py <dir_casos2>/<dataset> <dir_salida>

Reutiliza A, H, full y las buenas del banco 2 (enlaces) y copia sus defectos de privacidad y utilidad
(copias exactas, casi copias, objetivo barajado, etiquetas cambiadas).  Añade sobre cada buena:

  Privacidad
    raras_copiadas_2     un 2 % de las filas sustituido por las filas de A con la combinación de categóricas
                         más rara (las más expuestas a reidentificación)
    casi_copias_002 / 010 / 030   un 10 % de las filas sustituido por filas de A con las numéricas movidas
                         0,02, 0,10 o 0,30 desviaciones típicas (categóricas intactas)
    cuasi_copiadas_10    un 10 % de las filas toma de filas de A todas las categóricas y el objetivo; las
                         numéricas se quedan (copia parcial: cuasi-identificadores y atributo sensible)
  Utilidad
    predictora_barajada  la variable más informativa sobre el objetivo en A (información mutua) se baraja
    tres_predictoras_barajadas   ídem con las tres más informativas
    objetivo_por_regla   el objetivo se rehace con un umbral sobre la variable más informativa, con la misma
                         prevalencia: señal fuerte, pero no la de la tabla real
    minoritaria_reducida el 80 % de las filas de la clase minoritaria pasa a filas de la mayoritaria
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
meta = json.load(open(SRC / "meta.json"))
DS, T, schema = meta["dataset"], meta["target"], meta["schema"]
OUT = OUT / DS
OUT.mkdir(parents=True, exist_ok=True)
A = pd.read_pickle(SRC / "A.pkl")


def typ(spec):
    return spec if isinstance(spec, str) else spec.get("type")


CAT = [c for c, s in schema.items() if typ(s) in ("nominal", "ordinal") and c != T and c in A.columns]
NUM = [c for c, s in schema.items() if typ(s) in ("continuous", "discrete") and c != T and c in A.columns]
DISC = {c for c, s in schema.items() if typ(s) == "discrete"}

# rareza de cada fila de A: producto de las frecuencias de sus categóricas
logf = sum(np.log(A[c].astype(str).map(A[c].astype(str).value_counts(normalize=True))) for c in CAT) if CAT else pd.Series(0, index=A.index)
RARE_ROWS = A.loc[np.argsort(logf.to_numpy())].reset_index(drop=True)
# información mutua de cada variable con el objetivo en A
X = pd.DataFrame({c: (pd.to_numeric(A[c], errors="coerce").fillna(pd.to_numeric(A[c], errors="coerce").median())
                      if c in NUM else A[c].astype(str).astype("category").cat.codes) for c in NUM + CAT})
mi = mutual_info_classif(X, A[T].astype(str), discrete_features=[c in CAT for c in X.columns], random_state=0)
TOP = list(pd.Series(mi, index=X.columns).sort_values(ascending=False).index[:3])
y_counts = A[T].astype(str).value_counts()
MINOR, MAJOR = y_counts.index[-1], y_counts.index[0]


def replace_rows(good, src, frac, rng):
    x = good.copy()
    k = int(round(frac * len(x)))
    rows = rng.choice(len(x), k, replace=False)
    for c in x.columns:
        x[c] = x[c].astype(object)
        x.loc[rows, c] = src[c].iloc[:k].astype(object).to_numpy()
    return x.infer_objects()


def defects(good, rng):
    d = {}
    d["raras_copiadas_2"] = replace_rows(good, RARE_ROWS, 0.02, rng)
    for name, s in (("casi_copias_002", 0.02), ("casi_copias_010", 0.10), ("casi_copias_030", 0.30)):
        src = A.sample(len(good), replace=False, random_state=int(rng.integers(1e9))).reset_index(drop=True)
        for c in NUM:
            v = pd.to_numeric(src[c], errors="coerce")
            sd = pd.to_numeric(A[c], errors="coerce").std()
            w = v + rng.normal(0, s * (sd if np.isfinite(sd) and sd > 0 else 1.0), len(v))
            src[c] = np.round(w) if c in DISC else w
        d[name] = replace_rows(good, src, 0.10, rng)
    x = good.copy()
    k = int(round(0.10 * len(x)))
    rows = rng.choice(len(x), k, replace=False)
    src = A.sample(k, replace=False, random_state=int(rng.integers(1e9))).reset_index(drop=True)
    for c in CAT + [T]:
        x[c] = x[c].astype(object)
        x.loc[rows, c] = src[c].astype(object).to_numpy()
    d["cuasi_copiadas_10"] = x.infer_objects()
    for name, cols in (("predictora_barajada", TOP[:1]), ("tres_predictoras_barajadas", TOP[:3])):
        x = good.copy()
        for c in cols:
            x[c] = rng.permutation(x[c].to_numpy())
        d[name] = x
    x = good.copy()
    c = TOP[0]
    score = (pd.to_numeric(x[c], errors="coerce") if c in NUM else x[c].astype(str).map(
        A.groupby(A[c].astype(str))[T].apply(lambda s: (s.astype(str) == MINOR).mean()))).fillna(0)
    n_minor = int((x[T].astype(str) == MINOR).sum())
    order = np.argsort(-score.to_numpy(), kind="stable")
    y = np.array([MAJOR] * len(x), dtype=object)
    y[order[:n_minor]] = MINOR
    x[T] = pd.Series(y).astype(good[T].dtype, errors="ignore").to_numpy() if good[T].dtype != object else y
    d["objetivo_por_regla"] = x
    x = good.copy()
    minor_rows = np.flatnonzero((x[T].astype(str) == MINOR).to_numpy())
    major_rows = np.flatnonzero((x[T].astype(str) == MAJOR).to_numpy())
    drop = rng.choice(minor_rows, int(round(0.8 * len(minor_rows))), replace=False)
    fill = rng.choice(major_rows, len(drop), replace=True)
    x.iloc[drop] = x.iloc[fill].to_numpy()
    d["minoritaria_reducida"] = x
    return d


NEW = {"raras_copiadas_2": "privacy", "casi_copias_002": "privacy", "casi_copias_010": "privacy",
       "casi_copias_030": "privacy", "cuasi_copiadas_10": "privacy", "predictora_barajada": "utility",
       "tres_predictoras_barajadas": "utility", "objetivo_por_regla": "utility", "minoritaria_reducida": "utility"}
family = {"bueno": None}
for name, fam in meta["family"].items():
    if fam in ("privacy", "utility"):
        family[name] = fam
for k in range(meta["n_good"]):
    good = pd.read_pickle(SRC / f"k{k}__bueno.pkl")
    for name in family:
        dst = OUT / f"k{k}__{name}.pkl"
        if not dst.exists():
            os.symlink(SRC / f"k{k}__{name}.pkl", dst)
    for name, df in defects(good, np.random.default_rng(2000 + k)).items():
        assert list(df.columns) == list(good.columns), name
        df.to_pickle(OUT / f"k{k}__{name}.pkl")
family.update(NEW)
for f in ("A.pkl", "H.pkl", "full.pkl"):
    if not (OUT / f).exists():
        os.symlink(SRC / f, OUT / f)
json.dump({**meta, "family": family, "privutil": {"top": TOP, "minoritaria": MINOR}},
          open(OUT / "meta.json", "w"), indent=1, default=str)
print(DS, "top", TOP, "minoritaria", MINOR, len(family), "casos por muestra")
