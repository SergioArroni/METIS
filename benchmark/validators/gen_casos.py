"""Banco de defectos para comparar validadores (METIS, SDMetrics, SynthEval, synthcity).

Uso: gen_casos.py <dataset> <dir_salida> [n_buenos]

La tabla real (hasta 8000 filas) se parte con semilla 0 en cuartos: A, la «real» que vio el
generador (1/4); H, holdout (1/4); B, reserva (1/2).  Las buenas son submuestras de B del tamaño de
A, en bruto, sin reemplazo: filas reales sin duplicar y ninguna de A (con reemplazo, el 36 % de las
filas salían repetidas y eso bajaba la utilidad de una tabla perfecta, 29 oct).  La calibración de
METIS usa A + H + |A| filas de B, cuyos tercios tienen el tamaño de A.  A cada buena se le aplica
cada defecto, y cada defecto tiene la familia que debería bajar:

  fidelidad   desplazar_numericas (+0,5 sd), escala_numericas (dispersión ×1,5), colas_cortadas
              (numéricas recortadas a sus percentiles 10–90), categorias_aplanadas (cada categórica
              uniforme sobre sus niveles), relaciones_rotas (cada columna barajada por separado:
              marginales intactas, ninguna dependencia), mitad_rota (la mitad de las columnas
              barajadas), ruido_uniforme (cada columna uniforme en su rango o sus niveles)
  utilidad    objetivo_barajado (solo el objetivo barajado: su marginal intacta, su relación con el
              resto perdida), etiquetas_ruidosas (30 % del objetivo sustituido al azar)
  privacidad  copias_10, copias_25 (esa fracción de filas sustituida por copias exactas de A),
              copias_100 (A entera, reordenada), casi_copias_25 (copias con
              numéricas + 5 % de su sd y cada categórica cambiada con probabilidad 0,1)

Escribe <dir_salida>/<dataset>/{A,H}.pkl, meta.json y k<k>__<defecto>.pkl.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

DS, OUT = sys.argv[1], Path(sys.argv[2]) / sys.argv[1]
N_GOOD = int(sys.argv[3]) if len(sys.argv) > 3 else 5
DATA = Path("/home/ubuntu/metis-restricciones/data/real")
CONFIGS = Path("/home/ubuntu/doc-split/metis/metis/configs")
MAX_ROWS = 8000
ADULT_EDU = ["Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", "11th", "12th", "HS-grad",
             "Some-college", "Assoc-voc", "Assoc-acdm", "Bachelors", "Masters", "Prof-school", "Doctorate"]
TARGET = {"rrhh": "Attrition", "cardio": "cardio", "telco": "Churn", "adult": "Target"}[DS]

if DS == "adult":
    from metis.infrastructure.typing import infer_schema

    real = pd.read_csv(DATA / "adult_test.csv", skiprows=[1], skipinitialspace=True)
    real = real.drop(columns=["Education_Num"])
    for c in real.columns:
        if real[c].dtype == object:
            real[c] = real[c].str.strip().str.rstrip(".")
    schema = dict(infer_schema(real).schema)
    schema["Education"] = {"type": "ordinal", "levels": ADULT_EDU}
else:
    CSV = {"rrhh": "rrhh.csv", "cardio": "cardio_train.csv", "telco": "telco.csv"}
    schema = yaml.safe_load(open(CONFIGS / f"config_{DS}.yaml", encoding="utf-8"))["data"]["schema"]
    real = pd.read_csv(DATA / CSV[DS], sep=None, engine="python", encoding="utf-8-sig")
schema = {c: s for c, s in schema.items() if c in real.columns}
real = real[list(schema)]
if len(real) > MAX_ROWS:
    real = real.sample(MAX_ROWS, random_state=0).reset_index(drop=True)
from sklearn.model_selection import train_test_split


def _strat(df):
    lab = df[TARGET].astype(str)
    return lab if lab.nunique() > 1 and lab.value_counts().min() >= 3 else None


def muestra_buena(B, n, k):
    """n filas de B estratificadas por el objetivo, como la parte no vista del benchmark (11 nov)."""
    idx, _ = train_test_split(B.index, train_size=n, random_state=k, stratify=_strat(B))
    return B.loc[idx]


# Cuartos estratificados por el objetivo con el reparto del benchmark (11 nov): A y H de q filas, B el resto.
q = len(real) // 4
A, rest = train_test_split(real, train_size=q, random_state=0, stratify=_strat(real))
H, B = train_test_split(rest, train_size=q, random_state=0, stratify=_strat(rest))
A, H, B = (x.reset_index(drop=True) for x in (A, H, B))
# METIS calibra sobre la tabla entera con partes de |A| filas en tercios (calibration.split), el tamaño de lo
# que se evalúa: entrenamiento A, holdout H y sintética de |A| filas.
CALIB = pd.concat([A, H, B], ignore_index=True)


def typ(spec):
    return spec if isinstance(spec, str) else spec.get("type")


NUM = [c for c, s in schema.items() if typ(s) in ("continuous", "discrete") and c != TARGET]
DISC = {c for c, s in schema.items() if typ(s) == "discrete"}
CAT = [c for c, s in schema.items() if typ(s) in ("nominal", "ordinal") and c != TARGET]


def _num(df, c):
    return pd.to_numeric(df[c], errors="coerce")


def _round(c, v):
    return np.round(v) if c in DISC else v


def with_copies(good, frac, rng, near=False):
    x = good.copy()
    k = int(round(frac * len(x)))
    rows = rng.choice(len(x), k, replace=False)
    src = A.sample(k, replace=False, random_state=int(rng.integers(1e9))).reset_index(drop=True)
    if near:
        for c in NUM:
            sd = _num(A, c).std()
            v = _num(src, c) + rng.normal(0, 0.05 * (sd if np.isfinite(sd) and sd > 0 else 1.0), k)
            src[c] = _round(c, v)
        for c in CAT:
            change = rng.random(k) < 0.1
            if change.any():
                src.loc[change, c] = rng.choice(A[c].dropna().to_numpy(), int(change.sum()))
    for c in x.columns:
        x[c] = x[c].astype(object)
        x.loc[rows, c] = src[c].astype(object).to_numpy()
    return x.infer_objects()


def defects(good, rng):
    d = {}
    x = good.copy()
    for c in NUM:
        v = _num(x, c)
        x[c] = _round(c, v + 0.5 * v.std())
    d["desplazar_numericas"] = x
    x = good.copy()
    for c in NUM:
        v = _num(x, c)
        x[c] = _round(c, v.mean() + 1.5 * (v - v.mean()))
    d["escala_numericas"] = x
    x = good.copy()
    for c in NUM:
        v = _num(x, c)
        x[c] = v.clip(v.quantile(0.10), v.quantile(0.90))
    d["colas_cortadas"] = x
    x = good.copy()
    for c in CAT:
        x[c] = rng.choice(x[c].dropna().unique(), len(x))
    d["categorias_aplanadas"] = x
    x = good.copy()
    for c in x.columns:
        x[c] = rng.permutation(x[c].to_numpy())
    d["relaciones_rotas"] = x
    x = good.copy()
    for c in rng.choice(list(x.columns), len(x.columns) // 2, replace=False):
        x[c] = rng.permutation(x[c].to_numpy())
    d["mitad_rota"] = x
    x = good.copy()
    for c in x.columns:
        if c in NUM:
            v = _num(A, c)
            x[c] = _round(c, rng.uniform(v.min(), v.max(), len(x)))
        else:
            x[c] = rng.choice(A[c].dropna().unique(), len(x))
    d["ruido_uniforme"] = x
    x = good.copy()
    x[TARGET] = rng.permutation(x[TARGET].to_numpy())
    d["objetivo_barajado"] = x
    x = good.copy()
    m = rng.random(len(x)) < 0.3
    x.loc[m, TARGET] = rng.choice(A[TARGET].dropna().unique(), int(m.sum()))
    d["etiquetas_ruidosas"] = x
    d["copias_10"] = with_copies(good, 0.10, rng)
    d["copias_25"] = with_copies(good, 0.25, rng)
    d["copias_100"] = A.sample(frac=1, random_state=int(rng.integers(1e9))).reset_index(drop=True)
    d["casi_copias_25"] = with_copies(good, 0.25, rng, near=True)
    return d


FAMILY = {"bueno": None, "desplazar_numericas": "fidelity", "escala_numericas": "fidelity",
          "colas_cortadas": "fidelity", "categorias_aplanadas": "fidelity", "relaciones_rotas": "fidelity",
          "mitad_rota": "fidelity", "ruido_uniforme": "fidelity", "objetivo_barajado": "utility",
          "etiquetas_ruidosas": "utility", "copias_10": "privacy", "copias_25": "privacy",
          "copias_100": "privacy", "casi_copias_25": "privacy"}

OUT.mkdir(parents=True, exist_ok=True)
A.to_pickle(OUT / "A.pkl")
H.to_pickle(OUT / "H.pkl")
cat_cols = [c for c, s in schema.items() if typ(s) in ("nominal", "ordinal")]
json.dump({"dataset": DS, "schema": schema, "target": TARGET, "cat_cols": cat_cols, "family": FAMILY,
           "n_good": N_GOOD, "rows": len(A)}, open(OUT / "meta.json", "w"), indent=1, default=str)
for k in range(N_GOOD):
    rng = np.random.default_rng(100 + k)
    good = muestra_buena(B, len(A), k).reset_index(drop=True)
    for name, df in {"bueno": good, **defects(good, rng)}.items():
        assert list(df.columns) == list(A.columns), name
        df.to_pickle(OUT / f"k{k}__{name}.pkl")
print(DS, len(A), "filas por cuarto,", N_GOOD * len(FAMILY), "casos")
