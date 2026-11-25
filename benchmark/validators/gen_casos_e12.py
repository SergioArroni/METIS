"""Casos de E1 (heterogeneidad) y E2 (dosis) para el banco de validadores.

Uso: gen_casos_e12.py <dataset> <dir_casos>

Reconstruye la partición de gen_casos.py y añade a cada buena (familia que debería bajar):

E1, distancia ordinal con la misma masa movida (fidelidad):
  ordinal_cerca_20  20 % de las filas de cada ordinal pasa al nivel vecino
  ordinal_lejos_20  el mismo 20 % pasa al nivel más lejano a distancia ≥ K/2
  Una herramienta que trata la ordinal como categoría ve casi lo mismo en los dos; con orden, el
  segundo es mucho peor.

E2, dosis (la mínima que cada validador detecta):
  desplazar_005/010/025     numéricas + 0,05 / 0,10 / 0,25 sd                    (fidelidad)
  colas_extremas            numéricas recortadas a sus percentiles 1–99            (fidelidad)
  categorias_raras_perdidas los niveles con < 5 % de frecuencia en A pasan a la moda (fidelidad)
  ausentes_mcar             los huecos de cada columna redistribuidos al azar con la misma tasa:
                            marginales intactas, patrón conjunto roto (solo si A tiene huecos
                            en al menos dos columnas)                              (fidelidad)
  copias_1/2/5              esa fracción de filas sustituida por copias exactas de A (privacidad)
  etiquetas_ruidosas_10/50  ese porcentaje del objetivo sustituido al azar          (utilidad)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.argv = [sys.argv[0], sys.argv[1], sys.argv[2], "0"]
exec(open(Path(__file__).with_name("gen_casos.py")).read().split("OUT.mkdir")[0])  # noqa: S102

CASES = Path(sys.argv[2]) / DS
assert pd.read_pickle(CASES / "A.pkl").equals(A) and pd.read_pickle(CASES / "H.pkl").equals(H)
meta = json.load(open(CASES / "meta.json"))
ORD = [c for c, s in schema.items() if typ(s) == "ordinal" and c != TARGET]


def levels(c):
    return [str(v) for v in schema[c]["levels"]]


def ordinal_move(good, rng, far):
    x = good.copy()
    for c in ORD:
        lv = levels(c)
        k = len(lv)
        pos = x[c].astype(str).map({v: i for i, v in enumerate(lv)})
        rows = np.flatnonzero((rng.random(len(x)) < 0.2) & pos.notna().to_numpy())
        new = []
        for i in pos.iloc[rows].astype(int):
            if far:
                cands = [j for j in range(k) if abs(j - i) >= k / 2]
                j = max(cands, key=lambda j: abs(j - i)) if cands else (k - 1 if i < k / 2 else 0)
            else:
                j = i + 1 if i + 1 < k else i - 1
            new.append(lv[j])
        x[c] = x[c].astype(object)
        x.loc[rows, c] = new
    return x


def mcar(good, rng):
    x = good.copy()
    for c in x.columns:
        miss = x[c].isna().to_numpy()
        if not miss.any():
            continue
        observed = x[c].dropna().to_numpy()
        x[c] = x[c].astype(object)
        x.loc[miss, c] = rng.choice(observed, int(miss.sum()))
        x.loc[rng.choice(len(x), int(miss.sum()), replace=False), c] = np.nan
    return x.infer_objects()


def e12(good, rng):
    d = {}
    if ORD:
        d["ordinal_cerca_20"] = ordinal_move(good, np.random.default_rng(rng.integers(1e9)), far=False)
        d["ordinal_lejos_20"] = ordinal_move(good, np.random.default_rng(rng.integers(1e9)), far=True)
    for tag, f in (("005", 0.05), ("010", 0.10), ("025", 0.25)):
        x = good.copy()
        for c in NUM:
            v = _num(x, c)
            x[c] = _round(c, v + f * v.std())
        d[f"desplazar_{tag}"] = x
    x = good.copy()
    for c in NUM:
        v = _num(x, c)
        x[c] = v.clip(v.quantile(0.01), v.quantile(0.99))
    d["colas_extremas"] = x
    x, touched = good.copy(), False
    for c in CAT:
        freq = A[c].astype(str).value_counts(normalize=True)
        rare = set(freq[freq < 0.05].index)
        if rare and len(rare) < len(freq):
            x[c] = x[c].astype(object)
            x.loc[x[c].astype(str).isin(rare), c] = A[c].mode().iloc[0]
            touched = True
    if touched:
        d["categorias_raras_perdidas"] = x
    if (A.isna().sum() > 0).sum() >= 2:
        d["ausentes_mcar"] = mcar(good, rng)
    for tag, f in (("1", 0.01), ("2", 0.02), ("5", 0.05)):
        d[f"copias_{tag}"] = with_copies(good, f, rng)
    for tag, f in (("10", 0.10), ("50", 0.50)):
        x = good.copy()
        m = rng.random(len(x)) < f
        x.loc[m, TARGET] = rng.choice(A[TARGET].dropna().unique(), int(m.sum()))
        d[f"etiquetas_ruidosas_{tag}"] = x
    return d


FAM = {"copias": "privacy", "etiquetas": "utility"}
for k in range(meta["n_good"]):
    rng = np.random.default_rng(2000 + k)
    good = pd.read_pickle(CASES / f"k{k}__bueno.pkl")
    for name, df in e12(good, rng).items():
        assert list(df.columns) == list(A.columns), name
        df.to_pickle(CASES / f"k{k}__{name}.pkl")
        meta["family"][name] = FAM.get(name.split("_")[0], "fidelity")
json.dump(meta, open(CASES / "meta.json", "w"), indent=1, default=str)
print(DS, len(meta["family"]), "casos por buena:", ", ".join(n for n in meta["family"] if n not in ("bueno",)))
