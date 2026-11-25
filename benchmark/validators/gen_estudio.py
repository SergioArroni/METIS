"""Defectos para el estudio de métricas nuevas, sobre las buenas del banco 2.

Uso: gen_estudio.py <dataset> <dir_casos2> <dir_estudio>

Cada uno pone a prueba una métrica candidata en lo que ninguna actual debería ver bien:

  trios_rotos       se elige en A el trío de categóricas (≤ 8 niveles) con más interacción de
                    orden 3 (G² del modelo sin término triple, ajustado por IPF, entre filas); en
                    cada buena, la columna Z del trío (la menos ligada al resto de columnas) se
                    vuelve a sortear de q(Z | X, Y), el modelo de la buena con sus tres tablas de
                    pares intactas y sin interacción triple.  Los pares del trío se conservan; los
                    de Z con columnas de fuera del trío pueden moverse algo (se mide y se guarda).
  colapso_modos     k-medias (k = 5) sobre A codificada; las filas de la buena del grupo con masa
                    más cercana al 20 % se sustituyen por filas reales de la reserva B que no están
                    en la buena ni en ese grupo (sin reemplazo): falta una región, sin duplicados.
  filas_imposibles  el 10 % de las filas: cada columna toma el valor de otra fila al azar, por
                    separado.  Cada valor es plausible solo; la combinación, no.  Marginales intactas.

Escribe <dir_estudio>/<dataset>/: enlaces a A, H, B y full del banco 2, las buenas, los tres
defectos por buena y meta.json (familias: bueno + los tres).
"""

from __future__ import annotations

import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

DS, SRC, DST = sys.argv[1], Path(sys.argv[2]) / sys.argv[1], Path(sys.argv[3]) / sys.argv[1]
DST.mkdir(parents=True, exist_ok=True)
meta = json.load(open(SRC / "meta.json"))
T = meta["target"]


def typ(s):
    return s if isinstance(s, str) else s.get("type")


SCHEMA = {c: s for c, s in meta["schema"].items() if typ(s) != "id"}
CAT = [c for c, s in SCHEMA.items() if typ(s) in ("nominal", "ordinal")]
NUM = [c for c, s in SCHEMA.items() if typ(s) in ("continuous", "discrete")]
A, B = pd.read_pickle(SRC / "A.pkl"), pd.read_pickle(SRC / "B.pkl")
for f in ("A.pkl", "H.pkl", "B.pkl", "full.pkl"):
    if not (DST / f).exists():
        os.symlink(SRC / f, DST / f)


# ------------------------------------------------------------------ trío
def ipf(n, iters=100):
    """Modelo sin interacción triple con las tres tablas de pares de *n* (array 3-D)."""
    m = np.ones_like(n, dtype=float)
    for _ in range(iters):
        for ax in (2, 1, 0):
            target, cur = n.sum(axis=ax, keepdims=True), m.sum(axis=ax, keepdims=True)
            m = m * np.divide(target, cur, out=np.zeros_like(cur), where=cur > 0)
    return m


def table(df, cols, levels):
    codes = [pd.Categorical(df[c].astype(str), categories=levels[c]).codes for c in cols]
    n = np.zeros([len(levels[c]) for c in cols])
    ok = np.all(np.array(codes) >= 0, axis=0)
    np.add.at(n, tuple(np.array(codes)[:, ok]), 1)
    return n


def cramers_v(x, y):
    t = pd.crosstab(x.astype(str), y.astype(str)).to_numpy()
    if min(t.shape) < 2:
        return 0.0
    n = t.sum()
    e = t.sum(1, keepdims=True) * t.sum(0, keepdims=True) / n
    chi2 = np.nansum((t - e) ** 2 / np.where(e > 0, e, np.nan))
    return float(np.sqrt(chi2 / n / (min(t.shape) - 1)))


cands = [c for c in CAT + ([T] if T not in CAT else []) if 2 <= A[c].astype(str).nunique() <= 8]
LEVELS = {c: sorted(pd.concat([A[c], B[c]]).astype(str).unique()) for c in cands}
best = None
for trio in itertools.combinations(cands, 3):
    n = table(A, trio, LEVELS)
    m = ipf(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        g2 = 2 * np.nansum(np.where(n > 0, n * np.log(n / m), 0.0)) / n.sum()
    if best is None or g2 > best[0]:
        best = (g2, trio)
g2, trio = best
others = [c for c in SCHEMA if c not in trio]
Z = min(trio, key=lambda c: np.mean([cramers_v(A[c], pd.qcut(pd.to_numeric(A[o], errors="coerce"), 5, duplicates="drop")
                                               if o in NUM else A[o]) for o in others]))
XY = [c for c in trio if c != Z]


def trios_rotos(good, rng):
    cols = XY + [Z]
    n = table(good, cols, LEVELS)
    m = ipf(n)
    x = good.copy()
    x[Z] = x[Z].astype(object)
    cx = pd.Categorical(good[XY[0]].astype(str), categories=LEVELS[XY[0]]).codes
    cy = pd.Categorical(good[XY[1]].astype(str), categories=LEVELS[XY[1]]).codes
    for i in range(len(good)):
        if cx[i] < 0 or cy[i] < 0:
            continue
        q = m[cx[i], cy[i], :]
        if q.sum() <= 0:
            continue
        x.iat[i, x.columns.get_loc(Z)] = LEVELS[Z][rng.choice(len(q), p=q / q.sum())]
    # Los valores vuelven al tipo de la columna original cuando es numérica (cardio: 0/1).
    if pd.api.types.is_numeric_dtype(good[Z]):
        x[Z] = pd.to_numeric(x[Z])
    return x


# ------------------------------------------------------------------ colapso
def encode(df, ref):
    cols = []
    for c in NUM:
        v, r = pd.to_numeric(df[c], errors="coerce"), pd.to_numeric(ref[c], errors="coerce")
        cols.append(((v - r.mean()) / (r.std() or 1)).fillna(0).to_numpy())
    for c in CAT + ([T] if T not in CAT else []):
        for lv in sorted(ref[c].astype(str).unique()):
            cols.append((df[c].astype(str) == lv).to_numpy(float) / np.sqrt(2))
    return np.column_stack(cols)


km = KMeans(5, n_init=10, random_state=0).fit(encode(A, A))
mass = np.bincount(km.labels_, minlength=5) / len(A)
CLUSTER = int(np.argmin(np.abs(mass - 0.2)))


def colapso(good, k, rng):
    from sklearn.model_selection import train_test_split

    _lab = B[meta["target"]].astype(str)
    _st = _lab if _lab.nunique() > 1 and _lab.value_counts().min() >= 3 else None
    idx_good, _ = train_test_split(B.index, train_size=len(A), random_state=k, stratify=_st)
    pool = B.drop(idx_good)
    pool = pool[km.predict(encode(pool, A)) != CLUSTER]
    hit = np.flatnonzero(km.predict(encode(good, A)) == CLUSTER)
    x = good.copy()
    take = pool.sample(len(hit), replace=False, random_state=int(rng.integers(1e9)))
    for j, col in enumerate(x.columns):
        x.iloc[hit, j] = take[col].to_numpy()
    return x, len(hit) / len(good)


# ------------------------------------------------------------------ filas imposibles
def imposibles(good, rng, frac=0.1):
    x = good.copy()
    rows = rng.choice(len(x), int(round(frac * len(x))), replace=False)
    for c in x.columns:
        src = rng.integers(len(good), size=len(rows))
        x.iloc[rows, x.columns.get_loc(c)] = good[c].iloc[src].to_numpy()
    return x


fam = {"bueno": None, "trios_rotos": "fidelity", "colapso_modos": "fidelity", "filas_imposibles": "fidelity"}
info = {"trio": list(trio), "Z": Z, "g2_por_fila": g2, "grupo_colapsado": CLUSTER, "masa_grupo_en_A": float(mass[CLUSTER]),
        "fraccion_sustituida": [], "cramer_Z_fuera_delta": []}
for k in range(meta["n_good"]):
    rng = np.random.default_rng(5000 + k)
    good = pd.read_pickle(SRC / f"k{k}__bueno.pkl")
    good.to_pickle(DST / f"k{k}__bueno.pkl")
    t = trios_rotos(good, rng)
    t.to_pickle(DST / f"k{k}__trios_rotos.pkl")
    info["cramer_Z_fuera_delta"].append(float(np.mean([abs(cramers_v(t[Z], t[o] if o not in NUM else pd.qcut(pd.to_numeric(t[o], errors="coerce"), 5, duplicates="drop"))
                                                          - cramers_v(good[Z], good[o] if o not in NUM else pd.qcut(pd.to_numeric(good[o], errors="coerce"), 5, duplicates="drop")))
                                                      for o in others])))
    c, f = colapso(good, k, rng)
    c.to_pickle(DST / f"k{k}__colapso_modos.pkl")
    info["fraccion_sustituida"].append(f)
    imposibles(good, rng).to_pickle(DST / f"k{k}__filas_imposibles.pkl")
json.dump({**meta, "family": fam, "estudio": info}, open(DST / "meta.json", "w"), indent=1, default=str)
print(DS, json.dumps(info, default=float))
