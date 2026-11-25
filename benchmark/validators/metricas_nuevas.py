"""Prototipos de las métricas candidatas a METIS, fuera de METIS (estudio de si aportan).

Uso: metricas_nuevas.py <dir_casos>/<dataset> <salida.jsonl> [k_desde k_hasta]

Para cada caso (las tablas del banco, k<k>__<defecto>.pkl, frente a A y H) calcula:

  utilidad con varias métricas  CatBoost (protocolo de METIS: parada temprana en un 20 % del
      entrenamiento, 1 hilo) entrenado en la tabla y probado en H: AUC, log-loss, Brier, accuracy y
      F1 macro.  Cada una se lleva a la escala de METIS: ganancia sobre el modelo sin señal (las
      frecuencias de clase) relativa a la de TTR (entrenado en A), u = 1 - |1 - g|, recortada a
      [0, 1]; u_multi = media de las cinco.
  discriminador  CatBoost que distingue filas de A de filas de la tabla, AUC en validación cruzada
      de 3 pliegues; nota = 2 (1 - AUC) recortada a [0, 1] (1 = indistinguibles) y las 3 columnas
      que más lo delatan.
  alpha_precision, beta_coverage, authenticity  Alaa et al. (2022), versión sin red de embebido
      (la «naive» de synthcity, mismo cálculo) sobre filas codificadas: numéricas y ordinales
      (rango) estandarizadas con A, nominales en one-hot / sqrt(2), huecos a la media + indicador.
  rare_recall  de cada categórica, los niveles con < 5 % en A: media de min(f_s/f_r, f_r/f_s).
  trios  1 - TVD media de las tablas de contingencia de tríos de columnas (categóricas y
      numéricas en quintiles de A; todos los tríos o 200 fijos por dataset).
"""

from __future__ import annotations

import itertools
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")
CASES, OUT = Path(sys.argv[1]), Path(sys.argv[2])
meta = json.load(open(CASES / "meta.json"))
DS, T = meta["dataset"], meta["target"]
K_FROM = int(sys.argv[3]) if len(sys.argv) > 3 else 0
K_TO = int(sys.argv[4]) if len(sys.argv) > 4 else meta["n_good"]


def typ(s):
    return s if isinstance(s, str) else s.get("type")


SCHEMA = {c: s for c, s in meta["schema"].items() if typ(s) != "id"}
NUM = [c for c, s in SCHEMA.items() if typ(s) in ("continuous", "discrete")]
ORD = [c for c, s in SCHEMA.items() if typ(s) == "ordinal"]
NOM = [c for c, s in SCHEMA.items() if typ(s) == "nominal"]
CAT = ORD + NOM
A = pd.read_pickle(CASES / "A.pkl")[list(SCHEMA)]
H = pd.read_pickle(CASES / "H.pkl")[list(SCHEMA)]


def clean(df):
    out = df[list(SCHEMA)].copy()
    for c in out.columns:
        if c in CAT or c == T:
            out[c] = out[c].astype(str)
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    return out


A_c, H_c = clean(A), clean(H)

# ---------------------------------------------------------------- codificación de filas (α/β)
LEVELS = {c: sorted(A_c[c].unique()) for c in NOM}
RANK = {c: {str(v): i for i, v in enumerate(SCHEMA[c]["levels"])} for c in ORD if isinstance(SCHEMA[c], dict)}


def _ordinal(df, c):
    if c in RANK:
        return df[c].map(RANK[c]).astype(float)
    return pd.to_numeric(df[c], errors="coerce")


STATS = {}
for c in NUM + ORD:
    v = pd.to_numeric(A_c[c], errors="coerce") if c in NUM else _ordinal(A_c, c)
    STATS[c] = (v.mean(), v.std() or 1.0, bool(v.isna().any()))


def encode(df):
    cols = []
    for c in NUM + ORD:
        v = pd.to_numeric(df[c], errors="coerce") if c in NUM else _ordinal(df, c)
        mu, sd, miss = STATS[c]
        z = ((v - mu) / sd).to_numpy(dtype=float)
        cols.append(np.nan_to_num(z, nan=0.0))
        if miss:
            cols.append(v.isna().to_numpy(dtype=float))
    for c in NOM:
        for lv in LEVELS[c]:
            cols.append((df[c] == lv).to_numpy(dtype=float) / np.sqrt(2))
    return np.column_stack(cols)


X_A = encode(A_c)


def alpha_beta(X, X_syn):
    """Alaa et al. (2022), cálculo de synthcity (AlphaPrecision.metrics, sin embebido)."""
    center = X.mean(axis=0)
    alphas = np.linspace(0, 1, 30)
    radii = np.quantile(np.sqrt(((X - center) ** 2).sum(axis=1)), alphas)
    synth_center = X_syn.mean(axis=0)
    synth_to_center = np.sqrt(((X_syn - center) ** 2).sum(axis=1))
    real_to_real = NearestNeighbors(n_neighbors=2).fit(X).kneighbors(X)[0][:, 1]
    d, idx = NearestNeighbors(n_neighbors=1).fit(X_syn).kneighbors(X)
    real_to_synth, idx = d[:, 0], idx[:, 0]
    closest_d = np.sqrt(((X_syn[idx] - synth_center) ** 2).sum(axis=1))
    closest_radii = np.quantile(closest_d, alphas)
    prec = [np.mean(synth_to_center <= r) for r in radii]
    cov = [np.mean((real_to_synth <= real_to_real) * (closest_d <= r)) for r in closest_radii]
    d_prec = 1 - np.sum(np.abs(alphas - prec)) / np.sum(alphas)
    d_cov = 1 - np.sum(np.abs(alphas - cov)) / np.sum(alphas)
    auth = np.mean(real_to_real[idx] < real_to_synth)
    return max(0.0, float(d_prec)), max(0.0, float(d_cov)), float(auth)


# ---------------------------------------------------------------- utilidad con varias métricas
FEATS = [c for c in SCHEMA if c != T]
CATF = [c for c in FEATS if c in CAT]
CLASSES = sorted(A_c[T].unique())


def fit_predict(train, seed):
    X, y = train[FEATS], train[T]
    if y.nunique() < 2:
        p = np.zeros((len(H_c), len(CLASSES)))
        p[:, CLASSES.index(y.iloc[0])] = 1.0
        return p
    strat = y if y.value_counts().min() >= 2 else None
    Xf, Xv, yf, yv = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=strat)
    m = CatBoostClassifier(iterations=1000, one_hot_max_size=10, thread_count=1, random_seed=seed, verbose=0,
                           cat_features=CATF, early_stopping_rounds=50)
    m.fit(Xf, yf, eval_set=(Xv, yv))
    proba = m.predict_proba(H_c[FEATS])
    out = np.zeros((len(H_c), len(CLASSES)))
    for j, c in enumerate(m.classes_):
        if str(c) in CLASSES:
            out[:, CLASSES.index(str(c))] = proba[:, j]
    return out


Y_H = H_c[T].to_numpy()
Y_IDX = np.array([CLASSES.index(v) if v in CLASSES else -1 for v in Y_H])
ONEHOT = np.eye(len(CLASSES))[np.clip(Y_IDX, 0, None)] * (Y_IDX >= 0)[:, None]


def scores(p):
    p = np.clip(p, 1e-6, 1)
    p = p / p.sum(axis=1, keepdims=True)
    pred = np.array(CLASSES)[p.argmax(axis=1)]
    if len(CLASSES) == 2:
        auc = roc_auc_score(Y_IDX == 1, p[:, 1])
    else:
        auc = np.mean([roc_auc_score(Y_IDX == j, p[:, j]) for j in range(len(CLASSES)) if 0 < (Y_IDX == j).sum() < len(Y_IDX)])
    return {"auc": auc, "logloss": log_loss(Y_IDX, p, labels=list(range(len(CLASSES)))),
            "brier": float(np.mean(((p - ONEHOT) ** 2).sum(axis=1))),
            "accuracy": accuracy_score(Y_H, pred), "f1": f1_score(Y_H, pred, average="macro")}


def chance_scores(train):
    freq = train[T].value_counts(normalize=True)
    p = np.tile([freq.get(c, 0.0) for c in CLASSES], (len(H_c), 1))
    s = scores(p)
    s["auc"] = 0.5
    return s


LOWER_BETTER = {"logloss", "brier"}
TTR = scores(fit_predict(A_c, 0))
CHANCE = chance_scores(A_c)


def utility(table, seed):
    s = scores(fit_predict(table, seed))
    out = {}
    for m, v in s.items():
        span = (CHANCE[m] - TTR[m]) if m in LOWER_BETTER else (TTR[m] - CHANCE[m])
        g = ((CHANCE[m] - v) if m in LOWER_BETTER else (v - CHANCE[m])) / span if span > 1e-9 else np.nan
        out[f"u_{m}"] = float(np.clip(1 - abs(1 - g), 0, 1)) if np.isfinite(g) else None
        out[f"raw_{m}"] = float(v)
    vals = [out[f"u_{m}"] for m in s if out[f"u_{m}"] is not None]
    out["u_multi"] = float(np.mean(vals)) if vals else None
    return out


# ---------------------------------------------------------------- discriminador
def discriminator(table, seed):
    X = pd.concat([A_c, table], ignore_index=True)
    y = np.r_[np.zeros(len(A_c)), np.ones(len(table))]
    aucs = []
    for tr, te in StratifiedKFold(3, shuffle=True, random_state=seed).split(X, y):
        m = CatBoostClassifier(iterations=300, one_hot_max_size=10, thread_count=1, random_seed=seed, verbose=0,
                               cat_features=CAT + ([T] if T not in CAT else []))
        m.fit(X.iloc[tr], y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(X.iloc[te])[:, 1]))
    imp = pd.Series(m.get_feature_importance(), index=X.columns).sort_values(ascending=False)
    auc = float(np.mean(aucs))
    return {"disc_auc": auc, "discriminador": float(np.clip(2 * (1 - auc), 0, 1)), "disc_top": list(imp.index[:3])}


# ---------------------------------------------------------------- categorías raras
RARE = {}
for c in CAT + ([T] if T not in CAT else []):
    f = A_c[c].value_counts(normalize=True)
    r = f[f < 0.05]
    if len(r) and len(r) < len(f):
        RARE[c] = r.to_dict()


def rare_recall(table):
    vals = []
    for c, levels in RARE.items():
        fs = table[c].value_counts(normalize=True)
        for lv, fr in levels.items():
            s = fs.get(lv, 0.0)
            vals.append(min(s / fr, fr / s) if s > 0 else 0.0)
    return float(np.mean(vals)) if vals else None


# ---------------------------------------------------------------- tríos de columnas
EDGES = {}
for c in NUM:
    v = pd.to_numeric(A_c[c], errors="coerce").dropna()
    e = np.unique(np.quantile(v, [0.2, 0.4, 0.6, 0.8]))
    if len(e):
        EDGES[c] = e
TRIO_COLS = CAT + ([T] if T not in CAT else []) + list(EDGES)


def binned(df):
    out = pd.DataFrame(index=df.index)
    for c in TRIO_COLS:
        if c in EDGES:
            v = pd.to_numeric(df[c], errors="coerce")
            out[c] = np.where(v.isna(), -1, np.searchsorted(EDGES[c], v.to_numpy(), side="right")).astype(str)
        else:
            out[c] = df[c].astype(str)
    return out


ALL_TRIOS = list(itertools.combinations(TRIO_COLS, 3))
TRIOS = ALL_TRIOS if len(ALL_TRIOS) <= 200 else [ALL_TRIOS[i] for i in
                                                  np.random.default_rng(0).choice(len(ALL_TRIOS), 200, replace=False)]
B_A = binned(A_c)


def trios(table):
    b = binned(table)
    tvd = []
    for t in TRIOS:
        pa = B_A.groupby(list(t)).size() / len(B_A)
        pb = b.groupby(list(t)).size() / len(b)
        tvd.append(0.5 * pa.subtract(pb, fill_value=0).abs().sum())
    return float(1 - np.mean(tvd))


# ---------------------------------------------------------------- bucle
done = set()
if OUT.exists():
    for line in OUT.read_text().splitlines():
        r = json.loads(line)
        if r["dataset"] == DS:
            done.add((r["muestra"], r["defecto"]))

for k in range(K_FROM, K_TO):
    for name, fam in meta["family"].items():
        if (k, name) in done:
            continue
        t0 = time.perf_counter()
        rec, err, secs = {}, None, {}
        try:
            tab = clean(pd.read_pickle(CASES / f"k{k}__{name}.pkl"))
            for part, fn in (("utilidad", lambda: utility(tab, k)), ("discriminador", lambda: discriminator(tab, k)),
                             ("alaa", lambda: dict(zip(("alpha_precision", "beta_coverage", "authenticity"),
                                                       alpha_beta(X_A, encode(tab))))),
                             ("raras", lambda: {"rare_recall": rare_recall(tab)}), ("trios", lambda: {"trios": trios(tab)})):
                t1 = time.perf_counter()
                rec.update(fn())
                secs[part] = round(time.perf_counter() - t1, 2)
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"[:300]
        rec.update({"dataset": DS, "muestra": k, "defecto": name, "familia_defecto": fam, "error": err,
                    "segundos": secs, "total": round(time.perf_counter() - t0, 1)})
        with OUT.open("a") as fh:
            fh.write(json.dumps(rec, default=float) + "\n")
    print("nuevas", DS, "muestra", k, "hecha", flush=True)
