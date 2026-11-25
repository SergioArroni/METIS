"""Defectos para medir los «matices locales» que la representación por vistas podría perder (§3.2 del paper).

Uso: gen_matices.py <dir_casos2>/<dataset> <dir_salida>

Reutiliza A, H, full y las buenas del banco 2 (enlaces) y escribe defectos nuevos sobre cada buena:

  Alta cardinalidad (nominales con >= 8 niveles en A; los 5 más frecuentes son la «cabeza»):
    card_cola_fusionada   los niveles de la cola pasan todos al más frecuente de la cola
    card_cola_permutada   los niveles de la cola se reetiquetan con una permutación entre ellos
  Categorías raras (nominales; rara = frecuencia < umbral en A):
    raras_1               las raras al 1 % pasan a la moda, en todas las nominales
    raras_5_una           las raras al 5 % pasan a la moda, solo en la nominal con más masa rara
    raras_infladas        la frecuencia de cada rara al 5 % se duplica con filas de la moda
  Colas pesadas (la numérica con mayor curtosis en A, al menos 20 valores distintos):
    cola_p99_una          valores por encima del p99 recortados al p99
    cola_p95_una          ídem con el p95
    cola_estirada_una     valores por encima del p99 alejados al triple de su distancia a la mediana
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
meta = json.load(open(SRC / "meta.json"))
DS, TARGET, schema = meta["dataset"], meta["target"], meta["schema"]
OUT = OUT / DS
OUT.mkdir(parents=True, exist_ok=True)
A = pd.read_pickle(SRC / "A.pkl")


def typ(spec):
    return spec if isinstance(spec, str) else spec.get("type")


NOM = [c for c, s in schema.items() if typ(s) == "nominal" and c != TARGET and c in A.columns]
NUM = [c for c, s in schema.items() if typ(s) in ("continuous", "discrete") and c != TARGET and c in A.columns]
DISC = {c for c, s in schema.items() if typ(s) == "discrete"}
freq = {c: A[c].astype(str).value_counts(normalize=True) for c in NOM}
HC = [c for c in NOM if len(freq[c]) >= 8]
TAIL = {c: list(freq[c].index[5:]) for c in HC}
RARE1 = {c: [v for v, f in freq[c].items() if f < 0.01] for c in NOM}
RARE5 = {c: [v for v, f in freq[c].items() if f < 0.05] for c in NOM}
RARE5_ONE = max(NOM, key=lambda c: freq[c][RARE5[c]].sum()) if NOM else None
kurt = {c: float(pd.to_numeric(A[c], errors="coerce").kurt()) for c in NUM
        if pd.to_numeric(A[c], errors="coerce").nunique() >= 20}
HEAVY = max(kurt, key=kurt.get) if kurt else None
info = {"alta_cardinalidad": {c: len(freq[c]) for c in HC},
        "raras_1": {c: len(v) for c, v in RARE1.items() if v},
        "raras_5_una": RARE5_ONE and {RARE5_ONE: round(float(freq[RARE5_ONE][RARE5[RARE5_ONE]].sum()), 3)},
        "cola_pesada": HEAVY and {HEAVY: round(kurt[HEAVY], 1)}}


def defects(good, rng):
    d = {}
    if HC:
        x = good.copy()
        for c in HC:
            s = x[c].astype(str)
            x[c] = s.where(~s.isin(TAIL[c]), TAIL[c][0])
        d["card_cola_fusionada"] = x
        x = good.copy()
        for c in HC:
            perm = dict(zip(TAIL[c], rng.permutation(TAIL[c])))
            x[c] = x[c].astype(str).map(lambda v, p=perm: p.get(v, v))
        d["card_cola_permutada"] = x
    if any(RARE1.values()):
        x = good.copy()
        for c in NOM:
            if RARE1[c]:
                s = x[c].astype(str)
                x[c] = s.where(~s.isin(RARE1[c]), freq[c].index[0])
        d["raras_1"] = x
    if RARE5_ONE and RARE5[RARE5_ONE]:
        c = RARE5_ONE
        x = good.copy()
        s = x[c].astype(str)
        x[c] = s.where(~s.isin(RARE5[c]), freq[c].index[0])
        d["raras_5_una"] = x
        x = good.copy()
        for c in NOM:
            s = x[c].astype(str)
            mode_rows = np.flatnonzero((s == freq[c].index[0]).to_numpy())
            rng.shuffle(mode_rows)
            pos = 0
            for v in RARE5[c]:
                n = int((s == v).sum())
                take = mode_rows[pos:pos + n]
                pos += n
                x.loc[x.index[take], c] = v
        d["raras_infladas"] = x
    if HEAVY:
        c = HEAVY
        v = pd.to_numeric(good[c], errors="coerce")
        for name, q in (("cola_p99_una", 0.99), ("cola_p95_una", 0.95)):
            x = good.copy()
            x[c] = v.clip(upper=v.quantile(q))
            d[name] = x
        x = good.copy()
        med, hi = v.median(), v.quantile(0.99)
        w = v.where(v <= hi, med + 3 * (v - med))
        x[c] = np.round(w) if c in DISC else w
        d["cola_estirada_una"] = x
    return d


family = {}
for k in range(meta["n_good"]):
    good = pd.read_pickle(SRC / f"k{k}__bueno.pkl")
    for name, df in defects(good, np.random.default_rng(1000 + k)).items():
        df.to_pickle(OUT / f"k{k}__{name}.pkl")
        family[name] = "fidelity"
for f in ("A.pkl", "H.pkl", "full.pkl"):
    if not (OUT / f).exists():
        os.symlink(SRC / f, OUT / f)
json.dump({**meta, "family": family, "matices": info}, open(OUT / "meta.json", "w"), indent=1, default=str)
print(DS, info, sorted(family))
