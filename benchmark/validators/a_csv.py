"""Copia en CSV de los casos de un dataset (para el entorno de synthcity, con numpy 1).

Uso: a_csv.py <dir_casos>/<dataset>
Cada <nombre>.pkl pasa a <nombre>.csv.gz sin identificadores, con categóricas como texto y numéricas como float,
la misma limpieza que evalua.py aplica a SDMetrics y SynthEval.
"""
import json
import sys
from pathlib import Path

import pandas as pd

d = Path(sys.argv[1])
meta = json.load(open(d / "meta.json"))
cat_cols = meta["cat_cols"]
ids = [c for c, s in meta["schema"].items() if (s if isinstance(s, str) else s.get("type")) == "id"]
n = 0
for f in sorted(d.glob("*.pkl")):
    df = pd.read_pickle(f).drop(columns=ids, errors="ignore")
    for c in df.columns:
        df[c] = df[c].astype(str) if c in cat_cols else pd.to_numeric(df[c], errors="coerce").astype(float)
    df.to_csv(d / (f.stem + ".csv.gz"), index=False)
    n += 1
print(d.name, n)
