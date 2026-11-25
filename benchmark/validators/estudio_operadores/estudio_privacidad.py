"""¿Cómo agregar la privacidad?  Fugas inyectadas con el valor de cada métrica guardado.

Uso: estudio_privacidad.py <dataset> <salida.jsonl> [n_buenos]

Montaje de los estudios de tipos: A (la «real» que vio el generador), H (holdout) y B (reserva),
tercios de hasta 2000 filas.  Las buenas son remuestreos de B del tamaño de A (ninguna fila de A).
Defectos de privacidad sobre cada buena: una fracción de sus filas sustituida por copias exactas
de filas de A (1, 5, 10, 25, 100 %) o por casi copias (10, 25 %: numéricas + ruido de 5 % de su
desviación típica, discretas redondeadas, y cada categórica cambiada con probabilidad 0,1).
Control sin fuga: las columnas de la buena barajadas por separado (más privada, no menos).
Se evalúa la familia de privacidad completa con H como holdout; se guarda cada métrica.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from metis.interface.sdk import Evaluator

DS, OUT = sys.argv[1], Path(sys.argv[2])
N_GOOD = int(sys.argv[3]) if len(sys.argv) > 3 else 8
K_FROM = int(sys.argv[4]) if len(sys.argv) > 4 else 0
MAX_ROWS = 6000
ADULT_EDU = ["Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", "11th", "12th", "HS-grad",
             "Some-college", "Assoc-voc", "Assoc-acdm", "Bachelors", "Masters", "Prof-school", "Doctorate"]

if DS == "adult":
    from metis.infrastructure.typing import infer_schema

    real = pd.read_csv("data/real/adult_test.csv", skiprows=[1], skipinitialspace=True)
    real = real.drop(columns=["Education_Num"])
    for c in real.columns:
        if real[c].dtype == object:
            real[c] = real[c].str.strip().str.rstrip(".")
    schema = dict(infer_schema(real).schema)
    schema["Education"] = {"type": "ordinal", "levels": ADULT_EDU}
else:
    CSV = {"rrhh": "rrhh.csv", "cardio": "cardio_train.csv", "telco": "telco.csv"}
    schema = yaml.safe_load(open(f"metis/configs/config_{DS}.yaml", encoding="utf-8"))["data"]["schema"]
    real = pd.read_csv(f"data/real/{CSV[DS]}", sep=None, engine="python", encoding="utf-8-sig")
schema = {c: s for c, s in schema.items() if c in real.columns}
real = real[list(schema)]
if len(real) > MAX_ROWS:
    real = real.sample(MAX_ROWS, random_state=0).reset_index(drop=True)
idx = np.random.default_rng(0).permutation(len(real))
third = len(real) // 3
A, H, B = (real.iloc[idx[i * third : (i + 1) * third]].reset_index(drop=True) for i in range(3))


def typ(spec):
    return spec if isinstance(spec, str) else spec.get("type")


NUM = [c for c, s in schema.items() if typ(s) in ("continuous", "discrete")]
DISC = {c for c, s in schema.items() if typ(s) == "discrete"}
CAT = [c for c, s in schema.items() if typ(s) in ("nominal", "ordinal")]


def with_copies(good, frac, rng, near=False):
    x = good.copy()
    k = int(round(frac * len(x)))
    if k == 0:
        return x
    rows = rng.choice(len(x), k, replace=False)
    src = A.sample(k, replace=False, random_state=int(rng.integers(1e9))).reset_index(drop=True)
    if near:
        for c in NUM:
            v = pd.to_numeric(src[c], errors="coerce")
            sd = pd.to_numeric(A[c], errors="coerce").std()
            v = v + rng.normal(0, 0.05 * (sd if np.isfinite(sd) and sd > 0 else 1.0), k)
            src[c] = np.round(v) if c in DISC else v
        for c in CAT:
            change = rng.random(k) < 0.1
            if change.any():
                src.loc[change, c] = rng.choice(A[c].dropna().to_numpy(), int(change.sum()))
    for c in x.columns:
        x[c] = x[c].astype(object)
        x.loc[rows, c] = src[c].astype(object).to_numpy()
    return x.infer_objects()


def defects(good, rng):
    d = {f"copias_{int(f * 100)}": with_copies(good, f, rng) for f in (0.01, 0.05, 0.10, 0.25, 1.0)}
    d.update({f"casi_copias_{int(f * 100)}": with_copies(good, f, rng, near=True) for f in (0.10, 0.25)})
    ctrl = good.copy()
    for c in ctrl.columns:
        ctrl[c] = rng.permutation(ctrl[c].to_numpy())
    d["control_sin_fuga"] = ctrl
    return d


done = set()
if OUT.exists():
    for line in OUT.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        done.add((r["muestra"], r["defecto"]))

ev = Evaluator()
# El objetivo es el atributo sensible por defecto (inference_attack, l_diversity, t_closeness).
TARGET = {"rrhh": "Attrition", "cardio": "cardio", "telco": "Churn", "adult": "Target"}[DS]
conf = {"data": {"schema": schema, "target": TARGET, "task_type": "classification",
                 "constraints": {"from_schema": False}},
        "metrics": ["privacy"], "evaluation": {"n_runs": 1}}
for k in range(K_FROM, N_GOOD):
    rng = np.random.default_rng(100 + k)
    good = B.sample(len(A), replace=True, random_state=k).reset_index(drop=True)
    for defect, synth in {"bueno": good, **defects(good, rng)}.items():
        if (k, defect) in done:
            continue
        t0 = time.perf_counter()
        s = ev.evaluate(A, synth, conf, seed=k, holdout=H)
        rec = {"dataset": DS, "muestra": k, "defecto": defect, "privacy": s.aggregates.get("privacy_score"),
               "segundos": round(time.perf_counter() - t0, 1),
               "metricas": {r.id: (r.value if r.value == r.value else None) for r in s.results}}
        with OUT.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
    print(DS, "muestra", k, "hecha", flush=True)
