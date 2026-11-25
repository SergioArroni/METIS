"""Reescribe registros del banco con la taxonomía instalada (§26: tres categorías, sin fidelity.missingness).

Uso: reagrega_banco.py <dir_cache> <salida.jsonl> <entrada.jsonl> [<entrada.jsonl> ...]

Mismos valores por métrica y mismas calibraciones (banco3_<ds> o banco3_<ds>_plano según el validador del
registro); cambian las notas (compuesto, familias, jerarquía).  fidelity.missingness se quita de las
métricas, también para la media plana del informe.
"""
import json
import sys
from pathlib import Path

import reagrega as ra
from metis.calibrate.core.bounds import CalibrationBounds

CACHE, OUT, INS = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
DROP = {"fidelity.missingness"}
_b = {}


def bounds(ds, plano):
    k = (ds, plano)
    if k not in _b:
        pat = f"bounds_banco3_{ds}_plano_calibration_*.json" if plano else f"bounds_banco3_{ds}_calibration_*.json"
        files = [f for f in sorted(CACHE.glob(pat), key=lambda p: p.stat().st_mtime) if plano or "_plano_" not in f.name]
        _b[k] = CalibrationBounds.load(str(files[-1]))
    return _b[k]


OUT.parent.mkdir(parents=True, exist_ok=True)
n = 0
with OUT.open("w") as fh:
    for f in INS:
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r.get("error"):
                fh.write(line)
                continue
            r["metricas"] = {k: v for k, v in (r.get("metricas") or {}).items() if k not in DROP}
            a = ra.aggregate(r, bounds(r["dataset"], r.get("validador") == "metis_plano"))
            r.update({k: a[k] for k in ("composite", "fidelity", "utility", "privacy")})
            r["jerarquia"] = a["jerarquia"]
            fh.write(json.dumps(r, default=float) + "\n")
            n += 1
print("escrito", OUT, n)
