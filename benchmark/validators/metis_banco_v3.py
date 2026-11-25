"""METIS sobre el banco de defectos, configurado como en el benchmark v2.

Uso: metis_banco.py <dir_casos>/<dataset> <salida.jsonl> [k_desde k_hasta]

- Configuración: la del dataset (metis/configs/config_<ds>.yaml: métricas, calibración); adult no
  tiene, y usa la de rrhh con su esquema inferido.  Se escribe en ~/validadores/conf/ porque la
  caché de calibración la identifica por el fichero.
- Calibración: como el orquestador del benchmark (sota_models/comparison/orchestrator.py), una vez
  sobre la tabla real completa (full.pkl: A, H y B), cuyos tercios tienen el tamaño de lo que se
  evalúa, y cada evaluación apunta a esas cotas con calibration.bounds_file.
- Utilidad: la señal se mide una vez por tabla (Evaluator.measure_utility_signal sobre A y H).
- Se guardan las notas por familia (calibradas y en bruto), las métricas excluidas por la
  calibración y el eje de reglas (tasa de filas sintéticas que rompen alguna regla).
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import yaml

from metis.calibrate.cache.cache_manager import CacheManager
from metis.calibrate.cache.fingerprint import (
    compute_config_fingerprint,
    compute_data_fingerprint,
    generate_cache_key,
)
from metis.interface.sdk import Evaluator

warnings.filterwarnings("ignore")
CASES, OUT = Path(sys.argv[1]), Path(sys.argv[2])
meta = json.load(open(CASES / "meta.json"))
DS = meta["dataset"]
K_FROM = int(sys.argv[3]) if len(sys.argv) > 3 else 0
K_TO = int(sys.argv[4]) if len(sys.argv) > 4 else meta["n_good"]
CONFIGS = Path("/home/ubuntu/doc-split/metis/metis/configs")

base = yaml.safe_load(open(CONFIGS / f"config_{DS if DS != 'adult' else 'rrhh'}.yaml", encoding="utf-8"))
config = {k: base[k] for k in ("data", "metrics", "reproducibility", "calibration", "evaluation") if k in base}
schema = meta["schema"]
# Ablación E1b (PLANO=1): sin tipos ricos, como una herramienta que solo distingue numérica y
# categórica: ordinales a nominal y discretas a continua (sin niveles, paso ni reglas que salgan de ellos).
PLANO = os.environ.get("PLANO") == "1"
if PLANO:
    def _flat(s):
        t = s if isinstance(s, str) else s.get("type")
        return {"ordinal": "nominal", "discrete": "continuous"}.get(t, s)

    schema = {c: _flat(s) for c, s in schema.items()}
TAG = f"{DS}_plano" if PLANO else DS
config["data"] = {"schema": schema, "target": meta["target"], "task_type": "classification"}
config["evaluation"] = {"n_runs": 1}
_nA, _nF = len(pd.read_pickle(CASES / "A.pkl")), len(pd.read_pickle(CASES / "full.pkl"))
config["calibration"] = {**(config.get("calibration") or {}), "split": [1 / 3, 1 / 3, 1 / 3],
                         "sample_percentage": 100.0 * (_nA + 0.5) / _nF}
conf_dir = Path.home() / "strat" / "v3" / "conf"
conf_dir.mkdir(exist_ok=True)
conf_path = conf_dir / f"config_{TAG}.yaml"
# Varios procesos del mismo dataset (uno por muestra) comparten el fichero: solo se escribe si cambia,
# para que ninguno lea una escritura a medias (la caché de calibración lo identifica por su contenido).
conf_text = yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
if not conf_path.exists() or conf_path.read_text(encoding="utf-8") != conf_text:
    conf_path.write_text(conf_text, encoding="utf-8")

full = pd.read_pickle(CASES / "full.pkl")
A, H = pd.read_pickle(CASES / "A.pkl"), pd.read_pickle(CASES / "H.pkl")
cal = config["calibration"]
n_it, pct, seed0 = cal.get("n_iterations", 10), cal.get("sample_percentage", 100.0), cal.get("base_seed", 42)
t0 = time.perf_counter()
CacheManager(dataset_name=f"banco3_{TAG}").get_or_calibrate(
    real_data=full, config_path=str(conf_path), n_iterations=n_it, sample_percentage=pct, base_seed=seed0,
    n_jobs=cal.get("n_jobs", 1), tune_aggregators=cal.get("tune_aggregators", True))
key = generate_cache_key(data_fingerprint=compute_data_fingerprint(full),
                         config_fingerprint=compute_config_fingerprint(str(conf_path)), n_iterations=n_it,
                         sample_size=int(len(full) * pct / 100.0), base_seed=seed0)
bounds = CacheManager(dataset_name=f"banco3_{TAG}").find_cache_path(key)
assert bounds, "no se encuentra la calibración recién hecha"
config["calibration"] = {**cal, "bounds_file": str(bounds)}
print(DS, "calibración", bounds.name, round(time.perf_counter() - t0), "s", flush=True)

ev = Evaluator()
signal = ev.measure_utility_signal(A, config, holdout=H)
print(DS, "señal de utilidad", {k: v for k, v in (signal or {}).items() if not isinstance(v, (list, dict))}, flush=True)

def jerarquia(h):
    """Nota de cada categoría y subcategoría: «familia.categoría[.subcategoría]» -> nota."""
    out = {}
    for fam, fnode in h.items():
        for cat, cnode in ((fnode or {}).get("categories") or {}).items():
            out[f"{fam}.{cat}"] = cnode.get("score")
            for sub, snode in (cnode.get("subcategories") or {}).items():
                out[f"{fam}.{cat}.{sub}"] = snode.get("score")
    return out


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
        try:
            s = ev.evaluate(A, pd.read_pickle(CASES / f"k{k}__{name}.pkl"), config, seed=k, holdout=H,
                            utility_signal=signal)
            a, c = s.aggregates, s.artifacts.get("constraints") or {}
            rec = {f: a.get(f"{f}_score") for f in ("fidelity", "utility", "privacy")}
            rec.update({f"{f}_raw": a.get(f"{f}_score_raw") for f in ("fidelity", "utility", "privacy")})
            rec.update({"composite": a.get("composite_score"), "calibrated": a.get("composite_score_calibrated"),
                        "n_excluded": len(a.get("calibration_excluded") or []),
                        "validez": None if not c else 1.0 - (c.get("synthetic_rows_violating_any_rate") or 0.0),
                        "reglas_fallidas": c.get("n_failed"), "reglas": c.get("n_rules"),
                        "metricas": {r.id: (r.value if r.value == r.value else None) for r in s.results},
                        "jerarquia": jerarquia(a.get("hierarchy") or {})})
            err = None
        except Exception as exc:  # noqa: BLE001
            rec, err = {}, f"{type(exc).__name__}: {exc}"[:300]
        rec.update({"validador": "metis_plano" if PLANO else "metis", "dataset": DS, "muestra": k, "defecto": name, "familia_defecto": fam,
                    "error": err, "segundos": round(time.perf_counter() - t0, 1)})
        with OUT.open("a") as fh:
            fh.write(json.dumps(rec, default=float) + "\n")
    print("metis", DS, "muestra", k, "hecha", flush=True)
