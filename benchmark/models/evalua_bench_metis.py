"""Benchmark de la Sección 7 con METIS v2 actual (discriminador, log-loss, taxonomía §25), sobre las mismas
sintéticas y particiones del benchmark (~/metis-v2b-runs), sin volver a entrenar generadores.

Uso: evalua_bench_metis.py cal <dataset>                  calibra una vez sobre la tabla real completa
     evalua_bench_metis.py eval <dataset> <salida.jsonl> [i/n]   evalúa los 12 × 5 casos (o la parte i de n)

- Configuración: data, evaluation, reproducibility y calibration de la ejecución original
  (full/<ds>/cart/config.yaml); métricas de metis/configs/config_<ds>.yaml.  Se escribe en ~/bench2/conf.
- Calibración: como el orquestador (sota_models/comparison/orchestrator.py), una vez sobre la tabla real
  limpia completa (entrenamiento + holdout + no vista de la semilla 42); la caché se guarda en
  ~/bench2/cache para poder reagregar sin evaluar.
- Evaluación: real = entrenamiento de la semilla, holdout = su holdout, sintética = la guardada.  Se
  añaden TTS, TRTS y TTRS (no entran en la nota del paper; se quitan al reagregar) para medir la
  coherencia de la utilidad.
"""

from __future__ import annotations

import copy
import json
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import yaml

from metis.application.orchestrator import Orchestrator
from metis.calibrate.cache.cache_manager import CacheManager
from metis.calibrate.cache.fingerprint import compute_config_fingerprint, compute_data_fingerprint, generate_cache_key

warnings.filterwarnings("ignore")
MODE, DS = sys.argv[1], sys.argv[2]
BASE, W = Path("/home/ubuntu/metis-v2b-runs"), Path("/home/ubuntu/strat/bench2")
CONFIGS = Path("/home/ubuntu/doc-split/metis/metis/configs")
EXTRA_UTILITY = ["utility.tts", "utility.trts", "utility.ttrs"]
man = json.load(open(BASE / "partes" / "manifest.json"))[DS]
run_cfg = yaml.safe_load(open(BASE / "full" / DS / "cart" / "config.yaml", encoding="utf-8"))
metrics = yaml.safe_load(open(CONFIGS / f"config_{DS}.yaml", encoding="utf-8"))["metrics"]
config = {k: copy.deepcopy(run_cfg[k]) for k in ("data", "evaluation", "reproducibility", "calibration") if k in run_cfg}
config["data"] = {k: v for k, v in config["data"].items() if k not in ("real", "synthetic", "real_separator", "synth_separator")}
config["calibration"] = {k: v for k, v in config.get("calibration", {}).items() if k != "bounds_file"}
config["metrics"] = list(metrics)
(W / "conf").mkdir(parents=True, exist_ok=True)
(W / "cache").mkdir(parents=True, exist_ok=True)
conf_path = W / "conf" / f"config_{DS}.yaml"
text = yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
if not conf_path.exists() or conf_path.read_text(encoding="utf-8") != text:
    conf_path.write_text(text, encoding="utf-8")


def part(seed, name):
    return pd.read_csv(BASE / "partes" / DS / f"seed{seed}_{name}.csv")


real = pd.concat([part(42, n) for n in ("train", "holdout", "unseen")], ignore_index=True)
cal = config["calibration"]
n_it, pct = cal.get("n_iterations", 10), cal.get("sample_percentage", 100.0)
seed0 = cal.get("base_seed", run_cfg.get("reproducibility", {}).get("seed", 42))
cm = CacheManager(cache_dir=str(W / "cache"), dataset_name=f"bench2_{DS}")
key = generate_cache_key(data_fingerprint=compute_data_fingerprint(real),
                         config_fingerprint=compute_config_fingerprint(str(conf_path)), n_iterations=n_it,
                         sample_size=int(len(real) * pct / 100.0), base_seed=seed0)

if MODE == "cal":
    t0 = time.perf_counter()
    cm.get_or_calibrate(real_data=real, config_path=str(conf_path), n_iterations=n_it, sample_percentage=pct,
                        base_seed=seed0, n_jobs=cal.get("n_jobs", 1), tune_aggregators=cal.get("tune_aggregators", True))
    print(DS, "calibrado", cm.find_cache_path(key), round(time.perf_counter() - t0), "s", flush=True)
    sys.exit(0)

OUT = Path(sys.argv[3])
I, N = (int(v) for v in (sys.argv[4] if len(sys.argv) > 4 else "0/1").split("/"))
bounds = cm.find_cache_path(key)
assert bounds, "falta la calibración: ejecutar primero el modo cal"
eval_cfg = copy.deepcopy(config)
eval_cfg["metrics"] = list(metrics) + EXTRA_UTILITY
eval_cfg["calibration"] = {**cal, "bounds_file": str(bounds)}
done = set()
if OUT.exists():
    for line in OUT.read_text().splitlines():
        r = json.loads(line)
        done.add((r["generador"], r["semilla"]))


def hierarchy(h):
    out = {}
    for fam, fnode in (h or {}).items():
        for cat, cnode in ((fnode or {}).get("categories") or {}).items():
            out[f"{fam}.{cat}"] = cnode.get("score")
            for sub, snode in (cnode.get("subcategories") or {}).items():
                out[f"{fam}.{cat}.{sub}"] = snode.get("score")
    return out


jobs = [(g, int(s), p) for g, seeds in sorted(man["generators"].items()) for s, p in sorted(seeds.items())]
for j, (gen, seed, path) in enumerate(jobs):
    if j % N != I or (gen, seed) in done:
        continue
    t0 = time.perf_counter()
    try:
        s = Orchestrator().evaluate_dataframes(real_data=part(seed, "train"), synth_data=pd.read_csv(path),
                                               config=copy.deepcopy(eval_cfg), seed=seed, config_path=str(conf_path),
                                               holdout=part(seed, "holdout"))
        a = s.aggregates
        rec = {f: a.get(f"{f}_score") for f in ("fidelity", "utility", "privacy")}
        rec.update({"composite": a.get("composite_score"), "calibrated": a.get("composite_score_calibrated"),
                    "metricas": {r.id: (r.value if r.value == r.value else None) for r in s.results},
                    "jerarquia": hierarchy(a.get("hierarchy")), "avisos": a.get("calibration_warnings") or {}})
        err = None
    except Exception as exc:  # noqa: BLE001
        rec, err = {}, f"{type(exc).__name__}: {exc}"[:300]
    rec.update({"validador": "metis", "dataset": DS, "generador": gen, "semilla": seed, "error": err,
                "segundos": round(time.perf_counter() - t0, 1)})
    with OUT.open("a") as fh:
        fh.write(json.dumps(rec, default=float) + "\n")
    print(DS, gen, seed, err or "ok", rec["segundos"], flush=True)
