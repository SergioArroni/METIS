"""Señal de utilidad de RRHH sobre A y H (la que usa cada evaluación) con cada esquema."""
import warnings
from pathlib import Path

import pandas as pd
import yaml

from metis.interface.sdk import Evaluator

warnings.filterwarnings("ignore")
CASES = Path("/home/ubuntu/strat/validadores/casos2/rrhh")
A, H = pd.read_pickle(CASES / "A.pkl"), pd.read_pickle(CASES / "H.pkl")
ev = Evaluator()
for tag in ("rrhh", "rrhh_n2", "rrhh_n0", "rrhh_plano"):
    config = yaml.safe_load(open(f"/home/ubuntu/strat/v3/conf/config_{tag}.yaml", encoding="utf-8"))
    s = ev.measure_utility_signal(A, config, holdout=H)
    f = lambda d: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items() if k in ("applicable", "lower", "upper", "ttr_gain", "noise", "threshold")}
    print(tag, "AUC", f(s), "\n     log-loss", f(s.get("log_loss") or {}), (s.get("log_loss") or {}).get("reason"))
