"""z por nivel de la jerarquía para un defecto del banco (resultados del paper, ~/v3/res_tax3/metis.jsonl)."""
import json
import sys

import numpy as np

DS, DEF = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open("/home/ubuntu/v3/res_tax3/metis.jsonl") if '"error": null' in l]
rows = [r for r in rows if r["dataset"] == DS]
good = [r for r in rows if r["defecto"] == "bueno"]
bad = {r["muestra"]: r for r in rows if r["defecto"] == DEF}


def val(r, key):
    if key in ("composite", "fidelity", "utility", "privacy"):
        return r.get(key)
    if key in r.get("jerarquia", {}):
        return r["jerarquia"][key]
    return r.get("metricas", {}).get(key)


keys = ["composite", "fidelity", "utility", "privacy"]
keys += sorted(k for k in good[0].get("jerarquia", {}) if k.startswith("fidelity."))
keys += sorted(k for k in good[0].get("metricas", {}) if k.startswith("fidelity."))
for k in keys:
    g = np.array([val(r, k) for r in good], dtype=float)
    if np.isnan(g).any():
        continue
    mu, sd = g.mean(), max(g.std(ddof=1), 0.005)
    zs = [(mu - val(bad[m], k)) / sd for m in sorted(bad) if val(bad[m], k) is not None]
    if not zs:
        continue
    print(f"{k:55s} buenas {mu:.3f}±{g.std(ddof=1):.3f}  defecto {np.median([val(bad[m], k) for m in bad]):.3f}  z {np.median(zs):7.1f}")
