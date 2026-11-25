"""Estudio de los tipos: caída pareada (buena k - defecto k) por esquema.  Uso: analiza_tipos.py <dir_res>"""
import glob
import json
import sys
from collections import defaultdict

import numpy as np

R = defaultdict(dict)  # (ds, nivel) -> {(k, defecto): rec}
for f in glob.glob(sys.argv[1] + "/*.jsonl"):
    name = f.split("/")[-1][:-6]
    if name.startswith("cal_"):
        continue
    ds, lv, _ = name.rsplit("_", 2)
    for l in open(f):
        r = json.loads(l)
        if r.get("error"):
            print("ERROR", name, r["defecto"], r["error"][:150])
            continue
        R[(ds, lv)][(r["muestra"], r["defecto"])] = r
ORDEN = ["n0", "n1", "n2", "ninf", "n3"]
NOM = {"n0": "formato", "n1": "plano", "n2": "sin ord.", "ninf": "deducido", "n3": "completo"}
for nota in ("fidelity", "fidelity_raw", "composite", "validez"):
    print(f"\n===== {nota}: caída media pareada [mín, máx] · muestras con caída > 0 · z (caída / sd de las buenas, suelo 0,005)")
    for ds in sorted({k[0] for k in R}):
        lvs = [l for l in ORDEN if (ds, l) in R]
        defs = sorted({d for l in lvs for _, d in R[(ds, l)] if d != "bueno"})
        print(f"\n{ds}: sd de las buenas  " + "  ".join(
            f"{NOM[l]} {np.std([R[(ds, l)][(k, 'bueno')].get(nota) or np.nan for k in range(5) if (k, 'bueno') in R[(ds, l)]], ddof=1):.4f}"
            for l in lvs) + "   media buenas  " + "  ".join(
            f"{np.mean([R[(ds, l)][(k, 'bueno')].get(nota) or np.nan for k in range(5) if (k, 'bueno') in R[(ds, l)]]):.3f}" for l in lvs))
        for d in defs:
            row = []
            for l in lvs:
                rr = R[(ds, l)]
                ks = [k for k in range(5) if (k, d) in rr and (k, "bueno") in rr]
                g = np.array([rr[(k, "bueno")].get(nota) for k in ks], float)
                x = np.array([rr[(k, d)].get(nota) for k in ks], float)
                if not len(ks) or np.isnan(g).all():
                    row.append(f"{NOM[l]:9} —")
                    continue
                dr = g - x
                sd = max(np.std(g, ddof=1), 0.005) if len(g) > 1 else np.nan
                row.append(f"{NOM[l]:9} {dr.mean():.4f} [{dr.min():.3f},{dr.max():.3f}] {int((dr > 0).sum())}/{len(ks)} z{np.median(dr) / sd:5.1f}")
            print(f"  {d:14} " + " | ".join(row))
print("\n===== razón de caídas en fidelidad, completo / otro esquema, por muestra (mín de las cinco)")
for ds in sorted({k[0] for k in R}):
    if (ds, "n3") not in R:
        continue
    r3 = R[(ds, "n3")]
    for d in sorted({d for _, d in r3 if d != "bueno"}):
        out = []
        for l in ORDEN[:-1]:
            if (ds, l) not in R:
                continue
            rr = R[(ds, l)]
            ks = [k for k in range(5) if (k, d) in rr and (k, d) in r3]
            a = np.array([r3[(k, "bueno")]["fidelity"] - r3[(k, d)]["fidelity"] for k in ks])
            b = np.array([rr[(k, "bueno")]["fidelity"] - rr[(k, d)]["fidelity"] for k in ks])
            q = a / np.maximum(b, 1e-4)
            out.append(f"{NOM[l]} media {a.mean() / max(b.mean(), 1e-4):5.1f} mín {q.min():5.1f} ({int((a > b).sum())}/{len(ks)})")
        print(f"  {ds:9} {d:14} " + " | ".join(out))
