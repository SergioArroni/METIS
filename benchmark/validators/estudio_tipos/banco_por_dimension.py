"""Banco de 142 casos: cada familia de defectos medida con la nota que cada validador da para esa dimensión."""
import json
from collections import defaultdict

import numpy as np

W = "/home/ubuntu/strat/"
SRC = {"metis": W + "v3/res_tax3/metis.jsonl", "sdmetrics": W + "v3/res_tax3/sdm2_todos.jsonl",
       "syntheval": W + "v3/res_tax3/syntheval.jsonl", "synthcity": W + "validadores/res_sc/synthcity.jsonl"}
FAM = {"metis": {"fidelity": "fidelity", "utility": "utility", "privacy": "privacy"},
       "sdmetrics": {"fidelity": "quality", "privacy": "dcr_overfitting"},
       "syntheval": {"fidelity": "u", "utility": "u", "privacy": "p"},
       "synthcity": {"fidelity": "stats+detection", "utility": "performance", "privacy": "privacy"}}
ALT = {"synthcity solo stats": {"fidelity": "stats", "utility": "performance", "privacy": "privacy"},
       "synthcity solo detection": {"fidelity": "detection", "utility": "performance", "privacy": "privacy"}}
SD, ZD = 0.005, 3.0
R = {}
for v, f in SRC.items():
    by = defaultdict(dict)
    for l in open(f):
        r = json.loads(l)
        if r.get("error") or (v != "synthcity" and r.get("validador") not in (v, None)):
            continue
        if v == "synthcity" and r.get("stats") is not None and r.get("detection") is not None:
            r["stats+detection"] = (r["stats"] + r["detection"]) / 2
        by[r["dataset"]][(r["muestra"], r["defecto"])] = r
    R[v] = by
famof = {(ds, n): r["familia_defecto"] for ds, d in R["metis"].items() for (k, n), r in d.items() if n != "bueno"}
cases = sorted(c for c, f in famof.items() if f in ("fidelity", "utility", "privacy"))
print("casos", len(cases), "| campos synthcity", [k for k in next(iter(R["synthcity"]["rrhh"].values())) if not k.startswith("m:")][:20])


def z(v, col, c):
    d = R[v][c[0]]
    g = [d[(k, "bueno")].get(col) for k in range(5) if (k, "bueno") in d]
    x = [d[(k, c[1])].get(col) for k in range(5) if (k, c[1]) in d]
    g = np.array([a for a in g if a is not None], float)
    x = np.array([a for a in x if a is not None], float)
    if len(g) < 3 or not len(x):
        return None
    return float(np.median((g.mean() - x) / max(g.std(ddof=1), SD)))


def tabla(name, v, fam):
    out = {}
    for f in ("fidelity", "utility", "privacy"):
        ks = [c for c in cases if famof[c] == f]
        col = fam.get(f)
        out[f] = (sum((z(v, col, c) or -9) >= ZD for c in ks) if col else 0, len(ks))
    print(f"{name:26}", "  ".join(f"{f} {a}/{b}" for f, (a, b) in out.items()), " total", sum(a for a, _ in out.values()))


for v in SRC:
    tabla(v, v, FAM[v])
for n, fam in ALT.items():
    tabla(n, "synthcity", fam)

print("\nbuenas marcadas con alguna nota de dimensión (z >= 3 frente a las otras cuatro)")
for v in SRC:
    fp = []
    for ds, d in R[v].items():
        for i in range(5):
            for col in set(FAM[v].values()):
                g = np.array([d[(k, "bueno")].get(col) if (k, "bueno") in d else np.nan for k in range(5)], float)
                rest = np.delete(g, i)
                rest = rest[~np.isnan(rest)]
                if len(rest) >= 3 and not np.isnan(g[i]) and (rest.mean() - g[i]) / max(rest.std(ddof=1), SD) >= ZD:
                    fp.append((ds, i, col, round(float(g[i]), 3)))
    print(f"  {v:10}", len({(a, b) for a, b, _, _ in fp}), fp)

DSS = ["adult", "cardio", "rrhh", "telco", "airbnb"]
LAD = {"desplazamiento (fidelity)": ("fidelity", [("desplazar_005", "0,05"), ("desplazar_010", "0,10"), ("desplazar_025", "0,25"), ("desplazar_numericas", "0,50")]),
       "copias (privacy)": ("privacy", [("copias_1", "1"), ("copias_2", "2"), ("copias_5", "5"), ("copias_10", "10"), ("copias_25", "25"), ("copias_100", "100")]),
       "etiquetas (utility)": ("utility", [("etiquetas_ruidosas_10", "10"), ("etiquetas_ruidosas", "30"), ("etiquetas_ruidosas_50", "50")])}
for title, (f, lad) in LAD.items():
    print("\ndosis mínima:", title, "[z por dosis]")
    for ds in DSS:
        row = []
        for v in SRC:
            col = FAM[v].get(f)
            if not col:
                row.append(f"{v} sin nota")
                continue
            zs = [z(v, col, (ds, n)) for n, _ in lad]
            mins = [dose for (n, dose), zz in zip(lad, zs) if zz is not None and zz >= ZD]
            row.append(f"{v} {mins[0] if mins else '—'} [" + " ".join("·" if a is None else f"{a:.1f}" for a in zs) + "]")
        print(f"  {ds:7}", " | ".join(row))

print("\nz con la nota de fidelidad de cada validador")
names = sorted({c[1] for c in cases})
print("  defectos:", [n for n in names if famof[("adult", n)] == "fidelity"] if all(("adult", n) in famof for n in names) else names)
for n in names:
    if any(t in n for t in ("num_cat", "numcat", "ordinal_cerca", "ordinal_lejos", "mixta")):
        for ds in DSS:
            if (ds, n) in famof:
                print(f"  {n:22} {ds:7}", "  ".join(f"{v} {('—' if z(v, FAM[v]['fidelity'], (ds, n)) is None else format(z(v, FAM[v]['fidelity'], (ds, n)), '.1f'))}" for v in SRC))
