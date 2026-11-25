"""Cifras del banco sin los 15 casos de validez: Tabla 2 (combinaciones de operadores) y escalera de tipos."""
import json
from collections import defaultdict

import numpy as np

F = json.load(open("/home/ubuntu/strat/factorial/factorial.json"))
rows = F if isinstance(F, list) else F.get("rows") or F.get("resultados") or list(F.values())[0]
by = {r["combo"]: r for r in rows}
print("claves", [k for k in rows[0] if "valid" in k or k.endswith("_comp") or k in ("banco_general", "banco_n")])
base = ["mean", "mean", "ssd@3", "ssd@3"]
print("\nTabla 2 sin validez: nivel, operador, general, validez detectada por el compuesto, general sin validez, n sin validez")
for lv in range(4):
    for op in ("median", "mean", "ssd@3"):
        c = list(base)
        c[lv] = op
        r = by.get("/".join(c))
        if r is None:
            print("falta", c)
            continue
        v, nv = r.get("banco_validez_comp"), r.get("banco_validez_n")
        print(f"  L{lv + 1} {op:7} {r['banco_general']:4} {v} {r['banco_general'] - (v or 0):4} de {r['banco_n'] - (nv or 0)}"
              f"   fid {r.get('banco_fidelity_comp')}/{r.get('banco_fidelity_n')} util {r.get('banco_utility_comp')}/{r.get('banco_utility_n')} priv {r.get('banco_privacy_comp')}/{r.get('banco_privacy_n')}"
              f"  ruido uniforme {r.get('uniform_noise'):.3f}")
# ¿el orden de los operadores en cada nivel es el mismo en todas las combinaciones, sin validez?
gen = {c: r["banco_general"] - (r.get("banco_validez_comp") or 0) for c, r in by.items()}
print("\ncombinaciones", len(gen), "máximo sin validez", max(gen.values()), "METIS", gen["/".join(base)])
lam = {}
for c, g in gen.items():
    p = c.split("/")
    if p[:2] == ["mean", "mean"] and p[2] == p[3] and p[2].startswith("ssd"):
        lam[p[2]] = g
print("mean/mean/ssd@λ/ssd@λ sin validez:", dict(sorted(lam.items())))

# escalera de tipos sin validez
W = "/home/ubuntu/strat/v3/res_tax3/"
LV = {"0": "metis_n0.jsonl", "1": "metis_plano_todos.jsonl", "2": "metis_n2.jsonl", "3": "metis.jsonl"}
NOTAS = ["composite", "fidelity", "utility", "privacy", "validez"]


def stats(f, nota):
    by = defaultdict(dict)
    for l in open(W + f):
        r = json.loads(l)
        by[r["dataset"]][(r["muestra"], r["defecto"])] = r
    drop, sd, fam = {}, {}, {}
    for ds, d in by.items():
        g = [d[(k, "bueno")].get(nota) for k in range(5)]
        if any(x is None for x in g):
            continue
        g = np.array(g, float)
        sd[ds] = float(g.std(ddof=1))
        for n in sorted({n for _, n in d if n != "bueno"}):
            v = [d[(k, n)].get(nota) for k in range(5) if (k, n) in d]
            v = [x for x in v if x is not None]
            fam[(ds, n)] = d[(0, n)].get("familia_defecto")
            if v:
                drop[(ds, n)] = float(np.median(g.mean() - np.array(v, float)))
    return drop, sd, fam


S = {lv: {n: stats(f, n) for n in NOTAS} for lv, f in LV.items()}
fam = S["3"]["composite"][2]
cases = sorted(c for c in S["3"]["composite"][0] if fam[c] != "validez")
print("\nescalera sin validez, casos", len(cases))
sd3 = S["3"]["composite"][1]
for lv in LV:
    def z(nota, c, fix=None):
        drop, sd, _ = S[lv][nota]
        if c not in drop:
            return -9
        return drop[c] / max((fix or sd)[c[0]], 0.005)
    gen_ = sum(z("composite", c) >= 3 for c in cases)
    alg = sum(max(z(n, c) for n in NOTAS) >= 3 for c in cases)
    eq = sum(z("composite", c, sd3) >= 3 for c in cases)
    print(f"  n{lv} general {gen_}  más favorable {alg}  general con el ruido del completo {eq}")
