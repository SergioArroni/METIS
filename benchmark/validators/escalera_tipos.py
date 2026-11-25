"""Escalera de la tipología: detecciones del banco por nivel (0 formato, 1 plano, 2 sin ordinales, 3 METIS)."""
import json
from collections import defaultdict

import numpy as np

W = "/home/ubuntu/strat/v3/res_tax3/"
LV = {"0": "metis_n0.jsonl", "1": "metis_plano_todos.jsonl", "2": "metis_n2.jsonl", "3": "metis.jsonl"}
NOTAS = ["composite", "fidelity", "utility", "privacy", "validez"]
FLOOR = 0.005


def load(f):
    by = defaultdict(dict)
    for l in open(W + f):
        r = json.loads(l)
        by[r["dataset"]][(r["muestra"], r["defecto"])] = r
    return by


def stats(by, nota):
    """{(ds, defecto): caída mediana}, {ds: sd de las buenas}"""
    drop, sd = {}, {}
    for ds, d in by.items():
        g = [d[(k, "bueno")].get(nota) for k in range(5)]
        if any(x is None for x in g):
            continue
        g = np.array(g, float)
        sd[ds] = float(g.std(ddof=1))
        for n in sorted({n for _, n in d if n != "bueno"}):
            v = [d[(k, n)].get(nota) for k in range(5) if (k, n) in d]
            v = [x for x in v if x is not None]
            if v:
                drop[(ds, n)] = float(np.median(g.mean() - np.array(v, float)))
    return drop, sd


S = {lv: {n: stats(load(f), n) for n in NOTAS} for lv, f in LV.items()}
fam = {}
for l in open(W + "metis.jsonl"):
    r = json.loads(l)
    fam[(r["dataset"], r["defecto"])] = r.get("familia_defecto")
cases = sorted(S["3"]["composite"][0])
dss = sorted({c[0] for c in cases})
print("casos", len(cases))


def z(lv, nota, c, sdfix=None):
    drop, sd = S[lv][nota]
    if c not in drop:
        return None
    s = sdfix[c[0]] if sdfix else sd[c[0]]
    return drop[c] / max(s, FLOOR)


def det(lv, sdfix=None, alguna=False):
    out = set()
    for c in cases:
        zs = [z(lv, n, c, sdfix if n == "composite" else None) for n in (NOTAS if alguna else ["composite"])]
        zs = [x for x in zs if x is not None]
        if zs and max(zs) >= 3:
            out.add(c)
    return out


print("\nsd de las buenas (nota compuesta) por dataset y nivel")
for ds in dss:
    print(f"  {ds:7}", "  ".join(f"n{lv} {S[lv]['composite'][1][ds]:.4f}" for lv in LV))
sdmin = {ds: min(S[lv]["composite"][1][ds] for lv in LV) for ds in dss}
sdmax = {ds: max(S[lv]["composite"][1][ds] for lv in LV) for ds in dss}
sd3 = {ds: S["3"]["composite"][1][ds] for ds in dss}
D = {}
print("\nnivel  general  alguna  general(ruido mínimo)  general(ruido máximo)  general(ruido del nivel 3)")
for lv in LV:
    D[lv] = det(lv)
    print(f"  n{lv}   {len(D[lv]):4}    {len(det(lv, alguna=True)):4}        {len(det(lv, sdmin)):4}"
          f"                 {len(det(lv, sdmax)):4}                 {len(det(lv, sd3)):4}")
print("\ngeneral por familia de defecto")
for f in sorted({v for v in fam.values() if v}):
    cs = [c for c in cases if fam.get(c) == f]
    print(f"  {f:10} {len(cs):3}  " + "  ".join(f"n{lv} {sum(c in D[lv] for c in cs):3}" for lv in LV)
          + "   | ruido mínimo " + "  ".join(f"n{lv} {sum(c in det(lv, sdmin) for c in cs):3}" for lv in LV))
print("\ngeneral por dataset")
for ds in dss:
    cs = [c for c in cases if c[0] == ds]
    print(f"  {ds:7} {len(cs):3}  " + "  ".join(f"n{lv} {sum(c in D[lv] for c in cs):3}" for lv in LV))
print("\ncaída mediana de la nota compuesta (media sobre casos) por nivel")
for lv in LV:
    print(f"  n{lv} {np.mean([S[lv]['composite'][0][c] for c in cases]):.4f}")
print("\ncasos cuyo veredicto cambia entre niveles (z general por nivel)")
for c in cases:
    v = [c in D[lv] for lv in LV]
    if len(set(v)) > 1:
        print(f"  {c[0]:7} {c[1]:30}", "  ".join(f"{z(lv, 'composite', c):7.1f}" for lv in LV))
print("\ndefectos de ordinales y discretas: z general y caída por nivel")
for c in cases:
    if any(t in c[1] for t in ("ordinal", "discret", "nivel", "paso", "redonde", "entero")):
        print(f"  {c[0]:7} {c[1]:30}", "  ".join(f"{z(lv, 'composite', c):6.1f}/{S[lv]['composite'][0][c]:.3f}" for lv in LV))
