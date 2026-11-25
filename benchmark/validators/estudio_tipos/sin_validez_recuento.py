"""1) arregla la elección de la calibración; 2) banco sin validez; 3) semillas; 4) diagnóstico RRHH."""
import glob
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

S = Path("/home/ubuntu/strat")
W = S / "v3"
C = "/home/ubuntu/doc-split/metis/metis/calibrate/cache"
PY = sys.executable

# 1) patrón estricto: solo bounds_banco3_<ds>_calibration_* (o _plano_calibration_*)
for f in ("reagrega.py", "reagrega_banco.py", "analiza_aviso.py", "analiza_matices.py"):
    p = W / f
    s = p.read_text(encoding="utf-8")
    n = s.replace('bounds_banco3_{ds}_plano_*.json', 'bounds_banco3_{ds}_plano_calibration_*.json').replace(
        'bounds_banco3_{ds}_*.json', 'bounds_banco3_{ds}_calibration_*.json')
    if n != s:
        (W / (f + ".bak17")).write_text(s, encoding="utf-8")
        p.write_text(n, encoding="utf-8")
    print("patrón", f, "cambiado" if n != s else "ya estaba")

env = {"PYTHONPATH": f"/home/ubuntu/doc-split/metis:{W}", "OMP_NUM_THREADS": "1", "PATH": "/usr/bin:/bin"}


def run(*a, cwd=W):
    r = subprocess.run([PY, *map(str, a)], cwd=cwd, env=env, capture_output=True, text=True)
    print("  $", " ".join(map(str, a))[:150], "->", r.returncode, r.stdout.strip()[-200:], r.stderr.strip()[-600:])


# comprobación: reagregar de nuevo con el patrón arreglado da lo mismo que lo que usa el paper
def same(a, b):
    A = {(r["dataset"], r["muestra"], r["defecto"]): r for r in map(json.loads, open(a)) if not r.get("error")}
    B = {(r["dataset"], r["muestra"], r["defecto"]): r for r in map(json.loads, open(b)) if not r.get("error")}
    d = max((abs((A[k].get(c) or 0) - (B[k].get(c) or 0)) for k in A for c in ("composite", "fidelity", "utility", "privacy")), default=-1)
    return len(A), len(B), d


SV = W / "res_sv"
SV.mkdir(exist_ok=True)
run("reagrega_banco.py", C, SV / "chk_metis.jsonl", "res/metis.jsonl")
print("res_tax3/metis.jsonl frente a reagregado ahora:", same(W / "res_tax3/metis.jsonl", SV / "chk_metis.jsonl"))
for s in (1000, 2000):
    R = S / "sem" / f"s{s}"
    run("reagrega_banco.py", C, R / "chk_metis.jsonl", R / "ev_metis.jsonl")
    print(f"sem s{s}:", same(R / "metis.jsonl", R / "chk_metis.jsonl"))
for lv in ("n0", "n2"):
    print(f"bruto {lv} frente a res_tax3 {lv} (contaminado si > 0):", same(W / f"res/metis_{lv}.jsonl", W / f"res_tax3/metis_{lv}.jsonl"))

# 2) banco sin validez
n = 0
with open(SV / "metis.jsonl", "w") as fh:
    for l in open(W / "res_tax3/metis.jsonl"):
        r = json.loads(l)
        if r.get("familia_defecto") == "validez":
            continue
        fh.write(l)
        n += 1
print("res_sv/metis.jsonl", n)
run("reorganiza.py", "res_sv/metis.jsonl", "res_estudio_tax3/metis_est_*.jsonl", C, "informe_reorganiza_sv.md")
run("analiza_aviso.py", "res_sv", "res_estudio_tax3", C, "informe_aviso_sv.md")
print(open(W / "informe_reorganiza_sv.md").read())
print("\n".join(open(W / "informe_aviso_sv.md").read().split("\n")[4:22]))

# ρ sueltas (texto del Apéndice D): sin validez; missingness desde el bruto
from scipy.stats import spearmanr


def recs_of(main, est):
    out = [json.loads(l) for l in open(main)]
    for f in glob.glob(est):
        out += [json.loads(l) for l in open(f)]
    return [r for r in out if not r.get("error") and r.get("familia_defecto") != "validez" and r.get("metricas")]


G = {"num_num": ["pearson", "spearman", "dcor", "mi"],
     "tails": ["ks", "wasserstein", "anderson_darling", "hellinger", "kde_ise", "delta_exceedance"],
     "dist": ["mmd", "energy_distance"]}


def rho_fn(by):
    def rho(ds, a, b):
        x = np.array([np.nan if m.get(a) is None else m.get(a) for m in by[ds]], float)
        y = np.array([np.nan if m.get(b) is None else m.get(b) for m in by[ds]], float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 10 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
            return np.nan
        return spearmanr(x[ok], y[ok]).statistic
    return rho


by = defaultdict(list)
for r in recs_of(SV / "metis.jsonl", str(W / "res_estudio_tax3/metis_est_*.jsonl")):
    by[r["dataset"]].append(r["metricas"])
print("tablas por dataset", {d: len(v) for d, v in by.items()})
rho = rho_fn(by)
for met, grp in (("correlation_matrix", "num_num"), ("outliers_coverage", "tails"), ("mmd", "dist")):
    per = []
    for ds in by:
        v = [rho(ds, "fidelity." + met, "fidelity." + o) for o in G[grp] if o != met]
        v = [x for x in v if np.isfinite(x)]
        if v:
            per.append(np.median(v))
    print("rho", met, "con", grp, round(float(np.median(per)), 2), [round(float(x), 2) for x in per])
byr = defaultdict(list)
for r in recs_of(W / "res/metis.jsonl", str(W / "res_estudio/metis_est_*.jsonl")):
    byr[r["dataset"]].append(r["metricas"])
rr = rho_fn(byr)
allm = sorted({m for v in byr.values() for x in v for m in x if m.startswith("fidelity.")})
mx = {}
for o in allm:
    if o == "fidelity.missingness":
        continue
    v = [rr(ds, "fidelity.missingness", o) for ds in byr]
    v = [x for x in v if np.isfinite(x)]
    if v:
        mx[o] = float(np.median(v))
print("missingness: datasets con valor", [ds for ds in byr if any(m.get("fidelity.missingness") is not None for m in byr[ds])],
      "max rho mediana", sorted(((round(v, 2), k) for k, v in mx.items()), reverse=True)[:4])

# 3) semillas: recuento con la nota de cada dimensión sobre los 142
FAM = {"metis": {"fidelity": "fidelity", "utility": "utility", "privacy": "privacy"},
       "sdmetrics": {"fidelity": "quality", "privacy": "dcr_overfitting"},
       "syntheval": {"fidelity": "u", "utility": "u", "privacy": "p"},
       "synthcity": {"fidelity": "stats+detection", "utility": "performance", "privacy": "privacy"}}
SD, ZD = 0.005, 3.0


def load(v, f):
    by = defaultdict(dict)
    for l in open(f):
        r = json.loads(l)
        if r.get("error") or (v != "synthcity" and r.get("validador") not in (v, None)):
            continue
        if v == "synthcity" and r.get("stats") is not None and r.get("detection") is not None:
            r["stats+detection"] = (r["stats"] + r["detection"]) / 2
        by[r["dataset"]][(r["muestra"], r["defecto"])] = r
    return by


def z(d, col, n):
    g = np.array([d[(k, "bueno")][col] for k in range(5) if (k, "bueno") in d and d[(k, "bueno")].get(col) is not None], float)
    x = np.array([d[(k, n)][col] for k in range(5) if (k, n) in d and d[(k, n)].get(col) is not None], float)
    if len(g) < 3 or not len(x):
        return None
    return float(np.median((g.mean() - x) / max(g.std(ddof=1), SD)))


base = load("metis", W / "res_tax3/metis.jsonl")
famof = {(ds, n): r["familia_defecto"] for ds, d in base.items() for (k, n), r in d.items() if n != "bueno"}
cases = sorted(c for c, f in famof.items() if f in FAM["metis"])
DIRS = {"base": {"metis": W / "res_tax3/metis.jsonl", "sdmetrics": W / "res_tax3/sdm2_todos.jsonl",
                 "syntheval": W / "res_tax3/syntheval.jsonl", "synthcity": S / "validadores/res_sc/synthcity.jsonl"}}
for s in (1000, 2000):
    R = S / "sem" / f"s{s}"
    DIRS[f"s{s}"] = {"metis": R / "metis.jsonl", "sdmetrics": R / "sdm2_todos.jsonl", "syntheval": R / "syntheval.jsonl", "synthcity": R / "synthcity.jsonl"}
det = {}
for tag, src in DIRS.items():
    for v, f in src.items():
        R = load(v, f)
        s = set()
        miss = 0
        for c in cases:
            col = FAM[v].get(famof[c])
            if not col:
                continue
            zz = z(R.get(c[0], {}), col, c[1])
            miss += zz is None
            if zz is not None and zz >= ZD:
                s.add(c)
        det[(tag, v)] = s
        print(f"semilla {tag:6} {v:10} detecta {len(s):3} de {len(cases)} (sin dato {miss})")
for v in FAM:
    b = det[("base", v)]
    print(v, "recuentos", [len(det[(t, v)]) for t in DIRS], "| casos que cambian frente a base",
          [sorted(b ^ det[(t, v)]) for t in ("s1000", "s2000")])

# 4) RRHH: etiquetas ruidosas con cada esquema
print("\n== RRHH etiquetas ruidosas")
LV = {"completo": ("res/metis.jsonl", "bounds_banco3_rrhh_calibration_*"), "n2 (ordinales nominales)": ("res/metis_n2.jsonl", "bounds_banco3_rrhh_n2_calibration_*"),
      "n0 (formato)": ("res/metis_n0.jsonl", "bounds_banco3_rrhh_n0_calibration_*"), "plano": ("res/metis_plano_todos.jsonl", "bounds_banco3_rrhh_plano_calibration_*")}
for name, (f, pat) in LV.items():
    R = load("metis", W / f) if "plano" not in f else None
    if R is None:
        R = defaultdict(dict)
        for l in open(W / f):
            r = json.loads(l)
            if not r.get("error"):
                R[r["dataset"]][(r["muestra"], r["defecto"])] = r
    d = R["rrhh"]
    bf = sorted(glob.glob(f"{C}/{pat}"))
    b = json.load(open(bf[-1]))["bounds"]
    um = sorted({m for (k, n), r in d.items() for m in (r.get("metricas") or {}) if m.startswith("utility.")})
    print(f"\n-- {name} | {Path(bf[-1]).name[:60]}")
    for m in um:
        e = b.get(m, {})
        print("   cota", m, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in e.items() if k in ("calibratable", "lower", "upper", "lower_mean", "upper_mean", "lower_sd", "upper_sd", "reason", "lower_bound", "upper_bound")})
    for n in ("bueno", "etiquetas_ruidosas_10", "etiquetas_ruidosas", "etiquetas_ruidosas_50"):
        rows = [d[(k, n)] for k in range(5) if (k, n) in d]
        if not rows:
            continue
        print(f"   {n:24} utilidad", [round(r["utility"], 3) if r.get("utility") is not None else None for r in rows],
              "| fid", [round(r["fidelity"], 3) for r in rows])
        for m in um:
            print(f"      {m:28}", [None if r["metricas"].get(m) is None else round(r["metricas"][m], 4) for r in rows])
    print("   z utilidad", {n: (None if z(d, "utility", n) is None else round(z(d, "utility", n), 1)) for n in ("etiquetas_ruidosas_10", "etiquetas_ruidosas", "etiquetas_ruidosas_50")})
