"""¿Mejora la nota si cada subcategoría agrupa métricas que miden lo mismo?  Reagrega el banco con
taxonomías de fidelidad alternativas (mismos valores por métrica y mismas calibraciones).

Uso: reorganiza.py <res/metis.jsonl> <res_estudio/metis_est_*.jsonl glob> <dir_cache> <salida.md>

  actual     la taxonomía de METIS v2
  propuesta  global = {mmd, energy_distance}; missingness en una categoría propia; outliers_coverage a
             colas; correlation_matrix a num–num; escalas partida en localización {delta_mean,
             delta_median, cohens_d} y dispersión {delta_iqr, delta_mad}
  sin_ausentes_aparte   la propuesta con missingness dentro de global
  global_A   global con subcategorías distancias {mmd, energy_distance} y ausentes {missingness};
             correlation_matrix a num–num, outliers_coverage a colas; escalas partida (cuatro categorías)
  global_B   las cinco de global en cuatro subcategorías (distancias, estructura, cobertura, ausentes),
             sin mover métricas a otras categorías; escalas partida
  ausentes_en_marginal  la propuesta con missingness como subcategoría de marginal (cuatro categorías)
  solo_global           la propuesta en global, con escalas como estaba
Por variante: detecciones del banco (128), buenas marcadas, defectos diluidos (alguna métrica con z >= 3
y ninguna nota) sobre banco y estudio, y la coherencia de cada grupo (ρ de Spearman interno frente al
externo, mediana entre datasets).
"""

from __future__ import annotations

import copy
import glob
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import reagrega as ra
from metis.domain import taxonomy as tx

RES, EST, CACHE, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
recs = ra.load_records([RES] + sorted(glob.glob(EST)))
bank = {(r["dataset"], r["defecto"]) for r in ra.load_records([RES])}
DATASETS = sorted({r["dataset"] for r in recs})
BOUNDS = {ds: ra.load_bounds(CACHE, ds) for ds in DATASETS}
FAM = {(r["dataset"], r["defecto"]): r.get("familia_defecto") for r in recs
       if r["defecto"] != "bueno" and (r["dataset"], r["defecto"]) in bank}
INSTALADA = copy.deepcopy(tx.FIDELITY_METRICS)
_f = lambda *ms: [f"fidelity.{m}" for m in ms]
ACTUAL = {  # METIS v2 antes de la reorganización (§25)
    "global": {"description": "", "metrics": _f("correlation_matrix", "mmd", "energy_distance", "outliers_coverage", "missingness")},
    "detection": {"description": "", "metrics": _f("discriminator")},
    "marginal": {"description": "", "subcategories": {
        "tails": _f("ks", "wasserstein", "anderson_darling", "hellinger", "kde_ise", "delta_exceedance"),
        "scales": _f("delta_mean", "delta_median", "delta_iqr", "delta_mad", "cohens_d"),
        "coverage": _f("tvd", "js", "kl", "psi", "entropy_delta", "gini_delta")}},
    "conditional": {"description": "", "subcategories": {
        "num_num": _f("pearson", "spearman", "dcor", "mi"),
        "num_cat": _f("eta_squared", "point_biserial", "kruskal_epsilon"),
        "cat_cat": _f("cramers_v", "theils_u", "chi2_stat")}},
}
M = ACTUAL["marginal"]["subcategories"]
C = ACTUAL["conditional"]["subcategories"]


def variant(name):
    if name.startswith(("T", "G")):
        f = copy.deepcopy(INSTALADA)
        subs = f["marginal"]["subcategories"]
        loc, disp = subs.pop("location"), subs.pop("dispersion")
        f["marginal"]["subcategories"] = {"tails": subs["tails"], "scales": loc + disp, "coverage": subs["coverage"]}
        del f["detection"]
        D, M, Dis = ["fidelity.mmd", "fidelity.energy_distance"], ["fidelity.missingness"], ["fidelity.discriminator"]
        if name == "T1":    # global: distancias, detección, ausentes
            f["global"] = {"description": "", "subcategories": {"distances": D, "detection": Dis, "missingness": M}}
        elif name == "T2":  # global: distancias, detección; ausentes como subcategoría de marginal
            f["global"] = {"description": "", "subcategories": {"distances": D, "detection": Dis}}
            f["marginal"]["subcategories"]["missingness"] = M
        elif name == "T3":  # global: multivariante {mmd, energy, discriminador} y ausentes
            f["global"] = {"description": "", "subcategories": {"multivariate": D + Dis, "missingness": M}}
        elif name == "T4":  # global: detección sola; distancias y ausentes como subcategorías de marginal
            f["global"] = {"description": "", "metrics": Dis}
            f["marginal"]["subcategories"]["distances"] = D
            f["marginal"]["subcategories"]["missingness"] = M
        elif name in ("G2", "G3"):  # global: distancias y detección; G2 ausentes en marginal, G3 sin ausentes
            f["global"] = {"description": "", "subcategories": {"distances": D, "detection": Dis}}
            if name == "G2":
                f["marginal"]["subcategories"]["missingness"] = M
        elif name in ("T1w2", "T1w3"):
            f["global"] = {"description": "", "subcategories": {"distances": D, "detection": Dis, "missingness": M}}
        elif name == "T8":  # global: detección y ausentes; distancias como subcategoría de marginal
            f["global"] = {"description": "", "subcategories": {"detection": Dis, "missingness": M}}
            f["marginal"]["subcategories"]["distances"] = D
        elif name == "T5":  # global sin subcategorías {mmd, energy, discriminador}; ausentes en marginal
            f["global"] = {"description": "", "metrics": D + Dis}
            f["marginal"]["subcategories"]["missingness"] = M
        return f
    if name == "A":
        return copy.deepcopy(INSTALADA)
    if name == "A_deteccion_en_global":
        f = copy.deepcopy(INSTALADA)
        del f["detection"]
        f["global"]["subcategories"]["detection"] = ["fidelity.discriminator"]
        return f
    if name == "A_escalas_juntas":
        f = copy.deepcopy(INSTALADA)
        subs = f["marginal"]["subcategories"]
        loc, disp = subs.pop("location"), subs.pop("dispersion")
        f["marginal"]["subcategories"] = {"tails": subs["tails"], "scales": loc + disp, "coverage": subs["coverage"]}
        return f
    f = copy.deepcopy(ACTUAL)
    if name == "actual":
        return f
    f["global"]["metrics"] = ["fidelity.mmd", "fidelity.energy_distance"]
    f["missingness"] = {"description": "Amount and pattern of missing values", "metrics": ["fidelity.missingness"]}
    f["marginal"]["subcategories"]["tails"] = M["tails"] + ["fidelity.outliers_coverage"]
    f["conditional"]["subcategories"]["num_num"] = C["num_num"] + ["fidelity.correlation_matrix"]
    if name != "solo_global":
        subs = f["marginal"]["subcategories"]
        del subs["scales"]
        subs["location"] = ["fidelity.delta_mean", "fidelity.delta_median", "fidelity.cohens_d"]
        subs["dispersion"] = ["fidelity.delta_iqr", "fidelity.delta_mad"]
    if name == "sin_ausentes_aparte":
        del f["missingness"]
        f["global"]["metrics"].append("fidelity.missingness")
    if name == "global_A":
        del f["missingness"]
        f["global"] = {"description": "", "subcategories": {
            "distances": ["fidelity.mmd", "fidelity.energy_distance"], "missingness": ["fidelity.missingness"]}}
    if name == "global_B":
        del f["missingness"]
        f["marginal"]["subcategories"]["tails"] = M["tails"]
        f["conditional"]["subcategories"]["num_num"] = C["num_num"]
        f["global"] = {"description": "", "subcategories": {
            "distances": ["fidelity.mmd", "fidelity.energy_distance"], "structure": ["fidelity.correlation_matrix"],
            "coverage": ["fidelity.outliers_coverage"], "missingness": ["fidelity.missingness"]}}
    if name == "ausentes_en_marginal":
        del f["missingness"]
        f["marginal"]["subcategories"]["missingness"] = ["fidelity.missingness"]
    return f


# ρ entre métricas (todas las tablas de cada dataset, mediana entre datasets)
MDF = pd.DataFrame([{**{k: r[k] for k in ra.KEY}, **(r.get("metricas") or {})} for r in recs])
METRICS = [c for c in MDF.columns if c.startswith(("fidelity.", "privacy.", "utility.")) and MDF[c].notna().any()]
_rho = {}
for ds in DATASETS:
    d = MDF[MDF.dataset == ds]
    for a, b in combinations(METRICS, 2):
        x = d[[a, b]].dropna().astype(float)
        if len(x) > 20 and x[a].nunique() > 2 and x[b].nunique() > 2:
            r = spearmanr(x[a], x[b])[0]
            if np.isfinite(r):
                _rho.setdefault(frozenset((a, b)), []).append(r)
RHO = {k: float(np.median(v)) for k, v in _rho.items() if len(v) >= 2}


def group_of(m):
    h = tx.get_metric_hierarchy(m)
    return ".".join(x for x in (h["family"], h["category"], h.get("subcategory")) if x)


L = [__doc__.split("Uso:")[0].strip(), ""]
w = L.append
summary, groups_txt = [], []
for name in ("A", "G2", "G3"):
    ra.patch_taxonomy(variant(name))
    ra.WEIGHTS = {"fidelity.global.detection": float(name[-1])} if name.startswith("T1w") else None
    rows = []
    for r in recs:
        if name == "G3":
            r = {**r, "metricas": {k: v for k, v in (r.get("metricas") or {}).items() if k != "fidelity.missingness"}}
        a = ra.aggregate(r, BOUNDS[r["dataset"]])
        rows.append({**{k: r[k] for k in ra.KEY}, **{k: a[k] for k in ra.NOTES}, **(r.get("metricas") or {})})
    df = pd.DataFrame(rows)
    dbank = df[[(ds, n) in bank or n == "bueno" for ds, n in zip(df.dataset, df.defecto)]]
    c = ra.counts(dbank, FAM)
    zn = {}
    for col in ra.NOTES:
        for k, z in ra.z_table(df, col).items():
            zn[k] = max(zn.get(k, -np.inf), z)
    zm = {}
    for m in METRICS:
        for k, z in ra.z_table(df, m).items():
            zm[k] = max(zm.get(k, -np.inf), z)
    dil = sorted(k for k in zm if zm[k] >= ra.ZD and zn.get(k, -np.inf) < ra.ZD)
    summary.append((name, c, dil))
    G = {}
    for m in METRICS:
        G.setdefault(group_of(m), []).append(m)
    groups_txt.append(f"\n### {name}\n")
    groups_txt.append("| grupo | métricas | ρ interno | ρ externo |")
    groups_txt.append("|---|---|---|---|")
    for g in sorted(G):
        if not g.startswith("fidelity"):
            continue
        ms = G[g]
        inner = [RHO[frozenset(p)] for p in combinations(ms, 2) if frozenset(p) in RHO]
        outer = [RHO[frozenset((a, o))] for a in ms for o in METRICS if o not in ms and frozenset((a, o)) in RHO]
        f2 = lambda v: f"{np.median(v):.2f}" if v else "—"
        groups_txt.append(f"| {g} | {', '.join(x.split('.', 1)[1] for x in ms)} | {f2(inner)} | {f2(outer)} |")

w("| variante | general | alguna | nota de fidelidad (72) | buenas marcadas | diluidos (banco + estudio) |")
w("|---|---|---|---|---|---|")
for name, c, dil in summary:
    w(f"| {name} | {c['general']}/{c['n']} | {c['alguna']}/{c['n']} | {c['fid_fidelidad']}/{c['n_fid']} | "
      f"{c['fp']}/{c['n_buenas']} | {len(dil)} |")
for name, c, dil in summary:
    w(f"\n- **{name}**, diluidos: " + ("; ".join(f"{a} {b}" for a, b in dil) or "—"))
w("\n## Coherencia de los grupos de fidelidad")
L += groups_txt
ra.patch_taxonomy(INSTALADA)
open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
print("escrito", OUT)
