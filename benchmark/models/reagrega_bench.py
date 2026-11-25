"""Benchmark de la Sección 6 reagregado con la taxonomía instalada (§26), sin volver a evaluar.

Uso: reagrega_bench.py <dir_res_bench2> <dir_cache_bench2> <dir_salida>

Quita fidelity.missingness y los protocolos opcionales de utilidad (TTS, TRTS, TTRS): la nota es la de la
configuración del paper.  Escribe en <dir_salida>:
  por_semilla.csv     notas calibradas y en bruto por dataset, generador y semilla
  tablas.md           mediana (IQR) por dataset y generador; Friedman por dataset y dimensión; Friedman con
                      los datasets como bloques y la diferencia crítica de Nemenyi; ablación de la armonización
                      (§5.1) sobre el orden de los generadores
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, kendalltau, rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "validators"))
import reagrega as ra  # noqa: E402
from metis.application.aggregator import Aggregator
from metis.calibrate.core.bounds import CalibrationBounds

RES, CACHE, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
DROP = {"fidelity.missingness", "utility.tts", "utility.trts", "utility.ttrs"}
DS_ORDER = ["telco", "cardio", "rrhh", "airbnb", "adult"]
GEN_ORDER = ["real_data", "uniform_noise", "bootstrap", "smotenc", "delete_zero", "delete_mean", "bn", "cart",
             "gaussian_copula", "ctgan", "tvae", "dpctgan"]
DIMS = ["fidelity", "utility", "privacy", "composite"]

recs = []
for f in sorted(glob.glob(str(RES / "m_*.jsonl"))):
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        if not r.get("error"):
            r["metricas"] = {k: v for k, v in r["metricas"].items() if k not in DROP}
            recs.append(r)
BOUNDS = {}
for ds in DS_ORDER:
    files = sorted(CACHE.glob(f"bounds_bench2_{ds}_*.json"), key=lambda p: p.stat().st_mtime)
    BOUNDS[ds] = CalibrationBounds.load(str(files[-1]))


def notes(r, bounds, fn=lambda m, v: v):
    a = Aggregator(calibration_bounds=bounds).aggregate(ra._results(r, fn), None)
    return {d: a.get(f"{d}_score") for d in ("fidelity", "utility", "privacy")} | {"composite": a.get("composite_score")}


rows = []
for r in recs:
    cal = notes(r, BOUNDS[r["dataset"]])
    raw = notes(r, None)
    rows.append({"dataset": r["dataset"], "generator": r["generador"], "seed": r["semilla"], **cal,
                 **{f"{k}_raw": v for k, v in raw.items()}})
DF = pd.DataFrame(rows)
DF.to_csv(OUT / "por_semilla.csv", index=False)

L = []
w = L.append
fmt = lambda x: f"{x:.4f}"


def med_iqr(v):
    v = np.asarray(v, float)
    return f"{np.median(v):.4f} ({np.percentile(v, 75) - np.percentile(v, 25):.4f})"


w("## Mediana (IQR) calibrada por dataset y generador\n")
w("| dataset | generador | fidelidad | utilidad | privacidad | compuesto |")
w("|---|---|---|---|---|---|")
for ds in DS_ORDER:
    for g in GEN_ORDER:
        d = DF[(DF.dataset == ds) & (DF.generator == g)]
        w(f"| {ds} | {g} | " + " | ".join(med_iqr(d[c]) for c in DIMS) + " |")
w("\n## Mediana (IQR) en bruto (sin calibrar)\n")
w("| dataset | generador | fidelidad | utilidad | privacidad |")
w("|---|---|---|---|---|")
for ds in DS_ORDER:
    for g in GEN_ORDER:
        d = DF[(DF.dataset == ds) & (DF.generator == g)]
        w(f"| {ds} | {g} | " + " | ".join(med_iqr(d[c + "_raw"]) for c in ("fidelity", "utility", "privacy")) + " |")

w("\n## Friedman por dataset y dimensión (bloques = semillas)\n")
w("| dataset | dimensión | χ² | p | W | rango medio de los tres primeros |")
w("|---|---|---|---|---|---|")
for ds in DS_ORDER:
    for c in DIMS:
        piv = DF[DF.dataset == ds].pivot(index="seed", columns="generator", values=c)[GEN_ORDER]
        stat, pv = friedmanchisquare(*[piv[g].to_numpy() for g in GEN_ORDER])
        k, n = piv.shape[1], piv.shape[0]
        W = stat / (n * (k - 1))
        ranks = piv.apply(lambda row: pd.Series(rankdata(-row.to_numpy()), index=row.index), axis=1).mean()
        top = ", ".join(f"{g} {ranks[g]:.1f}" for g in ranks.sort_values().index[:3])
        w(f"| {ds} | {c} | {stat:.1f} | {pv:.2g} | {W:.3f} | {top} |")

w("\n## Friedman con los datasets como bloques (compuesto mediano) y Nemenyi\n")
med = DF.groupby(["dataset", "generator"]).composite.median().unstack()[GEN_ORDER].loc[DS_ORDER]
stat, pv = friedmanchisquare(*[med[g].to_numpy() for g in GEN_ORDER])
k, n = len(GEN_ORDER), len(DS_ORDER)
W = stat / (n * (k - 1))
ranks = med.apply(lambda row: pd.Series(rankdata(-row.to_numpy()), index=row.index), axis=1).mean()
cd = 3.268 * np.sqrt(k * (k + 1) / (6 * n))
w(f"χ² = {stat:.2f}, p = {pv:.2g}, W = {W:.3f}; CD de Nemenyi (α = 0,05, k = {k}, N = {n}) = {cd:.2f}\n")
w("| generador | rango medio |")
w("|---|---|")
for g in ranks.sort_values().index:
    w(f"| {g} | {ranks[g]:.2f} |")

w("\n## Puesto de real_data y bootstrap (compuesto mediano; 1 = mejor) y orden de los generadores\n")
for ds in DS_ORDER:
    order = med.loc[ds].sort_values(ascending=False)
    pos = {g: i + 1 for i, g in enumerate(order.index)}
    w(f"- {ds}: real_data {pos['real_data']}, bootstrap {pos['bootstrap']}; orden: " + ", ".join(
        f"{g} {order[g]:.3f}" for g in order.index))

w("\n## Ablación de la armonización (§5.1) sobre el orden de los generadores\n")
w("Compuesto mediano por generador con cada variante; ρ de Spearman y τ de Kendall con el orden de METIS, "
  "y si cambian el primero y el segundo generador (sin contar real_data).\n")
w("| variante | " + " | ".join(f"{ds} ρ / τ" for ds in DS_ORDER) + " | cambios en los dos primeros |")
w("|---|" + "---|" * (len(DS_ORDER) + 1))
base = None
for v in ["metis", "inversa", "exponencial", "lineal", "s*0.25", "s*0.5", "s*2", "s*4"]:
    fn = ra.harmonize(v)
    nb = {ds: ra.rebound(BOUNDS[ds], fn) for ds in DS_ORDER}
    vr = pd.DataFrame([{"dataset": r["dataset"], "generator": r["generador"], "composite": notes(r, nb[r["dataset"]], fn)["composite"]}
                       for r in recs])
    vm = vr.groupby(["dataset", "generator"]).composite.median().unstack()[GEN_ORDER].loc[DS_ORDER]
    if base is None:
        base = vm
    cells, changes = [], []
    for ds in DS_ORDER:
        rho = spearmanr(vm.loc[ds], base.loc[ds])[0]
        tau = kendalltau(vm.loc[ds], base.loc[ds])[0]
        cells.append(f"{rho:.3f} / {tau:.3f}")
        t_b = [g for g in base.loc[ds].sort_values(ascending=False).index if g != "real_data"][:2]
        t_v = [g for g in vm.loc[ds].sort_values(ascending=False).index if g != "real_data"][:2]
        if t_b != t_v:
            changes.append(f"{ds}: {'/'.join(t_b)} → {'/'.join(t_v)}")
    w(f"| {v} | " + " | ".join(cells) + f" | {'; '.join(changes) or '—'} |")
(OUT / "tablas.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("escrito", OUT)
