"""Prueba completa de operadores de agregación: mediana, media y ssd con λ = 0,5, 1, 2, 3, 4 y 5 en cada nivel.

Uso: factorial_lambda.py <dir_v3> <dir_cache_bancos> <dir_res_bench2> <dir_cache_bench2> <salida_dir> [n_procesos]

8^4 = 4096 combinaciones (L1, L2, L3, L4), la misma en las tres dimensiones, reagregando sin volver a evaluar
desde los valores por métrica guardados y las mismas calibraciones:
  banco      128 defectos del banco de validadores + los del estudio de métricas (res_tax3, res_estudio_tax3),
             detección z >= 3 frente a las 5 buenas (mediana de 5 muestras), con el compuesto y con la nota de la
             dimensión del defecto; buenas marcadas; ruido de las buenas
  pu         banco de privacidad y utilidad (pu/res): privacidad y utilidad con la nota de su dimensión
  bench      benchmark (240 tablas): ρ de Spearman del orden de generadores por compuesto mediano con el de METIS,
             si coinciden los dos primeros, real_data y uniform_noise, y fracción de tablas cuyo compuesto es
             su peor dimensión
  colapso    una tabla con 1 en dos dimensiones y 0 en la otra frente a otra con 0,5 en las tres
"""
from __future__ import annotations

import glob
import itertools
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, sys.argv[1])
import reagrega as ra  # noqa: E402
from metis.application.aggregator import Aggregator  # noqa: E402
from metis.calibrate.core.bounds import CalibrationBounds  # noqa: E402
from metis.shared.aggregation_registry import get_aggregation_function  # noqa: E402

V3, CACHE, BRES, BCACHE, OUT = (Path(a) for a in sys.argv[1:6])
NPROC = int(sys.argv[6]) if len(sys.argv) > 6 else 8
LAMBDAS = [0.5, 1, 2, 3, 4, 5]
OPS = ["median", "mean"] + [f"ssd@{l:g}" for l in LAMBDAS]
DEFAULT = ("mean", "mean", "ssd@3", "ssd@3")
DROP = {"fidelity.missingness", "utility.tts", "utility.trts", "utility.ttrs"}
DIMS = ["fidelity", "utility", "privacy"]


def read(paths):
    out = []
    for p in paths:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r.get("error") or r.get("validador", "metis") != "metis":
                continue
            r["metricas"] = {k: v for k, v in (r.get("metricas") or {}).items() if k not in DROP}
            out.append(r)
    return out


def dedup(recs, key):
    seen, out = set(), []
    for r in recs:
        k = tuple(r[x] for x in key)
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def bounds(cache, prefix, ds):
    files = [f for f in sorted(Path(cache).glob(f"bounds_{prefix}_{ds}_*.json"), key=lambda p: p.stat().st_mtime)
             if "_plano_" not in f.name]
    return CalibrationBounds.load(str(files[-1]))


BANK = dedup(read([V3 / "res_tax3" / "metis.jsonl"] + sorted((V3 / "res_estudio_tax3").glob("metis_est_*.jsonl"))),
             ["dataset", "muestra", "defecto"])
PU = dedup(read(sorted((V3 / "pu" / "res").glob("m_*.jsonl"))), ["dataset", "muestra", "defecto"])
BENCH = read(sorted(BRES.glob("m_*.jsonl")))
B_BANK = {ds: bounds(CACHE, "banco3", ds) for ds in {r["dataset"] for r in BANK}}
B_PU = {ds: bounds(CACHE, "banco4", ds) for ds in {r["dataset"] for r in PU}}
B_BENCH = {ds: bounds(BCACHE, "bench2", ds) for ds in {r["dataset"] for r in BENCH}}


class Agg(Aggregator):
    """ssd@λ: ssd con su propio λ en ese nivel."""

    def _build_aggregator(self, name):
        if name.startswith("ssd@"):
            return get_aggregation_function("ssd", risk_aversion=float(name[4:]))
        return super()._build_aggregator(name)


def build(name):
    if name.startswith("ssd@"):
        return get_aggregation_function("ssd", risk_aversion=float(name[4:]))
    return get_aggregation_function(name)


def make_agg(b, combo):
    a = Agg(calibration_bounds=b)
    l1, l2, l3, l4 = combo
    for fam in DIMS:
        setattr(a, f"{fam}_aggs", {"level_1": l1, "level_2": l2, "level_3": l3})
    a._family_agg_funcs = {}
    a.composite_agg_name = l4
    a.agg_layers_1_2_name, a.agg_layers_3_4_name = l2, l3
    a.agg_func_1_2 = a._build_aggregator(l2)
    a.agg_func_3_4 = a._build_aggregator(l3)
    a.agg_func_composite = a._build_aggregator(l4)
    return a


def notes(recs, bmap, combo, extra):
    aggs = {ds: make_agg(b, combo) for ds, b in bmap.items()}
    rows = []
    for r in recs:
        a = aggs[r["dataset"]].aggregate(ra._results(r, lambda m, v: v), None)
        rows.append({**{k: r.get(k) for k in extra}, "composite": a.get("composite_score"),
                     **{d: a.get(f"{d}_score") for d in DIMS}, "stored": r.get("composite")})
    return pd.DataFrame(rows)


def detect(df, fams):
    zc = ra.z_table(df, "composite")
    zd = {d: ra.z_table(df, d) for d in DIMS}
    out = {}
    for fam in ("fidelity", "utility", "privacy", "validez"):
        ks = [k for k, f in fams.items() if f == fam]
        out[f"{fam}_n"] = len(ks)
        out[f"{fam}_comp"] = sum(zc.get(k, -np.inf) >= ra.ZD for k in ks)
        if fam in zd:
            out[f"{fam}_dim"] = sum(zd[fam].get(k, -np.inf) >= ra.ZD for k in ks)
    out["n"] = len(zc)
    out["general"] = sum(z >= ra.ZD for z in zc.values())
    out["fp"], out["n_buenas"] = ra.goods_fp(df.assign(validez=np.nan), ["composite"] + DIMS)
    g = df[df.defecto == "bueno"]
    out["ruido"] = float(g.groupby("dataset").composite.std(ddof=1).mean())
    return out


FAM_BANK = {(r["dataset"], r["defecto"]): r.get("familia_defecto") for r in BANK if r["defecto"] != "bueno"}
FAM_PU = {(r["dataset"], r["defecto"]): r.get("familia_defecto") for r in PU if r["defecto"] != "bueno"}


def bench_order(df):
    med = df.groupby(["dataset", "generador"]).composite.median()
    return {ds: med[ds] for ds in df.dataset.unique()}


def collapse(combo):
    _, _, l3, l4 = combo
    f = build(l4)
    return f([1.0, 1.0, 0.0]) <= f([0.5, 0.5, 0.5])


def run(combo):
    bank = notes(BANK, B_BANK, combo, ["dataset", "muestra", "defecto"])
    pu = notes(PU, B_PU, combo, ["dataset", "muestra", "defecto"])
    bench = notes(BENCH, B_BENCH, combo, ["dataset", "generador", "semilla"])
    res = {"combo": "/".join(combo)}
    res.update({f"banco_{k}": v for k, v in detect(bank, FAM_BANK).items()})
    res.update({f"pu_{k}": v for k, v in detect(pu, FAM_PU).items()})
    med = bench.groupby(["dataset", "generador"]).composite.median().unstack(0)
    res["real_data"] = float(med.loc["real_data"].min()) if "real_data" in med.index else np.nan
    res["uniform_noise"] = float(med.loc["uniform_noise"].max()) if "uniform_noise" in med.index else np.nan
    dims = bench[DIMS].astype(float)
    worst = dims.min(axis=1, skipna=True)
    res["peor_dim"] = float((np.abs(bench.composite.astype(float) - worst) < 1e-3).mean())
    res["colapso_ok"] = bool(collapse(combo))
    res["_bench_med"] = med.to_dict()
    res["_ctrl_bank"] = float(np.nanmax(np.abs(bank.composite.astype(float) - bank.stored.astype(float))))
    return res


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    combos = list(itertools.product(OPS, repeat=4))
    with Pool(NPROC) as p:
        results = p.map(run, combos, chunksize=1)
    base = next(r for r in results if r["combo"] == "/".join(DEFAULT))
    bmed = pd.DataFrame(base["_bench_med"])
    for r in results:
        m = pd.DataFrame(r["_bench_med"])
        rhos, top2 = [], True
        for ds in bmed.columns:
            a, b = bmed[ds].drop(["real_data", "uniform_noise"], errors="ignore"), m[ds].reindex(bmed[ds].index)
            b = b.drop(["real_data", "uniform_noise"], errors="ignore")
            rhos.append(spearmanr(a, b)[0])
            top2 &= set(a.nlargest(2).index) == set(b.nlargest(2).index)
        r["rho_min"], r["top2_igual"] = float(np.nanmin(rhos)), bool(top2)
    json.dump([{k: v for k, v in r.items() if not k.startswith("_bench")} for r in results],
              open(OUT / "factorial.json", "w"), indent=1, default=float)
    cols = ["combo", "banco_general", "banco_n", "banco_fidelity_dim", "banco_fidelity_n", "banco_privacy_dim",
            "banco_privacy_n", "banco_utility_dim", "banco_utility_n", "banco_fp", "banco_n_buenas", "banco_ruido",
            "pu_privacy_dim", "pu_privacy_n", "pu_utility_dim", "pu_utility_n", "pu_fp", "pu_n_buenas",
            "rho_min", "top2_igual", "real_data", "uniform_noise", "peor_dim", "colapso_ok", "_ctrl_bank"]
    df = pd.DataFrame(results)[cols].sort_values(["banco_general", "banco_fidelity_dim"], ascending=False)
    df.to_csv(OUT / "factorial.csv", index=False)
    L = [__doc__.split("Uso:")[0].strip(), "",
         f"Control: con {'/'.join(DEFAULT)} la reagregación reproduce el compuesto guardado del banco con un error "
         f"máximo de {base['_ctrl_bank']:.2e}.", "",
         "| L1/L2/L3/L4 | banco general | fidelidad (su nota) | privacidad | utilidad | buenas marcadas | ruido buenas | "
         "pu privacidad | pu utilidad | pu buenas | ρ mín. bench | top-2 igual | real_data mín. | ruido máx. | "
         "compuesto = peor dim. | colapso evitado |",
         "|" + "---|" * 16]
    for _, r in df.iterrows():
        L.append(f"| {r.combo}{' **(METIS)**' if r.combo == '/'.join(DEFAULT) else ''} | {r.banco_general}/{r.banco_n} | "
                 f"{r.banco_fidelity_dim}/{r.banco_fidelity_n} | {r.banco_privacy_dim}/{r.banco_privacy_n} | "
                 f"{r.banco_utility_dim}/{r.banco_utility_n} | {r.banco_fp}/{r.banco_n_buenas} | {r.banco_ruido:.4f} | "
                 f"{r.pu_privacy_dim}/{r.pu_privacy_n} | {r.pu_utility_dim}/{r.pu_utility_n} | {r.pu_fp}/{r.pu_n_buenas} | "
                 f"{r.rho_min:.3f} | {'sí' if r.top2_igual else 'no'} | {r.real_data:.3f} | {r.uniform_noise:.3f} | "
                 f"{r.peor_dim:.0%} | {'sí' if r.colapso_ok else 'no'} |")
    (OUT / "factorial.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("escrito", OUT / "factorial.md")
