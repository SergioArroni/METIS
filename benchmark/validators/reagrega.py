"""Reagregación de METIS sin volver a evaluar: los valores por métrica (armonizados, antes de calibrar) del
banco y las iteraciones de sus calibraciones bastan para recalcular la nota con otra armonización o con
otra taxonomía.

  load_records(paths)          registros del banco (métricas, notas, validez, defecto, familia del defecto)
  load_bounds(cache, ds)       CalibrationBounds del banco 3 de un dataset
  harmonize(variant)           función (métrica, valor armonizado) -> valor con otra armonización
  rebound(bounds, fn)          las mismas iteraciones de calibración pasadas por fn, anclas recalculadas
  aggregate(rec, bounds, fn)   notas de METIS para un registro (compuesto, familias, jerarquía)
  patch_taxonomy(fidelity)     sustituye la taxonomía de fidelidad en caliente
  z_table / counts             criterio del banco: z >= 3 frente a las buenas, mediana de las muestras
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from metis.application.aggregator import Aggregator
from metis.calibrate.core.bounds import CalibrationBounds
from metis.domain import taxonomy as tx
from metis.domain.entities import MetricResult
from metis.shared.normalization import METRIC_NORMALIZATION_MAP, METRIC_NORMALIZATION_PARAMS, NormalizationType

SD_FLOOR, ZD = 0.005, 3.0
KEY = ["dataset", "muestra", "defecto"]
NOTES = ["composite", "fidelity", "utility", "privacy", "validez"]


def load_records(paths):
    rows = []
    for p in paths:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if not r.get("error") and r.get("validador", "metis") == "metis":
                rows.append(r)
    seen, out = set(), []
    for r in rows:
        k = tuple(r[x] for x in KEY)
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def load_bounds(cache, ds):
    files = [f for f in sorted(Path(cache).glob(f"bounds_banco3_{ds}_calibration_*.json"), key=lambda p: p.stat().st_mtime)
             if "_plano_" not in f.name]
    return CalibrationBounds.load(str(files[-1]))


# ── armonización ─────────────────────────────────────────────────────────


def _discrepancy(metric, v):
    """Discrepancia d >= 0 de la que salió el valor armonizado v (None si la métrica ya es una similitud)."""
    t = METRIC_NORMALIZATION_MAP.get(metric)
    v = min(max(v, 1e-12), 1.0)
    if t == NormalizationType.BOUNDED_DISTANCE:
        return 1.0 - v
    if t == NormalizationType.UNBOUNDED_DISTANCE:
        return -METRIC_NORMALIZATION_PARAMS.get(metric, {}).get("scale", 1.0) * math.log(v)
    if t == NormalizationType.DELTA:
        return 1.0 / v - 1.0
    return None


def harmonize(variant):
    """metis | inversa | exponencial | lineal | s*<k> (solo las no acotadas, escala multiplicada por k)."""
    if variant == "metis":
        return lambda m, v: v
    if variant.startswith("s*"):
        k = float(variant[2:])

        def fn(m, v):
            if METRIC_NORMALIZATION_MAP.get(m) != NormalizationType.UNBOUNDED_DISTANCE or v is None:
                return v
            return float(min(max(v, 1e-12), 1.0) ** (1.0 / k))
        return fn
    g = {"inversa": lambda d: 1.0 / (1.0 + d), "exponencial": lambda d: math.exp(-d),
         "lineal": lambda d: max(0.0, 1.0 - d)}[variant]

    def fn(m, v):
        if v is None or not m.startswith("fidelity."):
            return v
        d = _discrepancy(m, v)
        return v if d is None else float(g(d))
    return fn


def rebound(bounds, fn):
    nb = CalibrationBounds()
    for m, e in bounds.bounds.items():
        lo = [fn(m, x) for x in e.get("lower_iterations") or []]
        up = [fn(m, x) for x in e.get("upper_iterations") or []]
        if lo and up:
            nb.set_bounds(m, lower_iterations=lo, upper_iterations=up, family=e["family"],
                          reference_lower=e.get("reference_lower"), reference_upper=e.get("reference_upper"))
        else:
            nb.bounds[m] = dict(e)
    nb.metadata = dict(getattr(bounds, "metadata", {}) or {})
    return nb


# ── agregación ───────────────────────────────────────────────────────────


def _results(rec, fn):
    out = []
    for m, v in (rec.get("metricas") or {}).items():
        fam = m.split(".", 1)[0]
        if v is None or not np.isfinite(v):
            out.append(MetricResult.not_applicable(m, fam, "no aplica"))
        else:
            out.append(MetricResult(id=m, value=float(fn(m, v)), details={}, family=fam))
    return out


def _hierarchy(h):
    out = {}
    for fam, fnode in (h or {}).items():
        for cat, cnode in ((fnode or {}).get("categories") or {}).items():
            out[f"{fam}.{cat}"] = cnode.get("score")
            for sub, snode in (cnode.get("subcategories") or {}).items():
                out[f"{fam}.{cat}.{sub}"] = snode.get("score")
    return out


WEIGHTS = None


def aggregate(rec, bounds, fn=lambda m, v: v):
    a = Aggregator(calibration_bounds=bounds).aggregate(_results(rec, fn), None, WEIGHTS)
    out = {f: a.get(f"{f}_score") for f in ("fidelity", "utility", "privacy")}
    out["composite"] = a.get("composite_score")
    out["validez"] = rec.get("validez")
    out["jerarquia"] = _hierarchy(a.get("hierarchy"))
    return out


# ── taxonomía ────────────────────────────────────────────────────────────


def patch_taxonomy(fidelity):
    tx.FIDELITY_METRICS.clear()
    tx.FIDELITY_METRICS.update(fidelity)
    tx._METRIC_HIERARCHY_MAP.clear()
    tx._METRIC_HIERARCHY_MAP.update(tx._build_metric_hierarchy_map())
    import metis.application.aggregator as agg
    agg._WEIGHT_KEYS, agg._HIERARCHY_WEIGHT_KEYS = agg._build_weight_keys()


# ── criterio del banco ───────────────────────────────────────────────────


def z_table(df, col):
    """{(dataset, defecto): z} con la mediana de las muestras, frente a las buenas del dataset."""
    out = {}
    for ds, d in df.groupby("dataset"):
        if col not in d:
            continue
        g = d[d.defecto == "bueno"][col].astype(float).dropna()
        if len(g) < 3:
            continue
        mu, sd = g.mean(), max(g.std(ddof=1), SD_FLOOR)
        for n, x in d[d.defecto != "bueno"].groupby("defecto"):
            v = x[col].astype(float).dropna()
            if len(v):
                out[(ds, n)] = float(np.median((mu - v) / sd))
    return out


def goods_fp(df, cols):
    """Tablas buenas marcadas por alguna de las notas (z >= 3 frente a las demás buenas)."""
    n = fp = 0
    for ds, d in df.groupby("dataset"):
        g = d[d.defecto == "bueno"]
        for i in range(len(g)):
            n += 1
            hit = False
            for c in cols:
                v = g[c].astype(float).to_numpy()
                if np.isnan(v[i]):
                    continue
                rest = np.delete(v, i)
                rest = rest[~np.isnan(rest)]
                if len(rest) >= 3 and (rest.mean() - v[i]) / max(rest.std(ddof=1), SD_FLOOR) >= ZD:
                    hit = True
            fp += hit
    return fp, n


def counts(df, families):
    zc = z_table(df, "composite")
    zn = {}
    for c in NOTES:
        for k, z in z_table(df, c).items():
            zn[k] = max(zn.get(k, -np.inf), z)
    zf = z_table(df, "fidelity")
    out = {"general": sum(z >= ZD for z in zc.values()), "alguna": sum(z >= ZD for z in zn.values()),
           "n": len(zc), "fid_fidelidad": sum(zf.get(k, -np.inf) >= ZD for k, f in families.items() if f == "fidelity"),
           "n_fid": sum(f == "fidelity" for f in families.values())}
    for fam in ("fidelity", "utility", "privacy", "validez"):
        ks = [k for k, f in families.items() if f == fam]
        out[f"{fam}_general"] = sum(zc.get(k, -np.inf) >= ZD for k in ks)
        out[f"{fam}_alguna"] = sum(zn.get(k, -np.inf) >= ZD for k in ks)
        out[f"{fam}_n"] = len(ks)
    out["fp"], out["n_buenas"] = goods_fp(df, NOTES)
    return out
