"""Análisis del banco de defectos rediseñado (E1-E4, validez, ablaciones).

Uso: analiza2.py <dir_res> <salida.md>

Entradas (en dir_res): metis.jsonl, metis_plano_<ds>.jsonl, sdm2_<ds>.jsonl (SDMetrics con
diagnóstico), syntheval.jsonl.

Detección: z = (media de las 5 buenas - nota del caso) / sd de las buenas (suelo 0,005), mediana
sobre las 5 muestras; detectado si z >= 3.  Cada validador se mide con:
  general  su nota general (METIS composite, SDMetrics Quality, SynthEval media de todo)
  familia  su nota de la familia del defecto (None si no publica ninguna)
  alguna   el máximo z entre todas las notas agregadas que publica (lo más generoso para cada uno)
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RES, OUT = Path(sys.argv[1]), Path(sys.argv[2])
SD_FLOOR, Z_DET = 0.005, 3.0


def load(pattern, tag=None):
    rows = []
    for f in sorted(glob.glob(str(RES / pattern))):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r.get("error"):
                continue
            if tag:
                r["validador"] = tag
            rows.append(r)
    return rows


rows = load("metis.jsonl") + load("metis_plano_*.jsonl") + load("sdm2_*.jsonl", "sdmetrics") + load("syntheval.jsonl")

# E2b: METIS reducido a las métricas que usa SDMetrics (Column Shapes: KS y TVD; Column Pair Trends:
# correlación y contingencia) y la media plana de todas sus métricas de fidelidad.
SDM_LIKE = ["fidelity.ks", "fidelity.tvd", "fidelity.pearson", "fidelity.cramers_v"]
extra = []
for r in rows:
    if r["validador"] != "metis":
        continue
    m = r.get("metricas") or {}
    sub = [m[k] for k in SDM_LIKE if m.get(k) is not None]
    fid = [v for k, v in m.items() if k.startswith("fidelity.") and v is not None]
    extra.append({**{k: r[k] for k in ("dataset", "muestra", "defecto", "familia_defecto")},
                  "validador": "metis_sdm_metricas", "composite": np.mean(sub) if sub else np.nan,
                  "fidelity": np.mean(sub) if sub else np.nan})
    extra.append({**{k: r[k] for k in ("dataset", "muestra", "defecto", "familia_defecto")},
                  "validador": "metis_media_plana", "composite": np.mean(fid) if fid else np.nan,
                  "fidelity": np.mean(fid) if fid else np.nan})
rows += extra

# SynthEval por grupos de métricas (n_val, 1 = mejor) para E4.
SE_GROUPS = {
    "marginal": ["m:ks_tvd_stat", "m:frac_ks_sigs", "m:avg_h_dist"],
    "conditional": ["m:corr_mat_diff", "m:mutual_inf_diff", "m:pca_eigval_diff", "m:pca_eigvec_ang"],
    "global": ["m:avg_pMSE", "m:avg_qMSE", "m:avg_dwm_diff", "m:avg_cio"],
    "utility": ["m:avg_F1_diff", "m:avg_F1_diff_hout", "m:auroc"],
    "privacy": ["m:median_DCR", "m:avg_nndr", "m:priv_loss_nndr", "m:nnaa", "m:priv_loss_nnaa", "m:hit_rate",
                "m:eps_identif_risk", "m:priv_loss_eps", "m:mia_recall", "m:mia_precision"],
}
for r in rows:
    if r["validador"] == "syntheval":
        for g, ms in SE_GROUPS.items():
            v = [r[m] for m in ms if r.get(m) is not None and np.isfinite(r[m])]
            r[f"g:{g}"] = np.mean(v) if v else np.nan
    if r["validador"] in ("metis", "metis_plano"):
        for node, v in (r.get("jerarquia") or {}).items():
            r[f"h:{node}"] = v

df = pd.DataFrame(rows)
df = df.drop(columns=[c for c in ("metricas", "jerarquia") if c in df.columns])
VALS = ["metis", "metis_plano", "metis_sdm_metricas", "metis_media_plana", "sdmetrics", "syntheval"]
MAIN = ["metis", "sdmetrics", "syntheval"]
GENERAL = {"metis": "composite", "metis_plano": "composite", "metis_sdm_metricas": "composite",
           "metis_media_plana": "composite", "sdmetrics": "quality", "syntheval": "all"}
FAMILY = {
    "metis": {"fidelity": "fidelity", "utility": "utility", "privacy": "privacy", "validez": "validez"},
    "sdmetrics": {"fidelity": "quality", "privacy": "dcr_overfitting", "validez": "data_validity"},
    "syntheval": {"fidelity": "u", "utility": "u", "privacy": "p"},
}
FAMILY["metis_plano"] = FAMILY["metis"]
FAMILY["metis_sdm_metricas"] = FAMILY["metis_media_plana"] = {"fidelity": "fidelity"}
ALL_SCORES = {"metis": ["composite", "fidelity", "utility", "privacy", "validez"],
              "metis_plano": ["composite", "fidelity", "utility", "privacy", "validez"],
              "metis_sdm_metricas": ["composite"], "metis_media_plana": ["composite"],
              "sdmetrics": ["quality", "column_shapes", "column_pair_trends", "dcr_overfitting", "diagnostic",
                            "data_validity"],
              "syntheval": ["u", "p", "all"]}


def zscores(val, ds, col):
    """{defecto: (z mediana, caída mediana)} para una nota de un validador en un dataset."""
    d = df[(df.validador == val) & (df.dataset == ds)]
    if col not in d.columns:
        return {}
    good = d[d.defecto == "bueno"][col].astype(float).dropna()
    if len(good) < 3:
        return {}
    mu, sd = good.mean(), max(good.std(ddof=1), SD_FLOOR)
    out = {}
    for name, g in d[d.defecto != "bueno"].groupby("defecto"):
        x = g[col].astype(float).dropna()
        if len(x):
            out[name] = (float(np.median((mu - x) / sd)), float(np.median(mu - x)))
    return out


DATASETS = sorted(df.dataset.unique())
FAM_OF = df.drop_duplicates("defecto").set_index("defecto")["familia_defecto"].to_dict()
Z = {}  # (val, ds, modo) -> {defecto: (z, caída)}
for v in VALS:
    for ds in DATASETS:
        Z[(v, ds, "general")] = zscores(v, ds, GENERAL[v])
        fam = {}
        for col in set(FAMILY.get(v, {}).values()):
            zc = zscores(v, ds, col)
            for name, zz in zc.items():
                if FAMILY[v].get(FAM_OF.get(name)) == col:
                    fam[name] = zz
        Z[(v, ds, "familia")] = fam
        anyz = {}
        for col in ALL_SCORES[v]:
            for name, zz in zscores(v, ds, col).items():
                if name not in anyz or zz[0] > anyz[name][0]:
                    anyz[name] = zz
        Z[(v, ds, "alguna")] = anyz

L = []
w = L.append


def fmt(x, nd=1):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def det_table(defects, vals, modes=("general", "familia", "alguna")):
    w("| dataset | defecto | " + " | ".join(f"{v} {m}" for v in vals for m in modes) + " |")
    w("|---|---|" + "---|" * (len(vals) * len(modes)))
    for ds in DATASETS:
        for name in defects:
            cells = []
            seen = False
            for v in vals:
                for m in modes:
                    zz = Z[(v, ds, m)].get(name)
                    if zz:
                        seen = True
                    cells.append("—" if not zz else (f"**{zz[0]:.1f}**" if zz[0] >= Z_DET else f"{zz[0]:.1f}"))
            if seen:
                w(f"| {ds} | {name} | " + " | ".join(cells) + " |")


def det_count(vals, defects=None, modes=("general", "familia", "alguna")):
    w("| validador | " + " | ".join(modes) + " |")
    w("|---|" + "---|" * len(modes))
    for v in vals:
        cells = []
        for m in modes:
            n = k = 0
            for ds in DATASETS:
                for name, (z, _) in Z[(v, ds, m)].items():
                    if defects is None or name in defects:
                        n += 1
                        k += z >= Z_DET
            cells.append(f"{k}/{n}" if n else "—")
        w(f"| {v} | " + " | ".join(cells) + " |")


ALL_DEFECTS = sorted(d for d in df.defecto.unique() if d != "bueno")
w("# Banco de defectos: comparación de validadores\n")
w(f"z = (media de las 5 buenas − nota) / sd de las buenas (suelo {SD_FLOOR}); mediana de 5 muestras; "
  f"**negrita** = detectado (z ≥ {Z_DET:g}). «alguna» = el mayor z entre todas las notas agregadas que publica el validador.\n")

w("## 0. Ruido de las tablas buenas (media ± sd de la nota general)\n")
w("| validador | " + " | ".join(DATASETS) + " |")
w("|---|" + "---|" * len(DATASETS))
for v in VALS:
    cells = []
    for ds in DATASETS:
        g = df[(df.validador == v) & (df.dataset == ds) & (df.defecto == "bueno")][GENERAL[v]].astype(float)
        cells.append(f"{g.mean():.3f} ± {g.std(ddof=1):.3f}" if len(g) else "—")
    w(f"| {v} | " + " | ".join(cells) + " |")

w("\n## 1. Recuento de detecciones (casos dataset × defecto detectados / evaluados)\n")
w("### Todos los defectos\n")
det_count(VALS)
for fam in ("fidelity", "utility", "privacy", "validez"):
    names = {d for d in ALL_DEFECTS if FAM_OF.get(d) == fam}
    w(f"\n### Familia {fam}\n")
    det_count(VALS, names)

w("\n## 2. Detalle por defecto (validadores principales)\n")
det_table(ALL_DEFECTS, MAIN)

# E1 ------------------------------------------------------------------------------------------
w("\n## 3. E1 Heterogeneidad\n")
w("### 3a. Distancia ordinal: misma masa movida (20 %), nivel vecino frente a lejano\n")
w("Caída mediana de la nota (general y de fidelidad) y cociente lejos/cerca. Sin orden, el cociente debería rondar 1.\n")
w("| dataset | validador | nota | caída cerca | caída lejos | lejos/cerca | z cerca | z lejos |")
w("|---|---|---|---|---|---|---|---|")
for ds in DATASETS:
    for v in VALS:
        for m in ("general", "familia"):
            c, f = Z[(v, ds, m)].get("ordinal_cerca_20"), Z[(v, ds, m)].get("ordinal_lejos_20")
            if c and f:
                ratio = f[1] / c[1] if abs(c[1]) > 1e-9 else np.nan
                w(f"| {ds} | {v} | {m} | {c[1]:.4f} | {f[1]:.4f} | {fmt(ratio, 2)} | {c[0]:.1f} | {f[0]:.1f} |")
w("\n### 3b. Relaciones rotas por tipo y ordinales\n")
HET = ["num_cat_rota", "num_num_rota", "cat_cat_rota", "relaciones_rotas", "ordinal_orden_roto", "ordinal_desplazada",
       "ordinal_cerca_20", "ordinal_lejos_20"]
det_table(HET, MAIN)
w("\n### 3c. Validez (METIS: reglas del esquema declarado; SDMetrics: Data Validity inferida de la tabla real)\n")
VALID = ["discreta_fuera_reticula", "nivel_inventado", "fuera_de_rango"]
det_table(VALID, ["metis", "sdmetrics"], ("familia",))
w("\nValor medio de la nota de validez (1 = sin violaciones):\n")
w("| dataset | defecto | METIS validez | SDMetrics data_validity |")
w("|---|---|---|---|")
for ds in DATASETS:
    for name in ["bueno"] + VALID:
        a = df[(df.validador == "metis") & (df.dataset == ds) & (df.defecto == name)]
        b = df[(df.validador == "sdmetrics") & (df.dataset == ds) & (df.defecto == name)]
        if len(a) or len(b):
            w(f"| {ds} | {name} | {fmt(a['validez'].astype(float).mean(), 3) if len(a) else '—'} | "
              f"{fmt(b['data_validity'].astype(float).mean(), 3) if len(b) else '—'} |")

w("\n### 3d. E1b Ablación sin tipos (METIS completo frente a METIS con ordinal→nominal y discreta→continua)\n")
det_table(HET + VALID, ["metis", "metis_plano"], ("general", "familia"))
w("\n")
det_count(["metis", "metis_plano"], set(HET + VALID), ("general", "familia"))

# E2 ------------------------------------------------------------------------------------------
w("\n## 4. E2 Dosis\n")
LADDERS = {
    "desplazamiento (sd)": [("desplazar_005", 0.05), ("desplazar_010", 0.10), ("desplazar_025", 0.25)],
    "copias exactas": [("copias_1", 0.01), ("copias_2", 0.02), ("copias_5", 0.05), ("copias_10", 0.10),
                       ("copias_25", 0.25), ("copias_100", 1.0)],
    "etiquetas ruidosas": [("etiquetas_ruidosas_10", 0.10), ("etiquetas_ruidosas_50", 0.50)],
}
for title, ladder in LADDERS.items():
    w(f"\n### {title}: z por dosis (modo «alguna»); dosis mínima detectada y monotonía de la caída\n")
    w("| dataset | validador | " + " | ".join(n for n, _ in ladder) + " | mínima | monótona |")
    w("|---|---|" + "---|" * (len(ladder) + 2))
    for ds in DATASETS:
        for v in MAIN:
            zz = [Z[(v, ds, "alguna")].get(n) for n, _ in ladder]
            if not any(zz):
                continue
            mins = [d for (n, d), z in zip(ladder, zz) if z and z[0] >= Z_DET]
            drops = [z[1] for z in zz if z]
            mono = all(b >= a - 1e-9 for a, b in zip(drops, drops[1:]))
            w(f"| {ds} | {v} | " + " | ".join("—" if not z else (f"**{z[0]:.1f}**" if z[0] >= Z_DET else f"{z[0]:.1f}") for z in zz)
              + f" | {mins[0] if mins else 'ninguna'} | {'sí' if mono else 'no'} |")
w("\n### Resto de E2 (dosis única)\n")
det_table(["colas_extremas", "categorias_raras_perdidas", "ausentes_mcar"], MAIN)

w("\n### 4b. E2b Ablación de métricas: METIS con solo KS, TVD, Pearson y Cramér (las de SDMetrics) y METIS con la media plana de sus métricas de fidelidad\n")
fid = {d for d in ALL_DEFECTS if FAM_OF.get(d) == "fidelity"}
det_count(["metis", "metis_sdm_metricas", "metis_media_plana", "sdmetrics"], fid, ("general",))

# E3 ------------------------------------------------------------------------------------------
w("\n## 5. E3 Calibración: ¿un umbral fijado en un dataset sirve en los demás?\n")
w("Umbral = media − 3·sd de las buenas del dataset de origen; se aplica a las notas de cada otro dataset. "
  "FP = fracción de buenas marcadas; TP = fracción de casos de fidelidad marcados (por caso, no por mediana). "
  "Media sobre los 12 pares origen→destino.\n")
w("| validador | nota | FP | TP | dispersión entre datasets de la media de las buenas |")
w("|---|---|---|---|---|")
E3 = [("metis", "fidelity"), ("metis", "fidelity_raw"), ("metis", "composite"), ("metis_plano", "fidelity"),
      ("metis_media_plana", "fidelity"), ("sdmetrics", "quality"), ("syntheval", "u"), ("syntheval", "all")]
for v, col in E3:
    d = df[df.validador == v]
    if col not in d.columns:
        continue
    fps, tps, means = [], [], []
    for s in DATASETS:
        gs = d[(d.dataset == s) & (d.defecto == "bueno")][col].astype(float).dropna()
        if len(gs) < 3:
            continue
        means.append(gs.mean())
        t = gs.mean() - Z_DET * max(gs.std(ddof=1), SD_FLOOR)
        for dd in DATASETS:
            if dd == s:
                continue
            x = d[d.dataset == dd]
            g = x[x.defecto == "bueno"][col].astype(float).dropna()
            b = x[x.familia_defecto == "fidelity"][col].astype(float).dropna()
            if len(g) and len(b):
                fps.append((g < t).mean())
                tps.append((b < t).mean())
    w(f"| {v} | {col} | {fmt(np.mean(fps), 2)} | {fmt(np.mean(tps), 2)} | "
      f"{fmt(np.std(means), 3)} (de {fmt(min(means), 3)} a {fmt(max(means), 3)}) |")

# E4 ------------------------------------------------------------------------------------------
w("\n## 6. E4 Localización: ¿el nodo que más cae (en z) es el esperado?\n")
EXPECT = {}
for n in ["desplazar_numericas", "escala_numericas", "colas_cortadas", "desplazar_005", "desplazar_010", "desplazar_025",
          "colas_extremas", "categorias_aplanadas", "categorias_raras_perdidas", "ruido_uniforme", "ordinal_desplazada",
          "ordinal_cerca_20", "ordinal_lejos_20"]:
    EXPECT[n] = "marginal"
for n in ["relaciones_rotas", "num_cat_rota", "num_num_rota", "cat_cat_rota"]:
    EXPECT[n] = "conditional"
for n in ["copias_1", "copias_2", "copias_5", "copias_10", "copias_25", "copias_100", "casi_copias_25"]:
    EXPECT[n] = "privacy"
for n in ["objetivo_barajado", "etiquetas_ruidosas", "etiquetas_ruidosas_10", "etiquetas_ruidosas_50"]:
    EXPECT[n] = "utility"
SUBEXPECT = {"num_cat_rota": "num_cat", "num_num_rota": "num_num", "cat_cat_rota": "cat_cat"}
NODES = {
    "metis": {"marginal": "h:fidelity.marginal", "conditional": "h:fidelity.conditional", "global": "h:fidelity.global",
              "privacy": "privacy", "utility": "utility"},
    "sdmetrics": {"marginal": "column_shapes", "conditional": "column_pair_trends", "privacy": "dcr_overfitting"},
    "syntheval": {g: f"g:{g}" for g in SE_GROUPS},
}
NODES["metis_plano"] = NODES["metis"]
w("Niveles comparables: METIS (marginal, condicional, global, privacidad, utilidad); SDMetrics (Column Shapes, "
  "Column Pair Trends, DCR); SynthEval (sus métricas agrupadas en los mismos cinco). Se cuentan solo los casos con "
  "algún nodo en z ≥ 3; «posibles» = el nodo esperado existe en esa herramienta.\n")
w("| validador | aciertos top-1 | casos con detección | casos posibles (todos) |")
w("|---|---|---|---|")
loc_detail = []
for v, nodes in NODES.items():
    hit = det = poss = 0
    for ds in DATASETS:
        zn = {k: zscores(v, ds, col) for k, col in nodes.items()}
        for name, exp in EXPECT.items():
            if exp not in nodes:
                continue
            zs = {k: zn[k][name][0] for k in nodes if name in zn[k]}
            if not zs:
                continue
            poss += 1
            if max(zs.values()) < Z_DET:
                continue
            det += 1
            top = max(zs, key=zs.get)
            hit += top == exp
            loc_detail.append((v, ds, name, exp, top, zs[top]))
    w(f"| {v} | {hit}/{det} | {det} | {poss} |")
w("\n### Fallos de localización\n")
w("| validador | dataset | defecto | esperado | top-1 | z top-1 |")
w("|---|---|---|---|---|---|")
for v, ds, name, exp, top, z in loc_detail:
    if top != exp:
        w(f"| {v} | {ds} | {name} | {exp} | {top} | {z:.1f} |")
w("\n### Subcategoría de dependencia en METIS (num_num / num_cat / cat_cat)\n")
w("| validador | dataset | defecto | esperado | top-1 | z por subnodo |")
w("|---|---|---|---|---|---|")
for v in ("metis", "metis_plano"):
    for ds in DATASETS:
        zn = {s: zscores(v, ds, f"h:fidelity.conditional.{s}") for s in ("num_num", "num_cat", "cat_cat")}
        for name, exp in SUBEXPECT.items():
            zs = {s: zn[s][name][0] for s in zn if name in zn[s]}
            if zs:
                top = max(zs, key=zs.get)
                w(f"| {v} | {ds} | {name} | {exp} | {'**' + top + '**' if top == exp else top} | "
                  + ", ".join(f"{s} {z:.1f}" for s, z in zs.items()) + " |")

w("\n## 7. ¿Se premia copiar? (copias_100: z negativo = la nota sube)\n")
w("| dataset | " + " | ".join(f"{v} general" for v in MAIN) + " |")
w("|---|" + "---|" * len(MAIN))
for ds in DATASETS:
    w(f"| {ds} | " + " | ".join(fmt((Z[(v, ds, 'general')].get('copias_100') or (np.nan,))[0]) for v in MAIN) + " |")

OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
print("escrito", OUT, len(L), "líneas")
