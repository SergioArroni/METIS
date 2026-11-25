"""¿Aportan las métricas candidatas?  Regla fijada antes de ver los resultados (29 oct):

Una candidata se propone para METIS solo si cumple las tres:
  1. aporta      detecta (z >= 3) algún defecto, en >= 2 datasets, que METIS no detecta con
                 ninguna de sus notas (compuesto, familias, validez).  Se separa si ninguna métrica
                 individual de METIS lo detecta tampoco (señal nueva) o si alguna sí (la señal ya
                 está y se pierde al agregar: eso pide revisar la agregación, no añadir métrica).
  2. no redundante  |rho| de Spearman < 0,9 con todas las métricas actuales de METIS (mediana
                 entre datasets, sobre todos los casos del banco 2).
  3. sin ruido   ninguna buena marcada como defecto al compararla con las otras cuatro (z >= 3).

Uso: analiza_estudio.py <dir_res2> <dir_res_estudio> <salida.md>
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

RES2, REST, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
SD_FLOOR, ZD = 0.005, 3.0
KEY = ["dataset", "muestra", "defecto"]


def read(pattern):
    rows = []
    for f in sorted(glob.glob(str(pattern))):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if not r.get("error"):
                rows.append(r)
    return rows


met = []
for r in read(RES2 / "metis.jsonl") + read(REST / "metis_est_*.jsonl"):
    d = {k: r[k] for k in KEY + ["familia_defecto"]}
    d.update({s: r.get(s) for s in ("composite", "fidelity", "utility", "privacy", "validez")})
    d.update({f"m:{k}": v for k, v in (r.get("metricas") or {}).items()})
    met.append(d)
M = pd.DataFrame(met).drop_duplicates(KEY)
N = pd.DataFrame(read(REST / "nuevas_*.jsonl")).drop_duplicates(KEY)
CAND = ["u_multi", "u_auc", "u_logloss", "u_brier", "u_accuracy", "u_f1", "discriminador", "alpha_precision",
        "beta_coverage", "authenticity", "rare_recall", "trios"]
SCORES = ["composite", "fidelity", "utility", "privacy", "validez"]
MCOLS = [c for c in M.columns if c.startswith("m:")]
DF = M.merge(N[KEY + CAND + ["total"]], on=KEY, how="outer")
DATASETS = sorted(DF.dataset.unique())


def z_by_defect(ds, col):
    d = DF[DF.dataset == ds]
    if col not in d or d[col].isna().all():
        return {}
    g = d[d.defecto == "bueno"][col].astype(float).dropna()
    if len(g) < 3:
        return {}
    mu, sd = g.mean(), max(g.std(ddof=1), SD_FLOOR)
    out = {}
    for name, x in d[d.defecto != "bueno"].groupby("defecto"):
        v = x[col].astype(float).dropna()
        if len(v):
            out[name] = float(np.median((mu - v) / sd))
    return out


def max_z(ds, cols):
    out = {}
    for c in cols:
        for name, z in z_by_defect(ds, c).items():
            out[name] = max(out.get(name, -np.inf), z)
    return out


Z_SCORES = {ds: max_z(ds, SCORES) for ds in DATASETS}
Z_METRICS = {ds: max_z(ds, MCOLS) for ds in DATASETS}


def goods_fp(ds, col):
    g = DF[(DF.dataset == ds) & (DF.defecto == "bueno")][col].astype(float).dropna().to_numpy()
    fp = 0
    for i in range(len(g)):
        rest = np.delete(g, i)
        fp += (rest.mean() - g[i]) / max(rest.std(ddof=1), SD_FLOOR) >= ZD
    return int(fp), len(g)


def redundancy(col):
    best = {}
    for ds in DATASETS:
        d = DF[(DF.dataset == ds)].dropna(subset=[col])
        for m in MCOLS:
            x = d[[col, m]].dropna()
            if len(x) > 20 and x[m].nunique() > 2 and x[col].nunique() > 2:
                best.setdefault(m, []).append(abs(spearmanr(x[col], x[m])[0]))
    med = {m: float(np.median(v)) for m, v in best.items() if len(v) >= 2}
    if not med:
        return np.nan, "—"
    m = max(med, key=med.get)
    return med[m], m[2:]


L = []
w = L.append
w("# Estudio: ¿aportan las métricas candidatas a METIS?\n")
for line in __doc__.split("Uso:")[0].strip().splitlines():
    w("> " + line)
w("")
w("| candidata | detecciones | nuevas frente a las notas de METIS | … y frente a toda métrica de METIS | "
  "max \\|ρ\\| (métrica) | FP buenas | s/caso | ¿cumple? |")
w("|---|---|---|---|---|---|---|---|")
detail = {}
for c in CAND:
    det = new_s = new_m = 0
    new_s_list, new_m_list = {}, {}
    fp = n_g = 0
    for ds in DATASETS:
        zc = z_by_defect(ds, c)
        for name, z in zc.items():
            if z >= ZD:
                det += 1
                if Z_SCORES[ds].get(name, -np.inf) < ZD:
                    new_s_list.setdefault(name, []).append(ds)
                    if Z_METRICS[ds].get(name, -np.inf) < ZD:
                        new_m_list.setdefault(name, []).append(ds)
        f, n = goods_fp(ds, c) if c in DF else (0, 0)
        fp, n_g = fp + f, n_g + n
    rho, which = redundancy(c)
    secs = np.nan
    aporta = any(len(v) >= 2 for v in new_s_list.values())
    nueva = any(len(v) >= 2 for v in new_m_list.values())
    ok = aporta and (rho < 0.9 if np.isfinite(rho) else True) and fp == 0
    verdict = ("sí (señal nueva)" if ok and nueva else "sí (la señal estaba, se perdía al agregar)" if ok
               else "no")
    fmt = lambda d: "; ".join(f"{k} ({', '.join(v)})" for k, v in sorted(d.items())) or "—"
    w(f"| {c} | {det} | {fmt(new_s_list)} | {fmt(new_m_list)} | {rho:.2f} ({which}) | {fp}/{n_g} | — | {verdict} |")

TARGETS = {
    "utilidad": (["u_multi", "u_auc", "u_logloss", "u_brier", "u_accuracy", "u_f1"],
                 ["etiquetas_ruidosas_10", "etiquetas_ruidosas", "etiquetas_ruidosas_50", "objetivo_barajado"]),
    "discriminador": (["discriminador"], ["desplazar_005", "colas_extremas", "trios_rotos", "colapso_modos",
                                          "filas_imposibles", "categorias_raras_perdidas", "ordinal_cerca_20"]),
    "α-precisión": (["alpha_precision"], ["filas_imposibles", "ruido_uniforme", "fuera_de_rango"]),
    "β-cobertura": (["beta_coverage"], ["colapso_modos", "colas_extremas", "colas_cortadas", "categorias_raras_perdidas"]),
    "autenticidad": (["authenticity"], ["copias_1", "copias_5", "copias_10", "casi_copias_25"]),
    "categorías raras": (["rare_recall"], ["categorias_raras_perdidas", "categorias_aplanadas"]),
    "tríos": (["trios"], ["trios_rotos", "relaciones_rotas", "mitad_rota"]),
}
w("\n## Defectos diana: z de la candidata frente a METIS\n")
w("«METIS notas» = mayor z entre compuesto, familias y validez; «METIS métricas» = mayor z entre todas sus métricas sueltas.\n")
for title, (cols, defects) in TARGETS.items():
    w(f"\n### {title}\n")
    w("| dataset | defecto | " + " | ".join(cols) + " | METIS notas | METIS métricas |")
    w("|---|---|" + "---|" * (len(cols) + 2))
    for ds in DATASETS:
        zc = {c: z_by_defect(ds, c) for c in cols}
        for name in defects:
            if not any(name in zc[c] for c in cols):
                continue
            cells = []
            for c in cols + ["__s", "__m"]:
                z = (Z_SCORES[ds] if c == "__s" else Z_METRICS[ds] if c == "__m" else zc[c]).get(name)
                cells.append("—" if z is None or not np.isfinite(z) else (f"**{z:.1f}**" if z >= ZD else f"{z:.1f}"))
            w(f"| {ds} | {name} | " + " | ".join(cells) + " |")

w("\n## Buenas: media ± sd de cada candidata\n")
w("| candidata | " + " | ".join(DATASETS) + " |")
w("|---|" + "---|" * len(DATASETS))
for c in CAND:
    cells = []
    for ds in DATASETS:
        g = DF[(DF.dataset == ds) & (DF.defecto == "bueno")][c].astype(float).dropna()
        cells.append(f"{g.mean():.3f} ± {g.std(ddof=1):.3f}" if len(g) > 1 else "—")
    w(f"| {c} | " + " | ".join(cells) + " |")
w(f"\nTiempo medio por caso (las cinco familias de candidatas juntas): {DF['total'].astype(float).mean():.1f} s\n")
OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
print("escrito", OUT)
