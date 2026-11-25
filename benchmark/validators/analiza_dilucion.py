"""¿Dónde se pierde la señal al agregar?  METIS v3 sobre el banco 2 y los defectos del estudio.

Uso: analiza_dilucion.py <dir_res> <dir_res_estudio> <salida.md>

Para cada dataset y defecto, el z de cada nivel de la jerarquía (mismo criterio que el banco:
z = (media de las buenas − valor) / max(sd, 0,005), mediana de las 5 muestras):
  métrica  el mayor z entre las métricas sueltas de METIS
  sub / cat  el mayor z entre las subcategorías y categorías (``jerarquia`` del registro)
  familia  el mayor z entre fidelidad, utilidad y privacidad (más validez)
  compuesto  el z del compuesto
Un defecto «se diluye» si alguna métrica lo ve (z ≥ 3) y ninguna nota (compuesto, familias, validez)
lo ve.  Se dice en qué nivel cae por debajo de 3 y qué métrica lo veía.

También: detección y falsos positivos de las dos métricas nuevas (fidelity.discriminator,
utility.log_loss) y de sus notas (fidelity.detection, utility.probabilistic).
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RES, REST, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
SD_FLOOR, ZD = 0.005, 3.0
KEY = ["dataset", "muestra", "defecto"]


def read(pattern):
    rows = []
    for f in sorted(glob.glob(str(pattern))):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if not r.get("error") and r.get("validador", "metis") == "metis":
                rows.append(r)
    return rows


recs = []
for r in read(RES / "metis.jsonl") + read(REST / "metis_est_*.jsonl"):
    d = {k: r[k] for k in KEY}
    d.update({s: r.get(s) for s in ("composite", "fidelity", "utility", "privacy", "validez")})
    d.update({f"m:{k}": v for k, v in (r.get("metricas") or {}).items()})
    d.update({f"h:{k}": v for k, v in (r.get("jerarquia") or {}).items()})
    recs.append(d)
DF = pd.DataFrame(recs).drop_duplicates(KEY)
MCOLS = [c for c in DF.columns if c.startswith("m:")]
SUB = [c for c in DF.columns if c.startswith("h:") and c.count(".") == 2]
CAT = [c for c in DF.columns if c.startswith("h:") and c.count(".") == 1]
FAM = ["fidelity", "utility", "privacy", "validez"]
NOTES = ["composite"] + FAM
DATASETS = sorted(DF.dataset.unique())


def z_of(ds, col):
    d = DF[DF.dataset == ds]
    if col not in d or d[col].isna().all():
        return {}
    g = d[d.defecto == "bueno"][col].astype(float).dropna()
    if len(g) < 3:
        return {}
    mu, sd = g.mean(), max(g.std(ddof=1), SD_FLOOR)
    return {n: float(np.median((mu - x[col].astype(float).dropna()) / sd))
            for n, x in d[d.defecto != "bueno"].groupby("defecto") if x[col].notna().any()}


def best(ds, cols):
    out = {}
    for c in cols:
        for n, z in z_of(ds, c).items():
            if z > out.get(n, (-np.inf, None))[0]:
                out[n] = (z, c)
    return out


def goods_fp(ds, col):
    g = DF[(DF.dataset == ds) & (DF.defecto == "bueno")][col].astype(float).dropna().to_numpy()
    return sum((np.delete(g, i).mean() - g[i]) / max(np.delete(g, i).std(ddof=1), SD_FLOOR) >= ZD
               for i in range(len(g))), len(g)


L = [__doc__.split("Uso:")[0].strip(), ""]
w = L.append
n_det = n_dil = 0
w("## Defectos que alguna métrica ve y ninguna nota ve\n")
w("| dataset | defecto | métrica (z) | sub | cat | familia | compuesto |")
w("|---|---|---|---|---|---|---|")
for ds in DATASETS:
    zm, zs, zc, zf = best(ds, MCOLS), best(ds, SUB), best(ds, CAT), best(ds, FAM)
    zcomp = z_of(ds, "composite")
    for n in sorted(zm):
        notes = max([zf.get(n, (-np.inf,))[0], zcomp.get(n, -np.inf)])
        if notes >= ZD:
            n_det += 1
        if zm[n][0] >= ZD and notes < ZD:
            n_dil += 1
            f = lambda t: "—" if t is None else f"{t[0]:.1f} ({t[1][2:] if t[1][:2] in ('m:', 'h:') else t[1]})"
            w(f"| {ds} | {n} | {f(zm[n])} | {f(zs.get(n))} | {f(zc.get(n))} | {f(zf.get(n))} | "
              f"{zcomp.get(n, float('nan')):.1f} |")
w(f"\nDefectos vistos por alguna nota: {n_det}.  Diluidos (los ve una métrica, no una nota): {n_dil}.\n")

w("## Las métricas nuevas y sus notas\n")
w("| columna | detecciones (z ≥ 3) | buenas marcadas |")
w("|---|---|---|")
for col in ("m:fidelity.discriminator", "h:fidelity.detection", "m:utility.log_loss", "h:utility.probabilistic",
            "m:utility.tstr", "fidelity", "utility", "composite"):
    if col not in DF:
        w(f"| {col} | (no está) | |")
        continue
    det = sum(z >= ZD for ds in DATASETS for z in z_of(ds, col).values())
    fp = [goods_fp(ds, col) for ds in DATASETS]
    w(f"| {col} | {det} | {sum(a for a, _ in fp)}/{sum(b for _, b in fp)} |")

w("\n## z por defecto diana\n")
DIANA = ["trios_rotos", "filas_imposibles", "colapso_modos", "categorias_raras_perdidas", "colas_extremas",
         "etiquetas_ruidosas", "etiquetas_ruidosas_10", "etiquetas_ruidosas_50", "objetivo_barajado"]
COLS = ["m:fidelity.discriminator", "m:utility.log_loss", "m:utility.tstr", "fidelity", "utility", "composite"]
w("| dataset | defecto | " + " | ".join(c[2:] if c[:2] == "m:" else c for c in COLS) + " |")
w("|---|---|" + "---|" * len(COLS))
for ds in DATASETS:
    zz = {c: z_of(ds, c) for c in COLS}
    for n in DIANA:
        if any(n in zz[c] for c in COLS):
            cells = [("—" if n not in zz[c] else (f"**{zz[c][n]:.1f}**" if zz[c][n] >= ZD else f"{zz[c][n]:.1f}"))
                     for c in COLS]
            w(f"| {ds} | {n} | " + " | ".join(cells) + " |")
OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
print("escrito", OUT)
