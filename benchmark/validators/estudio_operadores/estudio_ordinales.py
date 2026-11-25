"""Ordinales en METIS, segunda ronda: híbridos, espaciado aprendido y costes de la doble vista.

Uso: estudio_ordinales.py <dataset> <salida.jsonl> [n_buenos]

Mismo montaje que estudio_tipos.py (A referencia, H holdout, B reserva; las buenas son
remuestreos de B del tamaño de A; a cada buena se le aplica cada defecto).  Tratamientos de las
columnas ordinales:

  doble_vista  nivel (categórica) + rango i/(K-1) (numérica)            — lo actual
  nominal      solo el nivel
  rango        solo i/(K-1), como continua
  ridit        solo el ridit de A (Bross 1958), como continua
  warp         solo el espaciado aprendido (abajo), como continua
  doble_ridit  nivel + ridit en la vista numérica
  doble_warp   nivel + espaciado aprendido en la vista numérica
  doble_off    doble_vista, pero un nivel fuera de la lista queda «__off__» en la categórica
               en vez de hueco (la propuesta para los niveles inventados)
  doble_ridit01  nivel + ridit reescalado a [0, 1] (Podani 1999: misma forma, mismo peso que el rango)
  doble_probit   nivel + puntuaciones normales del umbral latente (Olsson 1979), en [0, 1]

Espaciado aprendido: el «ordinal warping» de Picheny, Vakili y Artemev (2019) coloca el nivel k en
s_k = δ_1 + … + δ_k con incrementos δ > 0 aprendidos.  Allí se aprenden con un GP sobre un
objetivo; aquí no hay objetivo, así que se aprenden como en el escalado óptimo monótono
(Young, de Leeuw y Takane 1976; CATPCA): mínimos cuadrados alternos que buscan las puntuaciones
monótonas de la ordinal más explicables linealmente por el resto de la tabla A.  δ_k ≥ 2 % del
recorrido (estrictamente positivos, como en el paper) y escala final [0, 1].

Defectos nuevos: correlacion_inventada (la marginal de la ordinal intacta, pero ordenada según la
numérica que menos relación tiene con ella en A: una asociación que la real no tiene) y
rotura_en_otras (las ordinales intactas y el resto de columnas barajadas: ¿diluye la doble vista,
que pesa más las ordinales en los pares, los defectos de las demás variables?).

Además de las notas por subcategoría guarda el valor de cada métrica (para reagregar).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.isotonic import IsotonicRegression

from metis.application.pipeline import preprocessor as _pp
from metis.infrastructure.preprocess import caster as _caster
from metis.interface.sdk import Evaluator

DS, OUT = sys.argv[1], Path(sys.argv[2])
N_GOOD = int(sys.argv[3]) if len(sys.argv) > 3 else 8
K_FROM = int(sys.argv[4]) if len(sys.argv) > 4 else 0  # repartir muestras entre procesos
MAX_ROWS = 6000
METRICS = ["fidelity", "privacy.dcr", "privacy.mia", "privacy.nnaa", "privacy.record_linkage"]
ADULT_EDU = ["Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", "11th", "12th", "HS-grad",
             "Some-college", "Assoc-voc", "Assoc-acdm", "Bachelors", "Masters", "Prof-school", "Doctorate"]

if DS == "adult":
    from metis.infrastructure.typing import infer_schema

    real = pd.read_csv("data/real/adult_test.csv", skiprows=[1], skipinitialspace=True)
    real = real.drop(columns=["Education_Num"])
    for c in real.columns:
        if real[c].dtype == object:
            real[c] = real[c].str.strip().str.rstrip(".")
    schema = dict(infer_schema(real).schema)
    schema["Education"] = {"type": "ordinal", "levels": ADULT_EDU}
else:
    CSV = {"rrhh": "rrhh.csv", "cardio": "cardio_train.csv"}
    schema = yaml.safe_load(open(f"metis/configs/config_{DS}.yaml", encoding="utf-8"))["data"]["schema"]
    real = pd.read_csv(f"data/real/{CSV[DS]}", sep=None, engine="python", encoding="utf-8-sig")
real = real[[c for c in schema if c in real.columns]]
if len(real) > MAX_ROWS:
    real = real.sample(MAX_ROWS, random_state=0).reset_index(drop=True)
idx = np.random.default_rng(0).permutation(len(real))
third = len(real) // 3
A, H, B = (real.iloc[idx[i * third : (i + 1) * third]].reset_index(drop=True) for i in range(3))


def typ(spec):
    return spec if isinstance(spec, str) else spec.get("type")


cols = [c for c, sp in schema.items() if typ(sp) == "ordinal" and c in real.columns]
assert cols, f"{DS} no tiene ordinales"
norm = _caster.SimpleCaster._normalize_str


def levels(c):
    return [str(v) for v in schema[c]["levels"]]


def positions(df, c):
    pos = {v: i for i, v in enumerate(levels(c))}
    return df[c].astype(str).map(pos)


# ── vistas numéricas alternativas, aprendidas una vez sobre A ────────────


def ridit_scores(c):
    p = positions(A, c).value_counts(normalize=True).reindex(range(len(levels(c))), fill_value=0.0)
    return (np.cumsum(p.to_numpy()) - p.to_numpy() / 2).tolist()


def _design(c):
    """El resto de A como matriz numérica: numéricas estandarizadas, rangos de las otras
    ordinales y one-hot (hasta 20 categorías) de las nominales."""
    parts = []
    for o, sp in schema.items():
        if o == c or o not in A.columns:
            continue
        t = typ(sp)
        if t in ("continuous", "discrete"):
            v = pd.to_numeric(A[o], errors="coerce")
        elif t == "ordinal":
            v = positions(A, o).astype(float)
        elif t == "nominal":
            top = A[o].astype(str).value_counts().index[:20]
            for cat in top:
                d = (A[o].astype(str) == cat).astype(float)
                if d.std() > 0:
                    parts.append(((d - d.mean()) / d.std()).to_numpy())
            continue
        else:
            continue
        if v.notna().sum() > 10 and v.std() > 0:
            parts.append(((v - v.mean()) / v.std()).fillna(0.0).to_numpy())
    return np.column_stack(parts)


def warp_scores(c, iters=20):
    """Escalado óptimo monótono de la ordinal *c* (ALS con restricción de orden)."""
    k = len(levels(c))
    pos = positions(A, c)
    ok = pos.notna().to_numpy()
    X = _design(c)[ok]
    p = pos[ok].astype(int).to_numpy()
    n_k = np.bincount(p, minlength=k).astype(float)
    s = np.arange(k, dtype=float)
    for _ in range(iters):
        z = s[p]
        z = (z - z.mean()) / (z.std() or 1.0)
        beta, *_ = np.linalg.lstsq(X, z, rcond=None)
        fit = X @ beta
        m_k = np.array([fit[p == j].mean() if n_k[j] else np.nan for j in range(k)])
        seen = ~np.isnan(m_k)
        incr = np.corrcoef(np.arange(k)[seen], m_k[seen])[0, 1] >= 0 if seen.sum() > 1 else True
        iso = IsotonicRegression(increasing=bool(incr)).fit(np.arange(k)[seen], m_k[seen], sample_weight=n_k[seen])
        y = iso.predict(np.arange(k))
        y = y if incr else -y
        span = y[-1] - y[0]
        if not np.isfinite(span) or span <= 1e-12:
            return list(np.arange(k) / (k - 1))  # sin relación aprendible: el rango
        d = np.maximum(np.diff(y), 0.02 * span)
        s = np.concatenate([[0.0], np.cumsum(d)])
    return list(s / s[-1])


def _to01(sc):
    sc = np.asarray(sc, dtype=float)
    span = sc[-1] - sc[0]
    return list((sc - sc[0]) / span) if span > 0 else list(np.arange(len(sc)) / (len(sc) - 1))


def probit_scores(c):
    """Puntuaciones normales del umbral latente: τ_k = Φ⁻¹(F(k)) y s_k = E[Z | τ_{k−1} < Z ≤ τ_k]
    = (φ(τ_{k−1}) − φ(τ_k)) / p_k (Olsson 1979, primera etapa); un nivel vacío toma τ_k."""
    from scipy.stats import norm as N

    k = len(levels(c))
    p = positions(A, c).value_counts(normalize=True).reindex(range(k), fill_value=0.0).to_numpy()
    tau = N.ppf(np.clip(np.cumsum(p), 1e-12, 1 - 1e-12))
    lo = np.concatenate([[-np.inf], tau[:-1]])
    s = np.where(p > 0, (N.pdf(lo) - N.pdf(tau)) / np.where(p > 0, p, 1), tau)
    return _to01(np.maximum.accumulate(s))


RIDIT = {c: ridit_scores(c) for c in cols}
RIDIT01 = {c: _to01(RIDIT[c]) for c in cols}  # Podani: sin cambiar el peso de la ordinal
PROBIT = {c: probit_scores(c) for c in cols}
WARP = {c: warp_scores(c) for c in cols}

# ── parches de la vista numérica y del nivel fuera de lista ──────────────

NUMVIEW: dict[str, dict[str, float]] = {}
OFF = {"on": False}
_orig_cast = _caster.SimpleCaster._cast_ordinal
_orig_mark = _pp.DataPreprocessor._mark_invalid


def _cast_ordinal(self, col, s, col_spec, df_cat, df_num, ctx):
    _orig_cast(self, col, s, col_spec, df_cat, df_num, ctx)
    m = NUMVIEW.get(col)
    if m is not None:
        df_num[col + "__ord"] = s.astype("object").map(norm).map(m).astype(float)


def _mark_invalid(self, td, raw, type_schema, label):
    td2, inv = _orig_mark(self, td, raw, type_schema, label)
    if not OFF["on"] or inv.empty:
        return td2, inv
    cat = td2.cat.copy()
    for col in cols:
        if col in cat.columns and col in inv.columns:
            cat.loc[inv[col].to_numpy(), col] = "__off__"
    return type(td2)(cat=cat, num=td2.num, full=pd.concat([cat, td2.num], axis=1), meta=td2.meta,
                     excluded_ids=td2.excluded_ids, schema=td2.schema), inv


_caster.SimpleCaster._cast_ordinal = _cast_ordinal
_pp.DataPreprocessor._mark_invalid = _mark_invalid


def as_numeric(df, scores):
    out = df.copy()
    for c in cols:
        out[c] = positions(df, c).map(lambda i, sc=scores[c]: sc[int(i)] if pd.notna(i) else np.nan)
    return out


def treat(df, name):
    """(tabla, esquema) con las ordinales bajo el tratamiento *name*; fija los parches."""
    sch = dict(schema)
    NUMVIEW.clear()
    OFF["on"] = name == "doble_off"
    if name in ("doble_vista", "doble_off"):
        return df.copy(), sch
    if name in ("doble_ridit", "doble_warp", "doble_ridit01", "doble_probit"):
        sc = {"doble_ridit": RIDIT, "doble_warp": WARP, "doble_ridit01": RIDIT01, "doble_probit": PROBIT}[name]
        for c in cols:
            NUMVIEW[c] = {norm(v): sc[c][i] for i, v in enumerate(levels(c))}
        return df.copy(), sch
    if name == "nominal":
        for c in cols:
            sch[c] = "nominal"
        return df.copy(), sch
    sc = {"rango": {c: list(np.arange(len(levels(c))) / (len(levels(c)) - 1)) for c in cols},
          "ridit": RIDIT, "warp": WARP}[name]
    for c in cols:
        sch[c] = "continuous"
    return as_numeric(df, sc), sch


# ── defectos ─────────────────────────────────────────────────────────────


def _ord_map(df, c, fn):
    lv = levels(c)
    pos = {v: i for i, v in enumerate(lv)}
    out = df.copy()
    out[c] = df[c].astype(str).map(lambda v: lv[fn(pos[v], len(lv))] if v in pos else v)
    return out


def _weakest_numeric(c):
    best, best_r = None, np.inf
    for o, sp in schema.items():
        if o == c or o not in A.columns or typ(sp) not in ("continuous", "discrete"):
            continue
        v = pd.to_numeric(A[o], errors="coerce")
        if v.nunique() < 10:
            continue
        r = abs(pd.Series(positions(A, c)).corr(v, method="spearman"))
        if np.isfinite(r) and r < best_r:
            best, best_r = o, r
    return best


WEAK = {c: _weakest_numeric(c) for c in cols}


def defects(good, rng):
    d = {}
    x = good.copy()
    for c in cols:
        x = _ord_map(x, c, lambda i, k: min(i + 1, k - 1))
    d["desplazar_un_nivel"] = x
    x = good.copy()
    for c in cols:
        lv = levels(c)
        perm = dict(zip(lv, rng.permutation(lv)))
        x[c] = x[c].astype(str).map(lambda v, p=perm: p.get(v, v))
    d["orden_roto"] = x
    x = good.copy()
    for c in cols:
        x[c] = rng.choice(levels(c), len(x))
    d["aplanar"] = x
    x = good.copy()
    for c in cols:
        x[c] = x[c].astype(str).mode().iloc[0]
    d["colapso_moda"] = x
    x = good.copy()
    for c in cols:
        x.loc[rng.random(len(x)) < 0.2, c] = "__nivel_inventado__"
    d["nivel_inventado"] = x
    x = good.copy()
    for c in cols:
        x[c] = rng.permutation(x[c].to_numpy())
    d["relacion_rota"] = x
    x = good.copy()
    for c in cols:  # la marginal intacta, ordenada según una numérica sin relación en A
        o = WEAK[c]
        if o is None:
            continue
        vals = x[c].astype(str).to_numpy()
        vals = np.array(sorted(vals, key=lambda v, lv=levels(c): lv.index(v) if v in lv else -1))
        order = pd.to_numeric(x[o], errors="coerce").rank(method="first").fillna(0).to_numpy().argsort()
        new = np.empty(len(x), dtype=object)
        new[order] = vals
        x[c] = new
    d["correlacion_inventada"] = x
    x = A.sample(len(good), replace=True, random_state=int(rng.integers(1e9))).reset_index(drop=True)
    for c in cols:
        x = _ord_map(x, c, lambda i, k: i + 1 if i + 1 < k else i - 1)
    d["casi_copia"] = x
    x = good.copy()  # al final: no cambia lo que el rng da a los defectos anteriores
    for o, sp in schema.items():  # las ordinales intactas; se rompen las relaciones del resto
        if o in x.columns and o not in cols and typ(sp) in ("continuous", "discrete", "nominal"):
            x[o] = rng.permutation(x[o].to_numpy())
    d["rotura_en_otras"] = x
    return d


TREATMENTS = ["doble_vista", "nominal", "rango", "ridit", "warp", "doble_ridit", "doble_warp", "doble_off",
              "doble_ridit01", "doble_probit"]

meta = OUT.with_suffix(".meta.json")
meta.write_text(json.dumps({"dataset": DS, "ordinales": cols, "ridit": RIDIT, "ridit01": RIDIT01,
                            "probit": PROBIT, "warp": WARP,
                            "numerica_debil": WEAK, "filas_por_tercio": third}, indent=1), encoding="utf-8")

done = set()
if OUT.exists():
    for line in OUT.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        done.add((r["tratamiento"], r["muestra"], r["defecto"]))

# Sin caché de la real: su clave es (tabla, config) y no ve los parches de la vista numérica, así
# que los tratamientos doble_* (misma tabla, mismo esquema) reutilizaban una real preparada con la
# vista de otro tratamiento.  La primera ejecución de este estudio cayó en eso.
ev = Evaluator(cache_reference=False)
for k in range(K_FROM, N_GOOD):
    rng = np.random.default_rng(100 + k)
    good = B.sample(len(A), replace=True, random_state=k).reset_index(drop=True)
    cases = {"bueno": good, **defects(good, rng)}
    for tr in TREATMENTS:
        for defect, synth in cases.items():
            if (tr, k, defect) in done:
                continue
            t0 = time.perf_counter()
            real_t, sch = treat(A, tr)
            synth_t, _ = treat(synth, tr)
            hold_t, _ = treat(H, tr)
            conf = {"data": {"schema": sch}, "metrics": METRICS, "evaluation": {"n_runs": 1}}
            s = ev.evaluate(real_t, synth_t, conf, seed=k, holdout=hold_t)
            h = s.aggregates.get("hierarchy", {})
            rec = {"dataset": DS, "tipo": "ordinal", "tratamiento": tr, "muestra": k, "defecto": defect,
                   "fidelity": s.aggregates.get("fidelity_score"), "privacy": s.aggregates.get("privacy_score"),
                   "segundos": round(time.perf_counter() - t0, 1)}
            for fam in ("fidelity", "privacy"):
                for cat, node in ((h.get(fam) or {}).get("categories") or {}).items():
                    rec[f"{fam[:3]}_{cat}"] = node.get("score")
                    for sub, sn in (node.get("subcategories") or {}).items():
                        rec[f"{fam[:3]}_{cat}.{sub}"] = sn.get("score")
            rec["metricas"] = {r.id: (r.value if r.value == r.value else None) for r in s.results}
            with OUT.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
    print(DS, "muestra", k, "hecha", flush=True)
