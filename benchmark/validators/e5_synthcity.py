"""Añade synthcity a evalua_bench.py y a analiza_e5.py (en la AWS). Uso: e5_synthcity.py <evalua_bench.py> <analiza_e5.py>"""
import pathlib
import sys

p = pathlib.Path(sys.argv[1])
t = p.read_text(encoding="utf-8")
ancla = "for gen, seed, path in runs:"
rama = '''if VAL == "synthcity":
    # Mismas métricas y misma agregación que en el banco (evalua.py): media por categoría y en conjunto
    # de las métricas en [0, 1], orientadas para que más sea mejor.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from synthcity.metrics import Metrics
    from synthcity.plugins.core.dataloader import GenericDataLoader

    SC_METRICS = {
        "sanity": ["data_mismatch", "common_rows_proportion", "nearest_syn_neighbor_distance",
                   "close_values_probability", "distant_values_probability"],
        "stats": ["jensenshannon_dist", "chi_squared_test", "feature_corr", "inv_kl_divergence", "ks_test",
                  "max_mean_discrepancy", "wasserstein_dist", "prdc", "alpha_precision"],
        "performance": ["xgb"],
        "detection": ["detection_xgb"],
        "privacy": ["delta-presence", "k-anonymization", "k-map", "distinct l-diversity", "identifiability_score"],
    }

    def score(train, hold, syn, seed):
        fill = lambda d: d.fillna(d.median(numeric_only=True))
        res = Metrics.evaluate(GenericDataLoader(fill(train), target_column=target),
                               GenericDataLoader(fill(syn), target_column=target), metrics=SC_METRICS,
                               task_type=task, random_state=seed, use_cache=False)
        out, by_cat = {}, {}
        for name, row in res.iterrows():
            v = float(row["mean"])
            if not np.isfinite(v):
                continue
            out[f"m:{name}"] = v
            if 0.0 <= v <= 1.0:
                by_cat.setdefault(name.split(".")[0], []).append(v if row["direction"] == "maximize" else 1.0 - v)
        for cat, vals in by_cat.items():
            out[cat] = float(np.mean(vals))
        allv = [x for vals in by_cat.values() for x in vals]
        out["all"] = float(np.mean(allv)) if allv else float("nan")
        return out

'''
assert t.count(ancla) == 1 and "synthcity" not in t
p.write_text(t.replace(ancla, rama + ancla), encoding="utf-8")

q = pathlib.Path(sys.argv[2])
u = q.read_text(encoding="utf-8")
for a, b in (('"syntheval": ["all", "u", "p"]}', '"syntheval": ["all", "u", "p"],\n          "synthcity": ["all", "stats", "privacy"]}'),
             ('"syntheval": "all"}', '"syntheval": "all", "synthcity": "all"}'),
             ('for v in ("sdmetrics", "syntheval"):', 'for v in ("sdmetrics", "syntheval", "synthcity"):')):
    assert u.count(a) == 1, a
    u = u.replace(a, b)
q.write_text(u, encoding="utf-8")
print("ok")
