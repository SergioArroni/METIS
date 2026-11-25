"""Lectura de pisos_privacy.csv (de pisos_familia.py): fugas frente a control, por regla.

Uso: analiza_privacidad.py <dir>
"""

import sys
from pathlib import Path

import pandas as pd

d = Path(sys.argv[1])
r = pd.read_csv(d / "pisos_privacy.csv")
keys = [c for c in r.columns if c.count("/") == 2 and "|" not in c]
fugas = r[r.defecto != "control_sin_fuga"]
ctrl = r[r.defecto == "control_sin_fuga"]
pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 200)

print("=== fugas: detectadas (z ≥ 3), no detectadas (z < 2); control: falsas alarmas (z ≥ 3) ===")
print(pd.DataFrame({
    "fugas_detectadas": (fugas[keys] >= 3).mean().round(3),
    "fugas_no_detectadas": (fugas[keys] < 2).sum(),
    "falsas_alarmas_control": (ctrl[keys] >= 3).sum(),
    "z_mediano_fugas": fugas[keys].median().round(1),
}))

orden = ["copias_1", "copias_5", "copias_10", "copias_25", "copias_100", "casi_copias_10", "casi_copias_25",
         "control_sin_fuga"]
print("\n=== cuánto baja la nota de privacidad (Δ, media de las 4 tablas) por fuga y regla ===")
dl = r.groupby("defecto")[[k + "|delta" for k in keys]].mean().reindex(orden)
dl.columns = keys
print(dl.round(3))
print("\n=== z por fuga y regla (mediana de las 4 tablas) ===")
print(r.groupby("defecto")[keys].median().reindex(orden).round(1))
print("\n=== z por tabla, copias del 5 % y 10 % ===")
print(r[r.defecto.isin(["copias_5", "copias_10"])].set_index(["defecto", "dataset"])[keys].round(1))
