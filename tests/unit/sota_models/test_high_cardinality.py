"""CTGAN/TVAE refilled high-cardinality columns with resampled *real* values
(Airbnb names, host names and licences reached the synthetic table verbatim)."""

import pandas as pd

from metis.sota_models.generators.gan_utils import OTHER_CATEGORY, collapse_high_cardinality


def test_high_cardinality_columns_are_collapsed_not_dropped():
    data = pd.DataFrame({"name": [f"n{i}" for i in range(300)] + ["top"] * 50, "c": ["a"] * 350})
    out, collapsed = collapse_high_cardinality(data, ["name", "c"], threshold=10)
    assert collapsed == ["name"]
    assert out["name"].nunique() <= 11
    assert "top" in set(out["name"])
    assert (out["name"] == OTHER_CATEGORY).sum() >= 300 - 9
    assert out["c"].equals(data["c"])
