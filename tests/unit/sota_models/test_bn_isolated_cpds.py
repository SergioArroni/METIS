"""pgmpy 1.0 creates no CPD for a node without edges; sampling then failed with
"No CPD associated with <node>" on the AWS install (pgmpy 1.1 does create it)."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pgmpy")

from metis.sota_models.generators.bayesian_network import PgmpyBayesianNetwork as BayesianNetwork  # noqa: E402

from metis.sota_models.generators.bayesian_network import _ensure_isolated_cpds  # noqa: E402


def test_isolated_node_gets_its_smoothed_marginal():
    rng = np.random.default_rng(0)
    data = pd.DataFrame({"iso": rng.choice(["p", "q"], 400, p=[0.8, 0.2])}).astype("category")
    model = BayesianNetwork()
    model.add_nodes_from(["iso"])
    _ensure_isolated_cpds(model, data)
    cpd = model.get_cpds("iso")
    probs = dict(zip(cpd.state_names["iso"], cpd.values.ravel()))
    assert probs["p"] == pytest.approx((data["iso"] == "p").mean(), abs=0.01)
    assert sum(probs.values()) == pytest.approx(1.0)
