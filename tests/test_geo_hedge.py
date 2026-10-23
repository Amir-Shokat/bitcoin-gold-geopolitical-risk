import numpy as np
import pandas as pd
import pytest

from geo_hedge import fit, results_table, simulate_dataset
from geo_hedge.data import prepare
from geo_hedge.models import CONTROLS, to_markdown


@pytest.fixture(scope="module")
def data():
    return simulate_dataset(n=4000, gold_beta=2.0, btc_beta=0.0, seed=1)


def test_dataset_columns(data):
    for col in ["gold_ret", "btc_ret", "gold_vol", "btc_vol", "d_log_gpr", "d_log_gpr_lag1", *CONTROLS]:
        assert col in data.columns


def test_recovers_known_gpr_sensitivity(data):
    res, _ = fit(data, "gold_ret", ["d_log_gpr", *CONTROLS])
    assert res.params["d_log_gpr"] == pytest.approx(2.0, abs=0.4)
    assert res.pvalues["d_log_gpr"] < 0.01


def test_no_false_positive_for_bitcoin(data):
    res, _ = fit(data, "btc_ret", ["d_log_gpr"])
    assert res.pvalues["d_log_gpr"] > 0.01


def test_uses_hac_standard_errors(data):
    res, _ = fit(data, "gold_vol", ["d_log_gpr"])
    assert res.cov_type == "HAC"


def test_results_table_and_markdown(data):
    table = results_table(data, {"Gold": "gold_ret", "BTC": "btc_ret"}, ["d_log_gpr"])
    assert list(table.columns) == ["Gold", "BTC"]
    assert ("d_log_gpr", "coef") in table.index and ("BP test p", "") in table.index
    md = to_markdown(table)
    assert md.startswith("| | Gold | BTC |") and "d_log_gpr" in md


def test_bitcoin_sampled_on_gold_days():
    idx = pd.date_range("2024-01-01", periods=14, freq="D")      # includes weekends
    prices = pd.DataFrame({"gold": np.where(idx.day_of_week < 5, 100.0, np.nan),
                           "btc": np.linspace(100, 113, 14)}, idx)
    gpr = pd.Series(100.0, idx)
    df = prepare(prices, gpr)
    assert (df.index.day_of_week < 5).all()
    # Monday's Bitcoin return covers Friday -> Monday (three calendar days)
    monday = df.index[df.index.day_of_week == 0][1]
    friday = monday - pd.Timedelta(days=3)
    expected = 100 * np.log(prices.loc[monday, "btc"] / prices.loc[friday, "btc"])
    assert df.loc[monday, "btc_ret"] == pytest.approx(expected)


def test_too_few_observations():
    tiny = simulate_dataset(n=12)
    with pytest.raises(ValueError):
        fit(tiny, "gold_vol", ["d_log_gpr", *CONTROLS])
