"""Data: GPR index (Caldara & Iacoviello), Bitcoin and gold prices (Yahoo Finance) and
US term-premium and credit-spread controls (FRED).

All variables are aligned on gold trading days:

* ``gold_ret``, ``btc_ret``   log returns in percent
* ``gold_vol``, ``btc_vol``   5-day rolling standard deviation of those returns (percent)
* ``d_log_gpr``               change in the log daily GPR index
* ``d_term_premium``          daily change in the 10-year term premium (Kim-Wright, pp)
* ``d_credit_spread``         daily change in the Moody's Baa - 10y Treasury spread (pp)
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import requests

GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
FRED_SERIES = {"term_premium": "THREEFYTP10", "credit_spread": "BAA10Y"}
HEADERS = {"User-Agent": "Mozilla/5.0 (research script)"}


def fetch_gpr(timeout: int = 120) -> pd.Series:
    """Daily Geopolitical Risk index (GPRD), 1985 to the latest update."""
    resp = requests.get(GPR_URL, timeout=timeout, headers=HEADERS)
    resp.raise_for_status()
    raw = pd.read_excel(io.BytesIO(resp.content))
    raw = raw.dropna(subset=["GPRD"])
    raw.index = pd.to_datetime(raw["DAY"].astype(int).astype(str), format="%Y%m%d")
    return raw["GPRD"].astype(float).rename("gpr")


def fetch_fred(series_id: str, timeout: int = 120) -> pd.Series:
    # FRED stalls requests with an unrecognised browser-like User-Agent; use the library default
    resp = requests.get(FRED_URL.format(series=series_id), timeout=timeout)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), na_values=".")
    df.columns = ["date", series_id]
    return df.set_index(pd.to_datetime(df["date"]))[series_id].astype(float)


def fetch_prices(start: str, end: str | None = None) -> pd.DataFrame:
    import yfinance as yf

    data = yf.download(["GC=F", "BTC-USD"], start=start, end=end, auto_adjust=True, progress=False)["Close"]
    data.index = pd.to_datetime(data.index).tz_localize(None)
    return data.rename(columns={"GC=F": "gold", "BTC-USD": "btc"})


def prepare(prices: pd.DataFrame, gpr: pd.Series, controls: pd.DataFrame | None = None,
            vol_window: int = 5) -> pd.DataFrame:
    """Turn raw levels into the regression dataset (gold trading days only)."""
    days = prices["gold"].dropna().index
    df = pd.DataFrame(index=days)
    for name in ("gold", "btc"):
        if name in prices:
            # sample Bitcoin on gold trading days so both returns span the same calendar interval
            level = prices[name].reindex(days).ffill(limit=3)
            df[f"{name}_ret"] = 100 * np.log(level).diff()
            df[f"{name}_vol"] = df[f"{name}_ret"].rolling(vol_window).std()
    df["d_log_gpr"] = np.log(gpr.reindex(days, method="ffill")).diff()
    if controls is not None:
        aligned = controls.reindex(days, method="ffill")
        for col in aligned:
            df[f"d_{col}"] = aligned[col].diff()
    df["d_log_gpr_lag1"] = df["d_log_gpr"].shift(1)
    return df


def build_dataset(start: str = "2014-09-17", end: str | None = None, controls: bool = True,
                  vol_window: int = 5) -> pd.DataFrame:
    prices = fetch_prices(start, end)
    gpr = fetch_gpr()
    ctrl = None
    if controls:
        ctrl = pd.DataFrame({name: fetch_fred(sid) for name, sid in FRED_SERIES.items()})
    df = prepare(prices, gpr, ctrl, vol_window)
    return df.loc[start:end]


def simulate_dataset(n: int = 2500, gold_beta: float = 0.5, btc_beta: float = 0.0,
                     seed: int | None = 0) -> pd.DataFrame:
    """Synthetic data with known GPR sensitivities, for tests."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2014-01-01", periods=n)
    gpr = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.25, n)) * 0.1 + rng.normal(0, 0.3, n)), idx)
    d_gpr = np.log(gpr).diff().fillna(0).to_numpy()
    controls = pd.DataFrame({"term_premium": np.cumsum(rng.normal(0, 0.03, n)),
                             "credit_spread": 2 + np.cumsum(rng.normal(0, 0.02, n))}, idx)
    gold_ret = gold_beta * d_gpr + rng.normal(0, 0.9, n)
    btc_ret = btc_beta * d_gpr + rng.normal(0, 3.5, n)
    prices = pd.DataFrame({"gold": 1200 * np.exp(np.cumsum(gold_ret) / 100),
                           "btc": 400 * np.exp(np.cumsum(btc_ret) / 100)}, idx)
    return prepare(prices, gpr, controls)
