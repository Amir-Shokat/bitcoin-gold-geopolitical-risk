"""Download the data and run every specification; writes docs/results.md and a figure.

    python scripts/run_study.py                 # 2014-09-17 to 2024-12-31 (the paper's sample)
    python scripts/run_study.py --end 2026-10-01
    python scripts/run_study.py --no-controls   # if FRED is unreachable
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from geo_hedge.data import build_dataset, fetch_gpr
from geo_hedge.models import CONTROLS, results_table, to_markdown

RET = {"Gold return": "gold_ret", "Bitcoin return": "btc_ret"}
VOL = {"Gold volatility": "gold_vol", "Bitcoin volatility": "btc_vol"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="2014-09-17")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--no-controls", action="store_true")
    p.add_argument("--out", default="docs")
    a = p.parse_args()

    df = build_dataset(a.start, a.end, controls=not a.no_controls)
    controls = [] if a.no_controls else CONTROLS
    base = ["d_log_gpr", *controls]
    sections = []

    def add(title, frame, regressors, deps=None):
        deps = deps or {**RET, **VOL}
        table = results_table(frame, deps, regressors)
        sections.append(f"### {title}\n\n{to_markdown(table)}\n")
        print(f"\n{title}\n{table.round(3).to_string()}")

    add(f"1. GPR only, {a.start} to {a.end}", df, ["d_log_gpr"])
    add("2. With term-premium and credit-spread controls", df, base) if controls else None
    add("3. Adding one lag of the GPR change", df, ["d_log_gpr", "d_log_gpr_lag1", *controls])
    add("4. Proposal sample, June 2018 to December 2023", df.loc["2018-06-01":"2023-12-01"], base)
    add("5a. Sub-sample 2014 to 2019", df.loc[:"2019-06-30"], base)
    add("5b. Sub-sample 2019 to 2024", df.loc["2019-07-01":], base)
    shocks = df[df["d_log_gpr"] >= df["d_log_gpr"].quantile(0.9)]
    add("6. Days with the largest 10% of GPR increases", shocks, base)

    out = Path(a.out)
    out.mkdir(exist_ok=True)
    header = (f"# Results\n\nSample {df.index[0].date()} to {df.index[-1].date()}, {len(df)} gold trading days.\n"
              "Coefficients with Newey-West (5 lags) p-values in brackets. * 10%, ** 5%, *** 1%.\n"
              "Returns and volatility are in percent; `d_log_gpr` is the change in the log GPR index.\n\n")
    (out / "results.md").write_text(header + "\n".join(sections))

    gpr = fetch_gpr().loc[a.start:a.end]
    prices = (df[["gold_ret", "btc_ret"]].fillna(0).cumsum() / 100).apply(lambda s: 100 * (2.718281828 ** s))
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(gpr.index, gpr, lw=0.4, color="grey", label="daily GPR")
    ax[0].plot(gpr.index, gpr.rolling(30).mean(), color="black", label="30-day mean")
    ax[0].set_title("Geopolitical Risk index (Caldara & Iacoviello)"); ax[0].legend()
    ax[1].plot(prices.index, prices["gold_ret"], label="Gold")
    ax[1].plot(prices.index, prices["btc_ret"], label="Bitcoin")
    ax[1].set_yscale("log"); ax[1].set_title("Growth of 100 (log scale)"); ax[1].legend()
    fig.tight_layout(); fig.savefig(out / "gpr_and_prices.png", dpi=120)
    print(f"\nWrote {out / 'results.md'}")


if __name__ == "__main__":
    main()
