"""OLS regressions of returns and volatility on GPR changes.

    y_t = a + b * dln(GPR)_t [+ c * dln(GPR)_{t-1}] [+ controls] + e_t

Standard errors are Newey-West (HAC): daily returns and especially rolling volatility are
autocorrelated, and volatility is heteroskedastic, so plain OLS p-values overstate
significance. The Breusch-Pagan p-value is reported to show when that matters.
"""

from __future__ import annotations

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

CONTROLS = ["d_term_premium", "d_credit_spread"]


def fit(df: pd.DataFrame, dependent: str, regressors: list[str], hac_lags: int = 5):
    data = df[[dependent, *regressors]].dropna()
    if len(data) < len(regressors) + 10:
        raise ValueError(f"not enough observations for {dependent} ({len(data)})")
    X = sm.add_constant(data[regressors])
    model = sm.OLS(data[dependent], X)
    result = model.fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    bp_p = het_breuschpagan(model.fit().resid, X)[1]
    return result, float(bp_p)


def results_table(df: pd.DataFrame, dependents: dict[str, str], regressors: list[str],
                  hac_lags: int = 5) -> pd.DataFrame:
    """Coefficients and HAC p-values side by side, one column pair per dependent variable."""
    columns = {}
    for label, dep in dependents.items():
        res, bp = fit(df, dep, regressors, hac_lags)
        col = {}
        for name in ["const", *regressors]:
            col[(name, "coef")] = res.params[name]
            col[(name, "p")] = res.pvalues[name]
        col[("N", "")] = res.nobs
        col[("R2", "")] = res.rsquared
        col[("BP test p", "")] = bp
        columns[label] = pd.Series(col)
    return pd.DataFrame(columns)


def to_markdown(table: pd.DataFrame, digits: int = 3) -> str:
    """Render a results table with coefficients and p-values in one cell: 0.054 (0.033)."""
    rows = []
    names = list(dict.fromkeys(i[0] for i in table.index))
    header = "| | " + " | ".join(table.columns) + " |"
    rows += [header, "|---" * (len(table.columns) + 1) + "|"]
    for name in names:
        if (name, "coef") in table.index:
            cells = []
            for c in table.columns:
                coef, p = table.loc[(name, "coef"), c], table.loc[(name, "p"), c]
                stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
                cells.append(f"{coef:.{digits}f}{stars} ({p:.3f})")
        else:
            fmt = "{:.0f}" if name == "N" else "{:.3f}"
            cells = [fmt.format(table.loc[(name, ""), c]) for c in table.columns]
        rows.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(rows)
