import pandas as pd
from scipy import stats
from typing import Dict, Any


def combine_result(
    results: Dict[str, pd.DataFrame],
    identifier: str = "id",
):
    dfs = []
    for id in results:
        df = results[id].copy()
        df[identifier] = id
        dfs.append(df)
    return pd.concat(dfs)


def select(df: pd.DataFrame, conditions: Dict[str, Any]):
    query = True
    for column, value in conditions.items():
        query &= df[column] == value
    return df.loc[query]


def mean(
    df: pd.DataFrame,
    column: str,
):
    x = df[column]
    return x.mean()


def ci95(
    df: pd.DataFrame,
    column: str,
):
    x = df[column]
    sem = stats.sem(x)
    return sem * stats.t.ppf((1 + 0.95) / 2, len(x) - 1)


def mean_ci95_by(
    df: pd.DataFrame,
    column: str,
    conditions: Dict[str, Any],
):
    df = select(df, conditions)
    m, c = mean(df, column), ci95(df, column)

    return m, c


def remove_outliers(
    df: pd.DataFrame,
    column: str,
    quantile: float = 0.1,
):
    q_low = df[column].quantile(quantile)
    q_hi = df[column].quantile(1 - quantile)
    df = df[(df[column] < q_hi) & (df[column] > q_low)]
    return df


def format_mean_ci95(
    mean: float,
    ci95: float,
    mean_max: float = 100,
    bold: bool = False,
):
    value_tmpl = "{:.2f}$_{{\\pm {:.2f}}}$"
    value_tmpl_bold = "\\textbf{{{:.2f}}}$_{{\\pm \\mathbf{{{:.2f}}}}}$"

    if mean <= mean_max:
        if bold:
            return value_tmpl_bold.format(mean, ci95)
        return value_tmpl.format(mean, ci95)
    elif mean > mean_max:
        return '$>$100'
    else:
        return None
