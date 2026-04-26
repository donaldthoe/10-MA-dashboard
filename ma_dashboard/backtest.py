from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


STRATEGY_BUY_HOLD = "Buy & Hold"
STRATEGY_MA = "10m MA"
STRATEGY_LEVERED_MA = "10m MA on leverage"


@dataclass(frozen=True)
class BacktestResult:
    observations: pd.Series
    sma: pd.Series
    signal: pd.Series
    returns: pd.DataFrame
    equity: pd.DataFrame
    drawdowns: pd.DataFrame
    liquidated: bool


def monthly_observations(prices: pd.Series, observation_type: str) -> pd.Series:
    cleaned = prices.dropna().sort_index().astype(float)
    monthly = cleaned.resample("ME")
    if observation_type == "Monthly average":
        observations = monthly.mean()
    elif observation_type == "Monthly close":
        observations = monthly.last()
    else:
        raise ValueError("observation_type must be 'Monthly close' or 'Monthly average'")
    return observations.dropna().rename("Observation")


def drop_incomplete_current_month(prices: pd.Series, as_of: pd.Timestamp | None = None) -> pd.Series:
    cleaned = prices.dropna().sort_index()
    if cleaned.empty:
        return cleaned

    today = (pd.Timestamp.today() if as_of is None else pd.Timestamp(as_of)).normalize()
    last_date = pd.Timestamp(cleaned.index.max()).normalize()
    if last_date.to_period("M") != today.to_period("M"):
        return cleaned

    month_end = last_date + pd.offsets.MonthEnd(0)
    if last_date < month_end:
        return cleaned.loc[cleaned.index.to_period("M") != last_date.to_period("M")]
    return cleaned


def monthly_close_returns(prices: pd.Series) -> pd.Series:
    closes = monthly_observations(prices, "Monthly close")
    return closes.pct_change().fillna(0.0).rename("Monthly return")


def build_signal(observations: pd.Series, ma_length: int) -> tuple[pd.Series, pd.Series]:
    if ma_length < 1:
        raise ValueError("ma_length must be at least 1")
    sma = observations.rolling(ma_length, min_periods=ma_length).mean().rename("SMA")
    invested = (observations.shift(1) > sma.shift(1)).fillna(False)
    return invested.astype(bool).rename("Signal"), sma


def backtest_strategies(
    observations: pd.Series,
    monthly_returns: pd.Series,
    ma_length: int,
    cash_yield: float,
    leverage: float,
    signal: pd.Series | None = None,
    sma: pd.Series | None = None,
) -> BacktestResult:
    observations = observations.dropna().sort_index().astype(float)
    monthly_returns = monthly_returns.reindex(observations.index).fillna(0.0).astype(float)
    if signal is None or sma is None:
        signal, sma = build_signal(observations, ma_length)
    else:
        signal = signal.reindex(observations.index).fillna(False).astype(bool).rename("Signal")
        sma = sma.reindex(observations.index).rename("SMA")
    cash_monthly = (1.0 + cash_yield) ** (1.0 / 12.0) - 1.0

    buy_hold = monthly_returns
    ma_returns = monthly_returns.where(signal, cash_monthly)
    levered_returns, liquidated = _levered_returns(monthly_returns, signal, cash_monthly, leverage)

    returns = pd.DataFrame(
        {
            STRATEGY_BUY_HOLD: buy_hold,
            STRATEGY_MA: ma_returns,
            STRATEGY_LEVERED_MA: levered_returns,
        }
    )
    equity = (1.0 + returns).cumprod()
    equity = equity.where(equity > 0.0, 0.0)
    return BacktestResult(
        observations=observations,
        sma=sma,
        signal=signal,
        returns=returns,
        equity=equity,
        drawdowns=drawdown_frame(equity),
        liquidated=liquidated,
    )


def _levered_returns(
    monthly_returns: pd.Series,
    signal: pd.Series,
    cash_monthly: float,
    leverage: float,
) -> tuple[pd.Series, bool]:
    if leverage < 1.0:
        raise ValueError("leverage must be at least 1")

    values: list[float] = []
    liquidated = False
    for date, base_return in monthly_returns.items():
        if liquidated:
            values.append(0.0)
            continue
        if not bool(signal.loc[date]):
            values.append(cash_monthly)
            continue

        levered = float(base_return) * leverage
        if levered <= -1.0:
            values.append(-1.0)
            liquidated = True
        else:
            values.append(levered)

    return pd.Series(values, index=monthly_returns.index, name=STRATEGY_LEVERED_MA), liquidated


def drawdown_frame(equity: pd.DataFrame) -> pd.DataFrame:
    peaks = equity.cummax()
    return (equity / peaks - 1.0).fillna(0.0)


def performance_metrics(returns: pd.DataFrame, equity: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy in returns.columns:
        series = returns[strategy].dropna()
        curve = equity[strategy].dropna()
        years = max(len(series) / 12.0, 1.0 / 12.0)
        total_return = curve.iloc[-1] - 1.0 if len(curve) else np.nan
        cagr = (curve.iloc[-1] ** (1.0 / years) - 1.0) if len(curve) and curve.iloc[-1] > 0 else -1.0
        ann_vol = series.std(ddof=0) * np.sqrt(12.0)
        sharpe = (series.mean() * 12.0 / ann_vol) if ann_vol > 0 else np.nan
        drawdowns = curve / curve.cummax() - 1.0
        max_drawdown = drawdowns.min()
        ulcer = float(np.sqrt(np.mean(np.square(np.minimum(drawdowns.to_numpy(), 0.0) * 100.0))))
        rows.append(
            {
                "Strategy": strategy,
                "Total return": total_return,
                "CAGR": cagr,
                "Ann. vol": ann_vol,
                "Sharpe (0% rf)": sharpe,
                "Max drawdown": max_drawdown,
                "Ulcer index": ulcer,
            }
        )
    return pd.DataFrame(rows)


def calendar_year_returns(returns: pd.DataFrame) -> pd.DataFrame:
    grouped = returns.groupby(returns.index.year)
    return grouped.apply(lambda frame: (1.0 + frame).prod() - 1.0)


def latest_rule_signal(observations: pd.Series, sma: pd.Series) -> bool:
    if observations.empty or sma.empty:
        return False
    latest_observation = float(observations.iloc[-1])
    latest_sma = sma.reindex(observations.index).iloc[-1]
    return bool(pd.notna(latest_sma) and latest_observation > float(latest_sma))


def latest_signal_flips(signal: pd.Series, limit: int = 10) -> pd.DataFrame:
    changed = signal.ne(signal.shift(1)).fillna(False)
    changed.iloc[0] = False
    flips = signal[changed].sort_index(ascending=False).head(limit)
    return pd.DataFrame(
        {
            "Date": flips.index,
            "Signal": np.where(flips.to_numpy(dtype=bool), "Invested", "Cash"),
        }
    )
