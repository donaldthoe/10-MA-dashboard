from __future__ import annotations

from datetime import date

import pandas as pd

from ma_dashboard.backtest import STRATEGY_BUY_HOLD, STRATEGY_LEVERED_MA, STRATEGY_MA


CHART_STRATEGIES = [STRATEGY_BUY_HOLD, STRATEGY_MA, STRATEGY_LEVERED_MA]
STRATEGY_COLORS = {
    STRATEGY_BUY_HOLD: "#2563eb",
    STRATEGY_MA: "#dc2626",
    STRATEGY_LEVERED_MA: "#f59e0b",
}
DEFAULT_CASH_YIELD_PERCENT = 2.0
DEFAULT_LEVERAGE = 1.25
LEVERAGE_DISCLOSURE = (
    "Leveraged view is a monthly return multiple; it excludes financing costs, expenses, "
    "daily reset effects, margin rules, and intramonth liquidation."
)


def default_window_bounds(index: pd.DatetimeIndex, preferred_start_year: int = 1950) -> tuple[date, date]:
    if index.empty:
        raise ValueError("index must contain at least one date")
    min_date = index.min().date()
    max_date = index.max().date()
    preferred_start = date(preferred_start_year, 1, 1)
    return max(min_date, preferred_start), max_date


def window_slider_bounds(
    index: pd.DatetimeIndex, preferred_start_year: int = 1950
) -> tuple[date, date, tuple[date, date]]:
    start_floor, end = default_window_bounds(index, preferred_start_year)
    return start_floor, end, (start_floor, end)
