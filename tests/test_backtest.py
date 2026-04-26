import math
from io import BytesIO
from zipfile import ZipFile

import numpy as np
import pandas as pd

from ma_dashboard.backtest import (
    STRATEGY_BUY_HOLD,
    STRATEGY_LEVERED_MA,
    STRATEGY_MA,
    backtest_strategies,
    build_signal,
    calendar_year_returns,
    drop_incomplete_current_month,
    latest_signal_flips,
    latest_rule_signal,
    monthly_observations,
    performance_metrics,
)
from ma_dashboard.data import (
    find_local_stooq_csv,
    load_local_stooq_csv,
    load_stooq_bulk_zip,
    stooq_download_url,
    normalize_stooq_frame,
)
from ma_dashboard.ui import (
    CHART_STRATEGIES,
    DEFAULT_CASH_YIELD_PERCENT,
    DEFAULT_LEVERAGE,
    LEVERAGE_DISCLOSURE,
    STRATEGY_COLORS,
    default_window_bounds,
    window_slider_bounds,
)


def test_monthly_observations_can_use_month_end_close_or_monthly_average():
    prices = pd.Series(
        [10.0, 20.0, 30.0, 50.0],
        index=pd.to_datetime(["2020-01-02", "2020-01-31", "2020-02-03", "2020-02-28"]),
        name="close",
    )

    closes = monthly_observations(prices, "Monthly close")
    averages = monthly_observations(prices, "Monthly average")

    assert closes.tolist() == [20.0, 50.0]
    assert averages.tolist() == [15.0, 40.0]
    assert list(closes.index) == [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29")]


def test_ma_strategy_uses_previous_month_observation_against_previous_sma():
    observations = pd.Series(
        [100.0, 110.0, 90.0, 120.0],
        index=pd.date_range("2020-01-31", periods=4, freq="ME"),
    )
    monthly_returns = pd.Series(
        [0.0, 0.10, -0.1818181818, 0.3333333333],
        index=observations.index,
    )

    result = backtest_strategies(
        observations=observations,
        monthly_returns=monthly_returns,
        ma_length=2,
        cash_yield=0.0,
        leverage=2.0,
    )

    assert result.signal.tolist() == [False, False, True, False]
    assert result.returns["10m MA"].round(6).tolist() == [0.0, 0.0, -0.181818, 0.0]
    assert result.returns["10m MA on leverage"].round(6).tolist() == [0.0, 0.0, -0.363636, 0.0]


def test_backtest_can_use_pre_window_signal_history():
    observations_all = pd.Series(
        [90.0, 100.0, 110.0, 120.0],
        index=pd.date_range("1949-10-31", periods=4, freq="ME"),
    )
    monthly_returns_all = pd.Series([0.0, 0.10, 0.10, 0.10], index=observations_all.index)
    signal_all, sma_all = build_signal(observations_all, ma_length=2)
    window_index = observations_all.index[-1:]

    result = backtest_strategies(
        observations=observations_all.loc[window_index],
        monthly_returns=monthly_returns_all.loc[window_index],
        ma_length=2,
        cash_yield=0.0,
        leverage=1.25,
        signal=signal_all.loc[window_index],
        sma=sma_all.loc[window_index],
    )

    assert result.signal.tolist() == [True]
    assert result.returns["10m MA"].tolist() == [0.10]


def test_leveraged_strategy_can_be_liquidated_and_stays_at_zero():
    observations = pd.Series(
        [100.0, 120.0, 50.0, 80.0],
        index=pd.date_range("2020-01-31", periods=4, freq="ME"),
    )
    monthly_returns = pd.Series([0.0, 0.20, -0.5833333333, 0.60], index=observations.index)

    result = backtest_strategies(
        observations=observations,
        monthly_returns=monthly_returns,
        ma_length=2,
        cash_yield=0.0,
        leverage=2.0,
    )

    assert result.equity["10m MA on leverage"].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert result.liquidated is True


def test_performance_metrics_include_drawdown_and_ulcer_index():
    monthly_returns = pd.DataFrame(
        {
            "Buy & Hold": [0.10, -0.10, 0.05, 0.02],
            "10m MA": [0.0, 0.0, 0.05, 0.02],
        },
        index=pd.date_range("2020-01-31", periods=4, freq="ME"),
    )
    equity = (1 + monthly_returns).cumprod()

    metrics = performance_metrics(monthly_returns, equity)

    assert set(metrics.columns) == {
        "Strategy",
        "Total return",
        "CAGR",
        "Ann. vol",
        "Sharpe (0% rf)",
        "Max drawdown",
        "Ulcer index",
    }
    buy_hold = metrics.set_index("Strategy").loc["Buy & Hold"]
    assert buy_hold["Total return"] == equity["Buy & Hold"].iloc[-1] - 1
    assert buy_hold["Max drawdown"] < 0
    assert buy_hold["Ulcer index"] > 0
    assert math.isfinite(buy_hold["Sharpe (0% rf)"])


def test_calendar_year_returns_and_latest_signal_flips_are_reported():
    returns = pd.DataFrame(
        {
            "Buy & Hold": [0.10, 0.05, -0.20, 0.30],
            "10m MA": [0.0, 0.05, 0.0, 0.30],
        },
        index=pd.to_datetime(["2020-01-31", "2020-12-31", "2021-01-31", "2021-12-31"]),
    )
    signal = pd.Series([False, True, True, False], index=returns.index)

    calendar = calendar_year_returns(returns)
    flips = latest_signal_flips(signal, limit=3)

    assert calendar.loc[2020, "Buy & Hold"] == np.prod([1.10, 1.05]) - 1
    assert calendar.loc[2021, "10m MA"] == np.prod([1.0, 1.30]) - 1
    assert flips["Signal"].tolist() == ["Cash", "Invested"]
    assert flips["Date"].tolist() == [pd.Timestamp("2021-12-31"), pd.Timestamp("2020-12-31")]


def test_latest_rule_signal_uses_latest_completed_observation():
    observations = pd.Series([100.0, 90.0], index=pd.date_range("2026-02-28", periods=2, freq="ME"))
    sma = pd.Series([95.0, 95.0], index=observations.index)

    assert latest_rule_signal(observations, sma) is False


def test_drop_incomplete_current_month_removes_only_live_partial_month():
    prices = pd.Series(
        [100.0, 110.0, 120.0],
        index=pd.to_datetime(["2026-03-31", "2026-04-23", "2026-04-24"]),
    )

    trimmed = drop_incomplete_current_month(prices, as_of=pd.Timestamp("2026-04-26"))

    assert trimmed.index.tolist() == [pd.Timestamp("2026-03-31")]


def test_stooq_loader_builds_urls_and_normalizes_prices():
    assert stooq_download_url("SPY") == "https://stooq.com/q/d/l/?s=spy.us&i=d"
    assert (
        stooq_download_url("QQQ", api_key="abc123")
        == "https://stooq.com/q/d/l/?s=qqq.us&i=d&apikey=abc123"
    )

    raw = pd.DataFrame(
        {
            "Date": ["2020-01-02", "2020-01-03"],
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 102.0],
            "Volume": [1000, 1200],
        }
    )

    normalized = normalize_stooq_frame(raw)

    assert normalized.index.tolist() == [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")]
    assert normalized["close"].tolist() == [101.0, 102.0]


def test_stooq_bulk_zip_loader_finds_ticker_file_and_normalizes_metastock_columns():
    archive = BytesIO()
    with ZipFile(archive, "w") as zip_file:
        zip_file.writestr(
            "data/daily/us/nyse etfs/spy.us.txt",
            "<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>\n"
            "SPY.US,D,20200102,000000,100,103,99,102,1000,0\n"
            "SPY.US,D,20200103,000000,102,104,101,103,1200,0\n",
        )
    archive.seek(0)

    frame = load_stooq_bulk_zip(archive, "SPY")

    assert frame.index.tolist() == [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")]
    assert frame["close"].tolist() == [102, 103]


def test_local_csv_loader_finds_downloaded_stooq_alias_files(tmp_path):
    exact_csv_path = tmp_path / "spy_us_d.csv"
    exact_csv_path.write_text(
        "Date,Open,High,Low,Close,Volume\n"
        "2021-01-02,200,203,199,202,1000\n"
        "2021-01-03,202,204,201,203,1200\n",
        encoding="utf-8",
    )
    csv_path = tmp_path / "^spx_d.csv"
    csv_path.write_text(
        "Date,Open,High,Low,Close,Volume\n"
        "2020-01-02,100,103,99,102,1000\n"
        "2020-01-03,102,104,101,103,1200\n",
        encoding="utf-8",
    )

    assert find_local_stooq_csv("SPY", tmp_path) == exact_csv_path
    frame = load_local_stooq_csv("SPY", tmp_path)

    assert frame.index.tolist() == [pd.Timestamp("2021-01-02"), pd.Timestamp("2021-01-03")]
    assert frame["close"].tolist() == [202, 203]


def test_default_window_starts_at_1950_when_data_is_older():
    start, end = default_window_bounds(pd.to_datetime(["1938-01-31", "2026-03-31"]))

    assert start == pd.Timestamp("1950-01-01").date()
    assert end == pd.Timestamp("2026-03-31").date()


def test_window_slider_cannot_start_before_1950():
    min_value, max_value, value = window_slider_bounds(
        pd.to_datetime(["1938-01-31", "2026-03-31"])
    )

    assert min_value == pd.Timestamp("1950-01-01").date()
    assert max_value == pd.Timestamp("2026-03-31").date()
    assert value == (min_value, max_value)


def test_dashboard_chart_contract_includes_leverage_with_requested_colors():
    assert CHART_STRATEGIES == [STRATEGY_BUY_HOLD, STRATEGY_MA, STRATEGY_LEVERED_MA]
    assert STRATEGY_COLORS[STRATEGY_MA] == "#dc2626"
    assert STRATEGY_COLORS[STRATEGY_LEVERED_MA] == "#f59e0b"


def test_dashboard_defaults_use_two_percent_cash_yield():
    assert DEFAULT_CASH_YIELD_PERCENT == 2.0


def test_dashboard_defaults_use_conservative_leverage():
    assert DEFAULT_LEVERAGE == 1.25


def test_dashboard_discloses_simplified_leverage_model():
    assert "monthly return multiple" in LEVERAGE_DISCLOSURE
