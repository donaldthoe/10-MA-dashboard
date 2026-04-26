from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from ma_dashboard.backtest import (
    backtest_strategies,
    build_signal,
    calendar_year_returns,
    drop_incomplete_current_month,
    latest_rule_signal,
    latest_signal_flips,
    monthly_close_returns,
    monthly_observations,
    performance_metrics,
)
from ma_dashboard.data import (
    SUPPORTED_TICKERS,
    find_local_stooq_csv,
    load_local_stooq_csv,
)
from ma_dashboard.ui import (
    CHART_HEIGHTS,
    CHART_STRATEGIES,
    DEFAULT_CASH_YIELD_PERCENT,
    DEFAULT_LEVERAGE,
    LEVERAGE_DISCLOSURE,
    MARKET_LABELS,
    STRATEGY_COLORS,
    window_slider_bounds,
)


st.set_page_config(page_title="10-Month MA Dashboard", layout="wide")

APP_DIR = Path(__file__).resolve().parent
BUNDLED_DATA_DIR = APP_DIR / "data" / "stooq"
CHART_CONFIG = {"displayModeBar": False, "responsive": True}

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 0.9rem;
        padding-bottom: 2rem;
        max-width: 1180px;
    }
    [data-testid="stDeployButton"],
    [data-testid="stHeader"],
    #MainMenu,
    footer {
        display: none;
    }
    .status-grid,
    .metric-card-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.55rem;
        margin-bottom: 0.75rem;
    }
    .metric-card-grid {
        grid-template-columns: repeat(3, minmax(0, 1fr));
    }
    .status-card,
    .metric-card {
        border: 1px solid rgba(49, 51, 63, 0.14);
        border-radius: 8px;
        padding: 0.8rem;
        background: #ffffff;
    }
    .status-card strong,
    .metric-card strong {
        display: block;
        font-size: 0.82rem;
        line-height: 1.2;
        color: rgba(49, 51, 63, 0.72);
        margin-bottom: 0.35rem;
    }
    .status-card span,
    .metric-card span {
        display: block;
        font-size: 1rem;
        line-height: 1.25;
        font-weight: 650;
        overflow-wrap: anywhere;
    }
    @media (max-width: 640px) {
        .block-container {
            padding-left: 0.75rem;
            padding-right: 0.75rem;
        }
        h1 {
            font-size: 1.48rem !important;
            line-height: 1.18 !important;
        }
        h2, h3 {
            font-size: 1.08rem !important;
            line-height: 1.25 !important;
        }
        .status-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        [data-testid="stRadio"] [role="radiogroup"] {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.45rem;
        }
        [data-testid="stRadio"] [role="radiogroup"] label {
            min-height: 44px;
            margin: 0;
        }
        .status-card,
        .metric-card {
            padding: 0.7rem;
        }
        .metric-card-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def cached_local_csv(ticker: str, directory: str, file_signature: str) -> pd.DataFrame:
    return load_local_stooq_csv(ticker, directory)


def pct(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:.2%}"


def number(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:.2f}"


def prepare_long_frame(frame: pd.DataFrame, value_name: str) -> pd.DataFrame:
    return frame.reset_index(names="Date").melt("Date", var_name="Strategy", value_name=value_name)


def compact_plotly_layout(chart, height: int = 420):
    chart.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=16, b=8),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        legend_title_text=None,
        hovermode="x unified",
    )
    return chart


def metric_cards(metrics: pd.DataFrame) -> str:
    rows = []
    for row in metrics.to_dict("records"):
        rows.append(
            f"""
            <div class="metric-card">
                <strong>{row["Strategy"]}</strong>
                <span>{row["Total return"]} total</span>
                <span>{row["CAGR"]} CAGR</span>
                <span>{row["Max drawdown"]} max DD</span>
            </div>
            """
        )
    return f'<div class="metric-card-grid">{"".join(rows)}</div>'


def status_cards(items: list[tuple[str, str]]) -> str:
    cards = [
        f"""
        <div class="status-card">
            <strong>{label}</strong>
            <span>{value}</span>
        </div>
        """
        for label, value in items
    ]
    return f'<div class="status-grid">{"".join(cards)}</div>'


st.title("Moving Average Timing Rule")
st.caption("Invest when the previous monthly observation is above its moving average; otherwise hold cash.")

ticker = st.radio(
    "Market",
    SUPPORTED_TICKERS,
    horizontal=True,
    format_func=lambda value: MARKET_LABELS[value],
)
st.caption("SPY = S&P 500 ETF. QQQ = Nasdaq 100 ETF.")

observation_type = "Monthly close"
ma_length = 10
cash_yield = DEFAULT_CASH_YIELD_PERCENT / 100.0
leverage = DEFAULT_LEVERAGE

with st.expander("Settings"):
    observation_type = st.selectbox("Monthly observation", ["Monthly close", "Monthly average"])
    ma_length = st.slider("MA length", min_value=3, max_value=24, value=10, step=1)
    cash_yield = (
        st.slider(
            "Cash yield",
            min_value=0.0,
            max_value=10.0,
            value=DEFAULT_CASH_YIELD_PERCENT,
            step=0.25,
            key="cash_yield_percent_v3",
        )
        / 100.0
    )
    leverage = st.slider(
        "Leverage view",
        min_value=1.0,
        max_value=10.0,
        value=DEFAULT_LEVERAGE,
        step=0.25,
        key="leverage_v2",
    )

try:
    data_path = find_local_stooq_csv(ticker, BUNDLED_DATA_DIR)
    data_stat = data_path.stat()
    data_signature = f"{data_path.name}:{data_stat.st_size}:{data_stat.st_mtime_ns}"
    daily = cached_local_csv(ticker, str(BUNDLED_DATA_DIR), data_signature)
except Exception as exc:
    st.error(f"Could not load bundled {ticker} data: {exc}")
    st.stop()

completed_close = drop_incomplete_current_month(daily["close"])
if completed_close.empty:
    st.error("No completed monthly data is available in the bundled Stooq file.")
    st.stop()

observations_all = monthly_observations(completed_close, observation_type)
returns_all = monthly_close_returns(completed_close).reindex(observations_all.index).fillna(0.0)
signal_all, sma_all = build_signal(observations_all, ma_length)

if len(observations_all) <= ma_length + 2:
    st.warning("Not enough monthly data for the selected moving-average length.")
    st.stop()

min_date, max_date, window_value = window_slider_bounds(observations_all.index)
with st.container(border=True):
    st.subheader("Backtest Period")
    start_date, end_date = st.slider(
        "Period",
        min_value=min_date,
        max_value=max_date,
        value=window_value,
        format="YYYY-MM",
    )

window_mask = (observations_all.index.date >= start_date) & (observations_all.index.date <= end_date)
observations = observations_all.loc[window_mask]
monthly_returns = returns_all.loc[observations.index]
signal = signal_all.loc[observations.index]
sma = sma_all.loc[observations.index]

if len(observations) <= ma_length + 2:
    st.warning("The selected window is too short for this MA length. Expand the window or lower the MA length.")
    st.stop()

result = backtest_strategies(
    observations=observations,
    monthly_returns=monthly_returns,
    ma_length=ma_length,
    cash_yield=cash_yield,
    leverage=leverage,
    signal=signal,
    sma=sma,
)
metrics = performance_metrics(result.returns, result.equity)
calendar = calendar_year_returns(result.returns)

is_latest_invested = latest_rule_signal(result.observations, result.sma)
latest_signal = "Invested" if is_latest_invested else "Cash"
signal_help = (
    f"Latest completed {observation_type.lower()} is "
    f"{'above' if is_latest_invested else 'not above'} its {ma_length}-month moving average."
)
if result.liquidated:
    st.error(f"The {leverage:.2f}x leveraged MA strategy was liquidated in this test window.")

st.html(
    status_cards(
        [
            ("Rule status", latest_signal),
            (f"Last {observation_type.lower()}", f"{result.observations.iloc[-1]:,.2f}"),
            (f"{ma_length}-month average", f"{result.sma.iloc[-1]:,.2f}"),
            ("Through", result.observations.index[-1].strftime("%Y-%m")),
        ]
    )
)
st.caption(signal_help)
st.caption(f"Bundled file: {data_path.name}")
if data_path.name.startswith("^"):
    st.caption(
        "Bundled Stooq data is an index price proxy, not dividend-adjusted ETF total-return data."
    )
st.caption(LEVERAGE_DISCLOSURE)

metric_display = metrics.copy()
for column in ["Total return", "CAGR", "Ann. vol", "Max drawdown"]:
    metric_display[column] = metric_display[column].map(pct)
metric_display["Sharpe (0% rf)"] = metric_display["Sharpe (0% rf)"].map(number)
metric_display["Ulcer index"] = metric_display["Ulcer index"].map(number)
primary_metric_display = metric_display[metric_display["Strategy"].isin(CHART_STRATEGIES)]

st.subheader("Performance Snapshot")
st.html(metric_cards(primary_metric_display))
compact_metric_display = metric_display[["Strategy", "CAGR", "Max drawdown", "Sharpe (0% rf)"]]
with st.expander("Advanced metrics"):
    st.dataframe(compact_metric_display, width="stretch", hide_index=True)
    st.dataframe(metric_display, width="stretch", hide_index=True)

equity_long = prepare_long_frame(result.equity[CHART_STRATEGIES], "Growth of $1")
equity_chart = px.line(
    equity_long,
    x="Date",
    y="Growth of $1",
    color="Strategy",
    color_discrete_map=STRATEGY_COLORS,
)
compact_plotly_layout(equity_chart, height=CHART_HEIGHTS["equity"])
equity_chart.update_yaxes(tickformat=",.2f")
st.subheader("Growth of $1")
st.plotly_chart(equity_chart, width="stretch", config=CHART_CONFIG)

drawdown_long = prepare_long_frame(result.drawdowns[CHART_STRATEGIES], "Drawdown")
drawdown_chart = px.area(
    drawdown_long,
    x="Date",
    y="Drawdown",
    color="Strategy",
    color_discrete_map=STRATEGY_COLORS,
)
compact_plotly_layout(drawdown_chart, height=CHART_HEIGHTS["drawdown"])
drawdown_chart.update_yaxes(tickformat=".0%")
st.subheader("Worst Declines")
st.plotly_chart(drawdown_chart, width="stretch", config=CHART_CONFIG)

calendar_recent = calendar[CHART_STRATEGIES].tail(6)
calendar_long = calendar_recent.reset_index(names="Year").melt(
    "Year", var_name="Strategy", value_name="Return"
)
calendar_chart = px.bar(
    calendar_long,
    x="Year",
    y="Return",
    color="Strategy",
    barmode="group",
    color_discrete_map=STRATEGY_COLORS,
)
compact_plotly_layout(calendar_chart, height=CHART_HEIGHTS["calendar"])
calendar_chart.update_yaxes(tickformat=".0%")
st.subheader("Calendar-Year Returns")
st.plotly_chart(calendar_chart, width="stretch", config=CHART_CONFIG)

recent_display = calendar[CHART_STRATEGIES].tail(8).copy()
recent_display.index.name = "Year"
for column in recent_display.columns:
    recent_display[column] = recent_display[column].map(pct)
st.subheader("Recent Calendar-Year Returns")
st.dataframe(recent_display, width="stretch", height=320)

flips = latest_signal_flips(result.signal, limit=10)
if not flips.empty:
    flips = flips.merge(
        pd.DataFrame(
            {
                "Date": result.observations.index,
                "Observation": result.observations.to_numpy(),
                "SMA": result.sma.to_numpy(),
            }
        ),
        on="Date",
        how="left",
    )
    flips["Date"] = flips["Date"].dt.strftime("%Y-%m")
    flips["Observation"] = flips["Observation"].map(lambda value: f"{value:,.2f}")
    flips["SMA"] = flips["SMA"].map(lambda value: "-" if pd.isna(value) else f"{value:,.2f}")
st.subheader("Recent Rule Changes")
st.dataframe(flips, width="stretch", hide_index=True, height=320)
