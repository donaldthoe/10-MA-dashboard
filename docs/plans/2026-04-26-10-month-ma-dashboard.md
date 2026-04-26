# 10-Month Moving Average Dashboard Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an interactive dashboard for learning and testing a 10-month moving-average timing rule on SPY and QQQ using Stooq data.

**Architecture:** Use Streamlit for the dashboard and keep financial calculations in a small tested Python package. The UI only gathers user inputs, calls the data/backtest modules, and renders Plotly charts plus metrics tables.

**Tech Stack:** Python, Streamlit, pandas, NumPy, Plotly, pytest, Stooq daily CSV downloads.

### Task 1: Strategy Engine Tests

**Files:**
- Create: `tests/test_backtest.py`
- Create later: `ma_dashboard/backtest.py`

**Step 1: Write failing tests**

Cover monthly close versus monthly average resampling, previous-month signal lagging, leveraged liquidation, performance metrics, calendar returns, and signal flips.

**Step 2: Run tests to verify failure**

Run: `python -m pytest tests/test_backtest.py -q`

Expected: fail with `ModuleNotFoundError: No module named 'ma_dashboard'`.

### Task 2: Strategy Engine Implementation

**Files:**
- Create: `ma_dashboard/__init__.py`
- Create: `ma_dashboard/backtest.py`

**Step 1: Implement minimal functions**

Add functions for monthly observation selection, SMA signal generation, strategy return construction, drawdowns, annual metrics, calendar-year return tables, and latest signal flips.

**Step 2: Run tests**

Run: `python -m pytest tests/test_backtest.py -q`

Expected: all tests pass.

### Task 3: Stooq Loader

**Files:**
- Create: `ma_dashboard/data.py`
- Test coverage through a deterministic URL builder and schema normalization.

**Implementation notes:**
- Use `https://stooq.com/q/d/l/?s={ticker}.us&i=d`.
- Normalize columns to lowercase and use adjusted close if Stooq provides it; otherwise use close.
- Cache in Streamlit so repeated slider changes do not refetch.

### Task 4: Streamlit Dashboard

**Files:**
- Create: `app.py`
- Create: `requirements.txt`
- Create: `README.md`

**Implementation notes:**
- Sidebar controls: ticker, MA length, monthly observation type, cash yield, leverage, start and end date.
- Main charts: equity curve, drawdown profile, calendar-year returns.
- Tables: key performance metrics, recent calendar-year returns, latest signal flips.
- Strategies: Buy & Hold, MA strategy, leveraged MA strategy.

### Task 5: Verification

**Commands:**
- `python -m pytest -q`
- `python -m streamlit run app.py --server.port 8501`

**Expected:** tests pass and dashboard serves locally.
