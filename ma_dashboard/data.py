from __future__ import annotations

from pathlib import PurePosixPath
from pathlib import Path
from io import StringIO
from typing import BinaryIO
from urllib.parse import urlencode
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd


SUPPORTED_TICKERS = ("SPY", "QQQ")
STOOQ_BASE_URL = "https://stooq.com/q/d/l/"
LOCAL_CSV_CANDIDATES = {
    "SPY": ("spy_us_d.csv", "spy.us_d.csv", "spy_d.csv", "spy.csv", "^spx_d.csv"),
    "QQQ": ("qqq_us_d.csv", "qqq.us_d.csv", "qqq_d.csv", "qqq.csv", "^ndx_d.csv"),
}


def stooq_download_url(ticker: str, api_key: str | None = None) -> str:
    symbol = ticker.upper()
    if symbol not in SUPPORTED_TICKERS:
        raise ValueError(f"Unsupported ticker: {ticker}")
    query = {"s": f"{symbol.lower()}.us", "i": "d"}
    if api_key:
        query["apikey"] = api_key.strip()
    return f"{STOOQ_BASE_URL}?{urlencode(query)}"


def normalize_stooq_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = [
        str(column).strip().lower().replace("<", "").replace(">", "") for column in normalized.columns
    ]
    if "date" not in normalized or "close" not in normalized:
        raise ValueError("Stooq data must include Date and Close columns")

    normalized["date"] = pd.to_datetime(normalized["date"].astype(str))
    normalized = normalized.sort_values("date").set_index("date")
    normalized.index.name = None
    return normalized


def load_stooq_daily(ticker: str, api_key: str | None = None) -> pd.DataFrame:
    url = stooq_download_url(ticker, api_key=api_key)
    with urlopen(url, timeout=20) as response:
        payload = response.read().decode("utf-8")
    if "Get your apikey" in payload:
        raise ValueError("Stooq requires an API key for CSV downloads. Add it in the sidebar.")
    frame = pd.read_csv(StringIO(payload))
    if frame.empty:
        raise ValueError(f"No data returned by Stooq for {ticker}")
    return normalize_stooq_frame(frame)


def load_stooq_bulk_zip(archive: BinaryIO, ticker: str) -> pd.DataFrame:
    symbol = ticker.upper()
    target_name = f"{symbol.lower()}.us.txt"
    with ZipFile(archive) as zip_file:
        candidates = [
            name for name in zip_file.namelist() if PurePosixPath(name.lower()).name == target_name
        ]
        if not candidates:
            raise ValueError(f"{symbol}.US was not found in the Stooq bulk ZIP")
        with zip_file.open(candidates[0]) as ticker_file:
            frame = pd.read_csv(ticker_file)
    return normalize_stooq_frame(frame)


def find_local_stooq_csv(ticker: str, directory: str | Path) -> Path:
    symbol = ticker.upper()
    if symbol not in LOCAL_CSV_CANDIDATES:
        raise ValueError(f"Unsupported ticker: {ticker}")
    root = Path(directory).expanduser()
    for filename in LOCAL_CSV_CANDIDATES[symbol]:
        candidate = root / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No local Stooq CSV found for {symbol} in {root}")


def load_local_stooq_csv(ticker: str, directory: str | Path) -> pd.DataFrame:
    csv_path = find_local_stooq_csv(ticker, directory)
    frame = pd.read_csv(csv_path)
    return normalize_stooq_frame(frame)
