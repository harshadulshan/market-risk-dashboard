"""
market_client.py
Data-access layer: daily price history from Twelve Data
(https://twelvedata.com) — a real, sanctioned JSON API with a free tier
(800 requests/day, 8/minute), not a page-scraping approach.

Why this instead of Stooq or yfinance: both of those free "no signup"
sources rely on scraping a website that isn't designed to be an API, and
both are currently fighting real anti-bot protection as of 2026 — Stooq's
CSV endpoint now serves a JavaScript bot-verification challenge page
instead of data for automated requests, and yfinance has its own run of
open issues (session/crumb errors) from Yahoo's side changing things
underneath it. Twelve Data is a proper API: stable JSON, documented error
messages, and a generous free tier that just needs a quick, free sign-up
(no credit card) at https://twelvedata.com to get an API key.
"""

from __future__ import annotations

import pandas as pd
import requests

TIME_SERIES_URL = "https://api.twelvedata.com/time_series"

# Twelve Data's own symbol format: plain tickers for stocks/ETFs, "BASE/QUOTE"
# for FX and crypto — no Stooq-style ".us"/"^" suffixes needed.
COMMON_SYMBOLS = {
    "S&P 500 ETF (SPY)": "SPY",
    "Dow Jones ETF (DIA)": "DIA",
    "Nasdaq-100 ETF (QQQ)": "QQQ",
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "NVIDIA (NVDA)": "NVDA",
    "Amazon (AMZN)": "AMZN",
    "Tesla (TSLA)": "TSLA",
    "Bitcoin (BTC/USD)": "BTC/USD",
    "Gold (spot, XAU/USD)": "XAU/USD",
}


class MarketClientError(RuntimeError):
    pass


def fetch_daily_prices(symbol: str, api_key: str, outputsize: int = 5000) -> pd.DataFrame:
    """Fetch daily OHLCV history for a symbol from Twelve Data.

    `outputsize` is the number of most-recent daily bars to request
    (Twelve Data's max per call is 5000, which covers ~19 years of trading
    days — plenty for this app's purposes).
    """
    if not api_key or not api_key.strip():
        raise MarketClientError(
            "No Twelve Data API key set. Get a free key at https://twelvedata.com "
            "(sign up, no credit card) and paste it into the sidebar."
        )

    params = {
        "symbol": symbol,
        "interval": "1day",
        "outputsize": outputsize,
        "apikey": api_key.strip(),
        "order": "ASC",
    }

    try:
        resp = requests.get(TIME_SERIES_URL, params=params, timeout=20)
    except requests.RequestException as e:
        raise MarketClientError(f"Could not reach Twelve Data for '{symbol}': {e}") from e

    if resp.status_code == 429:
        raise MarketClientError(
            "Twelve Data rate limit hit (free tier: 8 requests/minute, 800/day). Wait a minute and try again."
        )

    try:
        payload = resp.json()
    except ValueError as e:
        raise MarketClientError(f"Twelve Data returned a non-JSON response for '{symbol}': {resp.text[:200]!r}") from e

    if payload.get("status") == "error" or "values" not in payload:
        message = payload.get("message", "Unknown error")
        raise MarketClientError(f"Twelve Data error for '{symbol}': {message}")

    values = payload["values"]
    if not values:
        raise MarketClientError(f"No data returned for '{symbol}' — check the symbol is correct.")

    df = pd.DataFrame(values)
    numeric_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.rename(columns={"datetime": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
    df = df.sort_values("Date").set_index("Date")
    df = df.dropna(subset=["Close"])
    if df.empty:
        raise MarketClientError(f"No usable price data for '{symbol}'.")
    return df
