"""
Market Risk & Volatility Dashboard
-------------------------------------
Real quantitative risk measurement: GARCH(1,1) volatility modeling, Value-at-
Risk and Expected Shortfall under three standard methodologies (historical
simulation, parametric/Gaussian, and GARCH-based), and Kupiec's
proportion-of-failures backtest — the actual tools bank and fund market-risk
desks use, applied to real daily price data.

Why this problem is a genuine real-world need: every regulated financial
institution is required to measure and backtest its market risk this way
(Basel Committee framework). A risk number that hasn't been backtested is
just a guess with decimal places — this app makes that backtesting step
visible and central, not an afterthought.

Run with:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from market_client import COMMON_SYMBOLS, MarketClientError, fetch_daily_prices
from risk_model import (
    fit_garch, kupiec_test, log_returns, rolling_backtest,
    var_es_garch, var_es_historical, var_es_parametric,
)

st.set_page_config(page_title="Market Risk & Volatility Dashboard", layout="wide", page_icon="\U0001F4C9")


@st.cache_data(ttl=1800, show_spinner=False)
def cached_prices(symbol: str, api_key: str):
    return fetch_daily_prices(symbol, api_key=api_key)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_garch(returns_key: str, returns: pd.Series, dist: str):
    return fit_garch(returns, dist=dist)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_backtest(returns_key: str, returns: pd.Series, method: str, confidence: float, window: int, dist: str):
    return rolling_backtest(returns, method=method, confidence=confidence, window=window, garch_dist=dist)


st.title("\U0001F4C9 Market Risk & Volatility Dashboard")
st.caption(
    "GARCH(1,1) volatility modeling + Value-at-Risk / Expected Shortfall, backtested with the "
    "Kupiec proportion-of-failures test — the real risk-quantification toolkit used in market-risk management."
)

st.sidebar.title("Data source")
api_key = st.sidebar.text_input(
    "Twelve Data API key",
    type="password",
    help="Free, no credit card — sign up at twelvedata.com and paste your key here. Not saved anywhere; only kept for this browser session.",
)
if not api_key:
    st.sidebar.info(
        "Get a free API key at [twelvedata.com](https://twelvedata.com) (free tier: 800 requests/day) "
        "and paste it above to fetch live price data."
    )

st.sidebar.title("Instrument")
choice = st.sidebar.selectbox("Pick a common instrument, or choose 'Custom' to type any symbol",
                               list(COMMON_SYMBOLS.keys()) + ["Custom symbol..."])
if choice == "Custom symbol...":
    symbol = st.sidebar.text_input("Symbol (e.g. 'AAPL', 'SPY', 'BTC/USD')", "AAPL")
else:
    symbol = COMMON_SYMBOLS[choice]

confidence = st.sidebar.select_slider("Confidence level", options=[0.90, 0.95, 0.975, 0.99], value=0.95)
garch_dist = st.sidebar.radio("GARCH innovation distribution", ["t", "normal"], index=0,
                               help="Student-t captures the 'fat tails' real financial returns have; Normal is the textbook-simple baseline.")

if not api_key:
    st.info("Enter a Twelve Data API key in the sidebar to load price data (see instructions there).")
    st.stop()

try:
    with st.spinner(f"Fetching daily price history for {symbol}..."):
        prices_df = cached_prices(symbol, api_key)
except MarketClientError as e:
    st.error(str(e))
    st.stop()

returns = log_returns(prices_df["Close"])
if len(returns) < 300:
    st.warning(f"Only {len(returns)} days of data available — results (especially the backtest) will be noisy with this little history.")

tab_vol, tab_risk, tab_backtest, tab_about = st.tabs(
    ["\U0001F4C8 Volatility", "⚠️ VaR / ES", "✅ Backtest", "ℹ️ Methodology"]
)

# --- Volatility ------------------------------------------------------------
with tab_vol:
    st.subheader(f"{symbol} — price and volatility")
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=prices_df.index, y=prices_df["Close"], name="Close price"))
    fig_price.update_layout(title="Price history", height=350)
    st.plotly_chart(fig_price, width="stretch")

    with st.spinner("Fitting GARCH(1,1)..."):
        garch = cached_garch(symbol + garch_dist, returns, garch_dist)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("GARCH alpha", f"{garch.alpha:.3f}", help="Reaction to yesterday's shock")
    c2.metric("GARCH beta", f"{garch.beta:.3f}", help="Persistence of past volatility")
    c3.metric("alpha + beta", f"{garch.persistence:.3f}", help="Closer to 1 = shocks decay slowly (long volatility memory)")
    c4.metric("Next-day forecast vol", f"{garch.forecast_vol_next*100:.2f}%")

    realized_vol = returns.rolling(21).std()
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(x=realized_vol.index, y=realized_vol.values * 100, name="21-day realized volatility (%)", opacity=0.6))
    fig_vol.add_trace(go.Scatter(x=garch.conditional_vol.index, y=garch.conditional_vol.values * 100,
                                  name="GARCH(1,1) conditional volatility (%)", line=dict(color="crimson")))
    fig_vol.update_layout(title="Realized vs. GARCH-modeled volatility — note the clustering", height=400, yaxis_title="Daily volatility (%)")
    st.plotly_chart(fig_vol, width="stretch")
    st.caption(
        "Volatility clustering: calm periods and turbulent periods each persist. A fixed-window "
        "realized-volatility estimate reacts slowly to regime changes; GARCH reacts within a day or two."
    )

# --- VaR / ES --------------------------------------------------------------
with tab_risk:
    st.subheader(f"Value-at-Risk & Expected Shortfall — {confidence:.1%} confidence, 1-day horizon")
    hist_est = var_es_historical(returns, confidence)
    param_est = var_es_parametric(returns, confidence)
    garch_est = var_es_garch(garch, mu=float(returns.mean()), confidence=confidence, dist=garch_dist, nu=garch.nu)

    comp_df = pd.DataFrame([
        {"Method": e.method, "VaR": e.var, "Expected Shortfall": e.es}
        for e in [hist_est, param_est, garch_est]
    ])
    st.dataframe(
        comp_df.style.format({"VaR": "{:.2%}", "Expected Shortfall": "{:.2%}"}),
        width="stretch", hide_index=True,
    )
    st.caption(
        "VaR: the loss you'd expect to exceed only (1 - confidence) of the time. "
        "Expected Shortfall: the *average* loss on those worst days — always ≥ VaR, and the "
        "measure Basel's post-2016 framework treats as primary, precisely because VaR says nothing "
        "about how bad the tail beyond it actually gets."
    )

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(x=comp_df["Method"], y=comp_df["VaR"] * 100, name="VaR"))
    fig_bar.add_trace(go.Bar(x=comp_df["Method"], y=comp_df["Expected Shortfall"] * 100, name="Expected Shortfall"))
    fig_bar.update_layout(barmode="group", title="VaR vs. ES by method", yaxis_title="% of position value", height=400)
    st.plotly_chart(fig_bar, width="stretch")

    st.info(
        f"**Reading this**: at {confidence:.0%} confidence, the GARCH-based model currently estimates a "
        f"1-day VaR of **{garch_est.var:.2%}** — meaning today's specific volatility regime, not a long-run "
        f"average, is driving the number. Compare this to the historical/parametric estimates to see how "
        f"much the current regime differs from the full-sample average."
    )

# --- Backtest ---------------------------------------------------------------
with tab_backtest:
    st.subheader("Kupiec proportion-of-failures backtest")
    st.caption(
        "Walk-forward: each day's VaR is estimated using only data available *before* that day, then "
        "checked against what actually happened — no lookahead. A model is 'rejected' when its exception "
        "rate is statistically inconsistent with its stated confidence level."
    )
    bt_method = st.selectbox("Method to backtest", ["historical", "parametric", "garch"], index=1)
    window = st.slider("Rolling estimation window (trading days)", 100, 500, 250, 50)
    available = len(returns) - window
    if available < 100:
        st.warning(
            f"Not enough history for a backtest: {len(returns)} returns loaded, but the "
            f"{window}-day window plus at least 100 test days needs {window + 100}. "
            "Pick a smaller window or load a longer price history."
        )
    else:
        max_days = st.slider("Backtest length (most recent N days)", 100, min(1000, available), min(500, available)) if available > 100 else 100

        bt_returns = returns.iloc[-(max_days + window):]
        with st.spinner(f"Running walk-forward backtest ({bt_method})... this may take a moment for GARCH."):
            bt_df = cached_backtest(symbol + bt_method + str(confidence) + str(window), bt_returns, bt_method, confidence, window, garch_dist)

        kup = kupiec_test(bt_df["breach"].values, confidence)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Observations", kup.n_obs)
        c2.metric("Exceptions (breaches)", kup.n_exceptions, help=f"Expected: ~{kup.expected_exceptions:.1f}")
        c3.metric("Exception rate", f"{kup.exception_rate:.2%}", help=f"Target: {kup.expected_rate:.2%}")
        c4.metric("Kupiec test", "REJECTED" if kup.reject_at_95 else "Not rejected", f"p = {kup.p_value:.3f}")

        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=bt_df.index, y=-bt_df["actual_return"] * 100, name="Actual daily loss (%)", opacity=0.5))
        fig_bt.add_trace(go.Scatter(x=bt_df.index, y=bt_df["var"] * 100, name="VaR estimate (%)", line=dict(color="crimson")))
        breaches = bt_df[bt_df["breach"]]
        fig_bt.add_trace(go.Scatter(x=breaches.index, y=-breaches["actual_return"] * 100, mode="markers",
                                     name="Breach", marker=dict(color="#f59e0b", size=9, symbol="x")))
        fig_bt.update_layout(title=f"{bt_method} VaR backtest — actual loss vs. estimated VaR", height=450, yaxis_title="% loss")
        st.plotly_chart(fig_bt, width="stretch")

        if kup.reject_at_95:
            st.error(
                f"The Kupiec test **rejects** this model at the 95% test level (p = {kup.p_value:.3f}): its "
                f"exception rate ({kup.exception_rate:.2%}) is statistically inconsistent with its stated "
                f"{confidence:.0%} confidence. In practice this means the model is miscalibrated for this "
                f"instrument/period — too many breaches means it understates risk; too few means it's overly conservative."
            )
        else:
            st.success(
                f"The Kupiec test does **not** reject this model (p = {kup.p_value:.3f}) — its exception rate "
                f"is statistically consistent with its stated confidence level over this backtest period."
            )

# --- About -----------------------------------------------------------------
with tab_about:
    st.markdown(
        """
### The real-world problem

Every bank, fund, and clearinghouse that takes market risk is required to
quantify it — "how much could we plausibly lose tomorrow?" — and, crucially,
to *prove* that quantification is honest by backtesting it against what
actually happened. A risk number nobody has checked against reality is not
a risk management tool, it's decoration. This is precisely what the Basel
Committee's market-risk framework requires of regulated institutions, and
it's the real discipline this dashboard demonstrates.

### The methods

**GARCH(1,1)** (Bollerslev, 1986): models volatility as reacting to
yesterday's shock (`alpha`) and persisting from yesterday's volatility
(`beta`):

    sigma_t^2 = omega + alpha * epsilon_(t-1)^2 + beta * sigma_(t-1)^2

Fit via the `arch` package (Kevin Sheppard) — the standard implementation
used in academic and industry practice — with a Student-t innovation
distribution option, since real financial returns have fatter tails than
the Normal distribution assumes.

**Value-at-Risk (VaR)**, three ways:
- *Historical simulation*: the empirical quantile of past returns — no
  distributional assumption, but assumes the past sample represents the
  future.
- *Parametric (Gaussian)*: assumes returns are Normally distributed —
  simple, but understates tail risk for most real assets.
- *GARCH-based*: uses tomorrow's GARCH-forecast volatility rather than a
  fixed historical window, so the estimate reacts to the current
  volatility regime.

**Expected Shortfall (ES)**: the average loss *given* that VaR was
breached — always ≥ VaR, and the measure Basel's post-2016 Fundamental
Review of the Trading Book adopted as primary, specifically because VaR
alone says nothing about how severe the tail beyond it is.

**Kupiec (1995) proportion-of-failures test**: a likelihood-ratio test of
whether a VaR model's observed exception rate is statistically consistent
with its stated confidence level — the actual backtesting method
referenced in the Basel framework.

### Data source

Daily price history from [Twelve Data](https://twelvedata.com) — a real
JSON API with a free tier (800 requests/day), requiring a quick free
sign-up for an API key. Chosen over scraping-based free sources
(Stooq, `yfinance`): both are currently fighting real anti-bot protection
as of 2026 — Stooq's CSV endpoint now serves a JavaScript verification
challenge instead of data for automated requests, and yfinance has its
own open issues from Yahoo changing things on their end. A proper API,
even one requiring a free key, is the more honest and reliable choice
right now.

### Honest limitations

1. **GARCH(1,1) is a workhorse model, not the state of the art** — real
   trading desks often use richer variants (EGARCH/GJR-GARCH for leverage
   effects, multivariate GARCH for portfolios). This is a genuine, correct
   implementation of the standard model, not a toy — but it's one model
   among a larger family.
2. **The GARCH backtest re-estimates its parameters every 5 days, not
   daily.** Refitting a maximum-likelihood model every day over a
   multi-year backtest is too slow for an interactive app. Between refits
   the parameters are held, but the volatility forecast is still updated
   every day from the previous day's return, as the GARCH recursion says.
3. **Backtesting on one instrument's own history is not the same as
   validating a model for regulatory use** — real model validation covers
   many instruments, market regimes, and out-of-time periods.
4. **Kupiec's test has known power limitations** — it can fail to detect
   *clustered* exceptions (many breaches in a row, which is worse than the
   same number spread evenly). This is a real, published critique of the
   test, which more advanced backtests (Christoffersen's conditional
   coverage test) address — a natural next step.
        """
    )

st.divider()
st.caption(
    "Prototype build. Real next steps: EGARCH/GJR-GARCH for asymmetric volatility response, "
    "Christoffersen's conditional-coverage test (checks for exception clustering, not just rate), "
    "and multi-asset portfolio VaR via a fitted covariance/copula structure."
)
