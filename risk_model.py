"""
risk_model.py
Quantitative market-risk engine: GARCH(1,1) volatility modeling, Value-at-Risk
(VaR) and Expected Shortfall (ES) under three standard methodologies, and
Kupiec's proportion-of-failures backtest — the real risk-measurement
machinery used in bank/fund market-risk desks, not a simplified toy.

Why GARCH matters here: raw historical or Gaussian VaR assumes volatility
is constant, which real markets flatly contradict (volatility clusters —
calm periods and turbulent periods each persist). GARCH(1,1) models that
clustering directly, so a VaR estimate built on today's GARCH-forecast
volatility reacts to current market conditions instead of averaging over a
fixed lookback window that may be stale.

Uses the `arch` package (Kevin Sheppard) for GARCH fitting — the standard,
peer-reviewed implementation used in academic and industry practice, rather
than a hand-rolled optimizer that would be much easier to get subtly wrong.

References:
  - Bollerslev, T. (1986). "Generalized autoregressive conditional
    heteroskedasticity." Journal of Econometrics, 31(3), 307-327.
  - Kupiec, P. (1995). "Techniques for verifying the accuracy of risk
    measurement models." Journal of Derivatives, 3(2), 73-84. (The
    proportion-of-failures / unconditional-coverage backtest, referenced
    in the Basel Committee's market-risk backtesting framework.)
  - Basel Committee on Banking Supervision — market risk framework
    (motivates why VaR models must be backtested, and why post-2016
    Basel moved toward Expected Shortfall as the primary risk measure).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

ConfidenceLevel = float  # e.g. 0.95, 0.99


def log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices / prices.shift(1)).dropna()


# ---------------------------------------------------------------------------
# GARCH(1,1)
# ---------------------------------------------------------------------------

@dataclass
class GarchFit:
    conditional_vol: pd.Series      # in-sample fitted daily volatility (same units as returns, not %)
    forecast_vol_next: float        # 1-step-ahead forecast volatility
    omega: float
    alpha: float
    beta: float
    dist: str
    nu: float | None                # degrees of freedom, if Student-t
    persistence: float              # alpha + beta; close to 1 => slow-decaying volatility shocks


def fit_garch(returns: pd.Series, dist: Literal["normal", "t"] = "t") -> GarchFit:
    """Fit a GARCH(1,1) model to a return series (in decimal, e.g. 0.01 = 1%).

    Returns are rescaled to percent internally (arch's optimizer is more
    numerically stable on that scale) and rescaled back on the way out.
    """
    from arch import arch_model

    scaled = returns * 100  # percent scale, standard practice for arch's optimizer
    am = arch_model(scaled, mean="Zero", vol="Garch", p=1, q=1, dist=dist)
    res = am.fit(disp="off")

    cond_vol = res.conditional_volatility / 100  # back to decimal-return scale
    cond_vol.index = returns.index

    fcast = res.forecast(horizon=1, reindex=False)
    next_vol = float(np.sqrt(fcast.variance.values[-1, 0])) / 100

    omega = res.params.get("omega", np.nan) / 100 ** 2
    alpha = res.params.get("alpha[1]", np.nan)
    beta = res.params.get("beta[1]", np.nan)
    nu = res.params.get("nu", None)

    return GarchFit(
        conditional_vol=cond_vol, forecast_vol_next=next_vol,
        omega=float(omega), alpha=float(alpha), beta=float(beta),
        dist=dist, nu=float(nu) if nu is not None else None,
        persistence=float(alpha + beta),
    )


# ---------------------------------------------------------------------------
# VaR / Expected Shortfall
# ---------------------------------------------------------------------------

@dataclass
class RiskEstimate:
    method: str
    confidence: ConfidenceLevel
    var: float   # positive number = a loss of this magnitude (as a fraction, e.g. 0.032 = 3.2%)
    es: float    # Expected Shortfall, same convention


def var_es_historical(returns: pd.Series, confidence: ConfidenceLevel) -> RiskEstimate:
    """Historical simulation: empirical quantile of past returns, no
    distributional assumption. ES = mean of the tail beyond VaR."""
    alpha = 1 - confidence
    q = np.percentile(returns, alpha * 100)
    var = -q
    tail = returns[returns <= q]
    es = -tail.mean() if len(tail) else var
    return RiskEstimate("Historical simulation", confidence, float(var), float(es))


def var_es_parametric(returns: pd.Series, confidence: ConfidenceLevel) -> RiskEstimate:
    """Parametric (variance-covariance) VaR/ES assuming returns ~ Normal(mu, sigma)."""
    alpha = 1 - confidence
    mu, sigma = returns.mean(), returns.std(ddof=1)
    z = stats.norm.ppf(alpha)
    var = -(mu + z * sigma)
    # Closed-form Gaussian Expected Shortfall: ES = -mu + sigma * phi(z)/alpha
    es = -mu + sigma * stats.norm.pdf(z) / alpha
    return RiskEstimate("Parametric (Gaussian)", confidence, float(var), float(es))


def var_es_garch(garch: GarchFit, mu: float, confidence: ConfidenceLevel, dist: str = "t", nu: float | None = None) -> RiskEstimate:
    """One-day-ahead VaR/ES using the GARCH model's forecast volatility for
    tomorrow, rather than a fixed historical window — this is what lets the
    estimate react immediately to a current volatility regime."""
    alpha = 1 - confidence
    sigma = garch.forecast_vol_next
    if dist == "t" and nu and nu > 2:
        t_q = stats.t.ppf(alpha, df=nu)
        # scale Student-t quantile to unit variance so sigma still represents actual vol
        scale = np.sqrt((nu - 2) / nu)
        z = t_q * scale
        # Expected shortfall for standardized Student-t (Bali & Theodossiou, standard formula)
        t_pdf = stats.t.pdf(t_q, df=nu)
        es_std = (t_pdf / alpha) * ((nu + t_q ** 2) / (nu - 1)) * scale
        var = -(mu + z * sigma)
        es = -mu + es_std * sigma
    else:
        z = stats.norm.ppf(alpha)
        var = -(mu + z * sigma)
        es = -mu + sigma * stats.norm.pdf(z) / alpha
    return RiskEstimate(f"GARCH(1,1) [{dist}]", confidence, float(var), float(es))


# ---------------------------------------------------------------------------
# Kupiec proportion-of-failures backtest
# ---------------------------------------------------------------------------

@dataclass
class KupiecResult:
    n_obs: int
    n_exceptions: int
    expected_exceptions: float
    exception_rate: float
    expected_rate: float
    lr_stat: float
    p_value: float
    reject_at_95: bool  # True => model's exception rate is statistically inconsistent with its stated confidence


def kupiec_test(exceptions: np.ndarray, confidence: ConfidenceLevel) -> KupiecResult:
    """Kupiec (1995) unconditional-coverage / proportion-of-failures test.

    `exceptions` is a boolean array: True on days where the actual loss
    exceeded the VaR estimate for that day. Tests whether the observed
    exception rate is statistically consistent with the VaR model's stated
    confidence level (e.g. a 99% VaR model "should" be exceeded about 1% of
    the time — too many exceptions means the model understates risk, too
    few means it's overly conservative).
    """
    p = 1 - confidence  # expected exception probability
    n = len(exceptions)
    x = int(np.sum(exceptions))
    if n == 0:
        return KupiecResult(0, 0, 0.0, float("nan"), p, float("nan"), float("nan"), False)

    pi_hat = x / n

    def _log_lik(prob: float, successes: int, trials: int) -> float:
        prob = min(max(prob, 1e-12), 1 - 1e-12)
        return successes * np.log(prob) + (trials - successes) * np.log(1 - prob)

    ll_null = _log_lik(p, x, n)          # model is correctly calibrated
    ll_alt = _log_lik(pi_hat, x, n)      # observed rate taken at face value
    lr_stat = -2 * (ll_null - ll_alt)
    lr_stat = max(lr_stat, 0.0)
    p_value = float(1 - stats.chi2.cdf(lr_stat, df=1))

    return KupiecResult(
        n_obs=n, n_exceptions=x, expected_exceptions=p * n,
        exception_rate=pi_hat, expected_rate=p,
        lr_stat=float(lr_stat), p_value=p_value, reject_at_95=p_value < 0.05,
    )


def rolling_backtest(
    returns: pd.Series,
    method: str,
    confidence: ConfidenceLevel,
    window: int = 250,
    garch_dist: str = "t",
    garch_refit_every: int = 5,
) -> pd.DataFrame:
    """Walk-forward backtest: for each day t (after an initial `window`-day
    burn-in), estimate VaR using only data up to t-1, then check whether the
    actual return on day t breached it. This is the honest way to backtest
    a risk model — using no information from the future.

    GARCH refitting is expensive (a numerical MLE optimization per fit), so
    for `method="garch"` the model is refit only every `garch_refit_every`
    days; on the days in between, the fitted parameters are kept and the
    GARCH(1,1) variance recursion is rolled forward with each new day's
    realized return (a standard practical shortcut) — refitting on every single day over a
    multi-year backtest would make this impractically slow for an
    interactive app without materially changing the result.
    """
    dates, var_series, actual, breach = [], [], [], []
    n = len(returns)
    cached_garch: GarchFit | None = None
    sigma2 = float("nan")
    for t in range(window, n):
        train = returns.iloc[t - window:t]
        today = returns.iloc[t]
        if method == "historical":
            est = var_es_historical(train, confidence)
        elif method == "parametric":
            est = var_es_parametric(train, confidence)
        elif method == "garch":
            if cached_garch is None or (t - window) % garch_refit_every == 0:
                cached_garch = fit_garch(train, dist=garch_dist)
                sigma2 = cached_garch.forecast_vol_next ** 2
            else:
                # Between refits, roll the fitted GARCH(1,1) recursion forward one day
                # with yesterday's realized return (zero-mean model, so residual = return):
                # sigma2_t = omega + alpha * r_{t-1}^2 + beta * sigma2_{t-1}
                r_prev = float(returns.iloc[t - 1])
                sigma2 = cached_garch.omega + cached_garch.alpha * r_prev ** 2 + cached_garch.beta * sigma2
            day_fit = replace(cached_garch, forecast_vol_next=float(np.sqrt(sigma2)))
            est = var_es_garch(day_fit, mu=0.0, confidence=confidence, dist=garch_dist, nu=cached_garch.nu)
        else:
            raise ValueError(f"Unknown method: {method}")
        dates.append(returns.index[t])
        var_series.append(est.var)
        actual.append(today)
        breach.append(bool(-today > est.var))  # loss (negative return) exceeds VaR
    return pd.DataFrame({"date": dates, "var": var_series, "actual_return": actual, "breach": breach}).set_index("date")
