"""
Verification suite for risk_model.py.

Runs offline (no API key, no network): every check uses simulated returns
with known properties, or a value worked out by hand / by an independent
method (numerical integration, the `arch` package itself).

    python tests/verify_engine.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import integrate, stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from risk_model import (  # noqa: E402
    fit_garch, kupiec_test, log_returns, rolling_backtest,
    var_es_garch, var_es_historical, var_es_parametric,
)

PASSED, FAILED = [], []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASSED if ok else FAILED).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def simulate_garch(n: int, omega: float, alpha: float, beta: float, nu: float | None, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    if nu:
        z = rng.standard_t(nu, n) * np.sqrt((nu - 2) / nu)
    else:
        z = rng.standard_normal(n)
    r = np.empty(n)
    s2 = omega / (1 - alpha - beta)
    for t in range(n):
        r[t] = np.sqrt(s2) * z[t]
        s2 = omega + alpha * r[t] ** 2 + beta * s2
    return pd.Series(r, index=pd.bdate_range("2015-01-01", periods=n))


# ---------------------------------------------------------------------------
print("\n1. Returns")
prices = pd.Series([100.0, 110.0, 99.0])
lr = log_returns(prices)
check("log returns = ln(P_t / P_t-1)", np.allclose(lr.values, [np.log(1.1), np.log(0.9)]))

# ---------------------------------------------------------------------------
print("\n2. GARCH(1,1) fit")
TRUE = dict(omega=2e-6, alpha=0.08, beta=0.90)
sim = simulate_garch(4000, **TRUE, nu=6, seed=7)
g = fit_garch(sim, dist="t")
check("alpha recovered within 0.03", abs(g.alpha - TRUE["alpha"]) < 0.03, f"fit {g.alpha:.3f} vs {TRUE['alpha']}")
check("beta recovered within 0.04", abs(g.beta - TRUE["beta"]) < 0.04, f"fit {g.beta:.3f} vs {TRUE['beta']}")
check("Student-t degrees of freedom near 6", g.nu is not None and 4 < g.nu < 9, f"nu = {g.nu:.2f}")
check("persistence = alpha + beta < 1", abs(g.persistence - (g.alpha + g.beta)) < 1e-12 and g.persistence < 1)

# omega is rescaled from percent^2 to decimal^2: the recursion with the returned
# parameters must reproduce arch's own conditional volatility exactly.
cv = g.conditional_vol.values
rec = np.empty_like(cv)
rec[0] = cv[0] ** 2
for t in range(1, len(cv)):
    rec[t] = g.omega + g.alpha * sim.iloc[t - 1] ** 2 + g.beta * rec[t - 1]
check("omega/alpha/beta reproduce arch's conditional volatility",
      np.allclose(np.sqrt(rec), cv, rtol=1e-8), f"max rel err {np.max(np.abs(np.sqrt(rec) / cv - 1)):.1e}")
nxt = np.sqrt(g.omega + g.alpha * sim.iloc[-1] ** 2 + g.beta * cv[-1] ** 2)
check("1-step forecast = recursion applied to the last day", np.isclose(nxt, g.forecast_vol_next, rtol=1e-8))

# ---------------------------------------------------------------------------
print("\n3. VaR / Expected Shortfall")
rng = np.random.default_rng(1)
normal = pd.Series(rng.normal(0.0005, 0.012, 200_000))
for conf in (0.95, 0.99):
    p = var_es_parametric(normal, conf)
    mu, sd = normal.mean(), normal.std(ddof=1)
    q = stats.norm.ppf(1 - conf, mu, sd)
    tail_mean, _ = integrate.quad(lambda x: x * stats.norm.pdf(x, mu, sd), -np.inf, q)
    es_num = -tail_mean / (1 - conf)
    check(f"Gaussian VaR {conf:.0%} = -quantile", np.isclose(p.var, -q, rtol=1e-10))
    check(f"Gaussian ES {conf:.0%} matches numerical integration", np.isclose(p.es, es_num, rtol=1e-6),
          f"{p.es:.6f} vs {es_num:.6f}")
    h = var_es_historical(normal, conf)
    check(f"historical VaR {conf:.0%} close to the true normal VaR", abs(h.var - p.var) / p.var < 0.02,
          f"{h.var:.5f} vs {p.var:.5f}")
    check(f"ES >= VaR (historical and Gaussian, {conf:.0%})", h.es >= h.var and p.es >= p.var)

small = pd.Series([-0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04])
h = var_es_historical(small, 0.80)
check("historical VaR on a small known sample", np.isclose(h.var, -np.percentile(small, 20)))
check("historical ES = mean loss beyond VaR", np.isclose(h.es, 0.045))

# Standardized Student-t: VaR and ES must match the unit-variance t distribution.
for nu in (4.0, 8.0):
    fit = g.__class__(conditional_vol=g.conditional_vol, forecast_vol_next=0.02, omega=g.omega,
                      alpha=g.alpha, beta=g.beta, dist="t", nu=nu, persistence=g.persistence)
    est = var_es_garch(fit, mu=0.0, confidence=0.99, dist="t", nu=nu)
    scale = np.sqrt((nu - 2) / nu)
    q = stats.t.ppf(0.01, nu) * scale
    tail, _ = integrate.quad(lambda x: x * stats.t.pdf(x / scale, nu) / scale, -np.inf, q)
    es_num = -tail / 0.01 * 0.02
    check(f"GARCH-t VaR uses unit-variance t quantile (nu={nu:g})", np.isclose(est.var, -q * 0.02, rtol=1e-10))
    check(f"GARCH-t ES matches numerical integration (nu={nu:g})", np.isclose(est.es, es_num, rtol=1e-6),
          f"{est.es:.6f} vs {es_num:.6f}")
gn = var_es_garch(fit, mu=0.0, confidence=0.99, dist="normal")
check("GARCH-normal VaR = 2.326 sigma", np.isclose(gn.var, 2.3263479 * 0.02, rtol=1e-6))
check("fat tails: t ES > normal ES at 99%", est.es > gn.es)

# ---------------------------------------------------------------------------
print("\n4. Kupiec proportion-of-failures test")
x, n, p = 5, 250, 0.01
lr_hand = -2 * ((x * np.log(p) + (n - x) * np.log(1 - p))
                - (x * np.log(x / n) + (n - x) * np.log(1 - x / n)))
k = kupiec_test(np.array([True] * x + [False] * (n - x)), 0.99)
check("LR statistic matches the hand formula", np.isclose(k.lr_stat, lr_hand), f"{k.lr_stat:.4f}")
check("p-value from chi-square(1)", np.isclose(k.p_value, 1 - stats.chi2.cdf(lr_hand, 1)))
check("5 breaches in 250 days at 99% is not rejected", not k.reject_at_95, f"p = {k.p_value:.3f}")
k = kupiec_test(np.array([True] * 10 + [False] * 240), 0.99)
check("10 breaches in 250 days at 99% is rejected (Basel red zone)", k.reject_at_95, f"p = {k.p_value:.4f}")
k = kupiec_test(np.zeros(250, dtype=bool), 0.99)
check("zero breaches handled without log(0)", np.isfinite(k.lr_stat) and np.isclose(k.lr_stat, -2 * 250 * np.log(0.99)))
k = kupiec_test(np.array([True] * 25 + [False] * 475), 0.95)
check("exactly the expected rate gives LR = 0, p = 1", np.isclose(k.lr_stat, 0) and np.isclose(k.p_value, 1))

# ---------------------------------------------------------------------------
print("\n5. Walk-forward backtest")
iid = pd.Series(np.random.default_rng(3).normal(0, 0.01, 1500), index=pd.bdate_range("2018-01-01", periods=1500))
bt = rolling_backtest(iid, "parametric", 0.95, window=250)
check("one forecast per day after the window", len(bt) == len(iid) - 250)
rate = bt["breach"].mean()
check("95% Gaussian VaR on Gaussian data breaches ~5% of days", 0.035 < rate < 0.065, f"{rate:.2%}")
check("Kupiec does not reject a correct model", not kupiec_test(bt["breach"].values, 0.95).reject_at_95)

future = iid.copy()
future.iloc[1000:] *= 5
bt2 = rolling_backtest(future, "historical", 0.99, window=250)
bt1 = rolling_backtest(iid, "historical", 0.99, window=250)
check("no look-ahead: changing future returns leaves earlier VaR unchanged",
      np.allclose(bt1["var"].iloc[:750], bt2["var"].iloc[:750]) and not np.allclose(bt1["var"].iloc[751:], bt2["var"].iloc[751:]))

# Between refits the variance must follow the GARCH recursion with fixed parameters.
gsim = simulate_garch(700, **TRUE, nu=None, seed=11)
bt_g = rolling_backtest(gsim, "garch", 0.99, window=500, garch_dist="normal", garch_refit_every=10 ** 9)
g0 = fit_garch(gsim.iloc[:500], dist="normal")
s2 = g0.forecast_vol_next ** 2
expected = []
for t in range(500, 700):
    if t > 500:
        s2 = g0.omega + g0.alpha * gsim.iloc[t - 1] ** 2 + g0.beta * s2
    expected.append(2.3263478740408408 * np.sqrt(s2))
check("GARCH VaR between refits follows sigma2_t = omega + alpha r_t-1^2 + beta sigma2_t-1",
      np.allclose(bt_g["var"].values, expected, rtol=1e-8))
check("GARCH VaR changes day to day between refits (not frozen)", bt_g["var"].iloc[:5].nunique() == 5)

bt_r = rolling_backtest(gsim, "garch", 0.99, window=500, garch_dist="normal", garch_refit_every=5)
refit_days = bt_r.iloc[::5]
check("refit days use a fresh fit on the trailing window",
      np.isclose(refit_days["var"].iloc[1],
                 2.3263478740408408 * fit_garch(gsim.iloc[5:505], dist="normal").forecast_vol_next, rtol=1e-6))

# ---------------------------------------------------------------------------
print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
if FAILED:
    print("Failed:", *FAILED, sep="\n  - ")
    sys.exit(1)
