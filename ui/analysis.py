"""All risk calculations: statistics, eight models, betas, stress tests, backtests."""

import numpy as np
from portfolio import align_asset_returns, build_portfolio_frame, diversification_summary
from stress import (
    downside_beta,
    load_scenarios,
    MARKETS,
    price_series,
    volatility_shock,
)
from var_calculator import (
    BACKTEST_WINDOW,
    calculate_portfolio_statistics,
    estimate_beta,
    historical_worst_losses,
    min_backtest_days,
    MONTE_CARLO_BACKTEST_NAME,
    recommend_model,
)
from ui.cache import cached_all_var, cached_backtest, cached_fetch, cached_rolling_forecasts, cached_scenarios
from ui.context import export
from ui.formatting import pct_label


def compute(ctx):
    """All risk calculations: statistics, eight models, betas, stress tests, backtests."""
    asset_returns = getattr(ctx, 'asset_returns', None)
    benchmark_tickers = ctx.benchmark_tickers
    confidence_level = ctx.confidence_level
    df = ctx.df
    holding_period = ctx.holding_period
    investment_amount = ctx.investment_amount
    is_portfolio = ctx.is_portfolio
    num_sims = ctx.num_sims
    period_input = ctx.period_input
    rebalance_mode = getattr(ctx, 'rebalance_mode', None)
    returns = ctx.returns
    risk_free_pct = ctx.risk_free_pct
    symbol = ctx.symbol
    weights = getattr(ctx, 'weights', None)
    weights_now = getattr(ctx, 'weights_now', None)

    stats = calculate_portfolio_statistics(returns, risk_free_rate=risk_free_pct / 100)

    confidence_levels = (0.90, 0.95, 0.975, 0.99)
    # GARCH(1,1)-t and Student-t MLE are fitted once (cached) and reused at every confidence level
    var_by_level = {cl: cached_all_var(returns, investment_amount, cl, holding_period, num_sims) for cl in confidence_levels}
    var_selected = var_by_level[confidence_level]
    method_names = list(var_selected)
    var_hist = var_selected["Historical"]
    var_ewma = var_selected["EWMA (RiskMetrics)"]
    cl_label = pct_label(confidence_level)
    var_garch = var_selected["GARCH(1,1)-t"]
    var_cf = var_selected["Cornish-Fisher"]
    if not var_garch.get("fallback"):
        gp = var_garch["garch_params"]
        garch_text = (f"Fitted α = {gp.alpha:.3f}, β = {gp.beta:.3f} (persistence {gp.persistence:.3f}), ν = {gp.nu:.1f}; "
                      f"today's volatility {var_garch['sigma_forecast'] * np.sqrt(252):.1%} a year against a long-run {var_garch['long_run_vol'] * np.sqrt(252):.1%}.")
    else:
        garch_text = "The fit did not converge on this sample, so EWMA is shown instead."

    # Benchmark: normal beta and downside (crisis) beta over the lookback window
    is_indian = all(t.endswith((".NS", ".BO")) for t in benchmark_tickers)
    market_key = "NIFTY50" if is_indian else "SP500"
    benchmark_symbol, benchmark_name = MARKETS[market_key]
    bench_res = cached_fetch(benchmark_symbol, period=period_input)
    beta = beta_down = float("nan")
    stock_by_date = df.set_index(df["Date"].dt.normalize())["Returns"]
    if bench_res["success"]:
        bench_df = bench_res["df"]
        bench_by_date = bench_df.set_index(bench_df["Date"].dt.normalize())["Returns"]
        beta = estimate_beta(stock_by_date, bench_by_date)
        beta_down = downside_beta(stock_by_date, bench_by_date)
    beta_used = beta if np.isfinite(beta) else 1.0
    proxy_beta = beta_down if np.isfinite(beta_down) else beta_used
    worst_df = historical_worst_losses(returns, investment_amount)

    # Crisis scenarios: replayed over the full price histories (independent of the lookback window)
    market_max = cached_fetch(benchmark_symbol, period="max")
    if not is_portfolio:
        position_max = cached_fetch(symbol, period="max")
        position_history = price_series(position_max["df"]).pct_change().dropna() if position_max["success"] else stock_by_date.dropna()
    else:
        max_frames = {t: cached_fetch(t, period="max") for t in weights.index}
        if all(res["success"] for res in max_frames.values()):
            long_returns = align_asset_returns({t: res["df"] for t, res in max_frames.items()})
            long_frame = build_portfolio_frame(long_returns, weights, rebalance=rebalance_mode)
            position_history = long_frame.set_index("Date")["Returns"]
        else:
            position_history = stock_by_date.dropna()
    if market_max["success"]:
        stress_table = cached_scenarios(position_history, price_series(market_max["df"]), load_scenarios(market=market_key),
                                        proxy_beta, investment_amount)
    else:
        stress_table = None
    vol_shock_table = volatility_shock(returns, investment_amount, confidence_level, holding_period)

    # Out-of-sample backtest of every model: each day's VaR comes only from the preceding 250 days.
    # The window stays fixed; too few remaining test days are flagged rather than hidden by shrinking it.
    backtest_window = BACKTEST_WINDOW
    test_days = max(len(returns) - backtest_window, 0)
    required_days = min_backtest_days(confidence_level)
    low_power = test_days < required_days
    if test_days > 0:
        rolling = cached_rolling_forecasts(returns, confidence_level, backtest_window, num_sims)
        forecasts = rolling["var"]
        backtest_table = cached_backtest(returns, forecasts, confidence_level, rolling["es"], rolling["sigma"])
        # At one day, Monte Carlo is GARCH-t plus simulation noise, so it is scored but not recommended
        recommendation = recommend_model(backtest_table, exclude=[MONTE_CARLO_BACKTEST_NAME])
        passing_models = backtest_table.loc[backtest_table["Verdict"] == "PASS", "Method"].tolist()
    else:
        rolling, forecasts, backtest_table, passing_models = None, None, None, []
        recommendation = {"model": None, "status": "low_power"}
    recommended_model = recommendation["model"]

    def point_name(backtest_name: str) -> str:
        """Map a backtest column name to the matching model in the point-estimate table."""
        return method_names[-1] if backtest_name == MONTE_CARLO_BACKTEST_NAME else backtest_name

    if recommendation["status"] == "recommended":
        backtest_summary = (f"{', '.join(passing_models)} passed all three VaR tests at {cl_label}; "
                            f"**{recommended_model}** is recommended (lowest tick loss among the models that also pass the ES test).")
    elif recommendation["status"] == "var_only":
        backtest_summary = (f"{', '.join(passing_models)} passed all three VaR tests at {cl_label}, but every one of them "
                            f"fails the ES test. **{recommended_model}** has the lowest tick loss among them; its VaR is "
                            "supported, but its ES understates the losses beyond VaR, so treat the ES figures with caution.")
    elif recommendation["status"] == "none_pass":
        backtest_summary = (f"no model passed all three tests at {cl_label}. **{recommended_model}** has the lowest tick loss, "
                            "but its breaches are still statistically off, so treat every VaR figure here with caution.")
    else:
        backtest_summary = (f"only {test_days} out-of-sample day{'' if test_days == 1 else 's'} available; at least {required_days} are needed for a meaningful "
                            f"test at {cl_label}. Choose a 5y or max lookback.")

    # Portfolio decomposition
    if is_portfolio:
        diversification = diversification_summary(asset_returns, weights_now, investment_amount, confidence_level, holding_period)
        correlation = asset_returns[weights.index].corr()

    # Distribution diagnostics
    skewness, excess_kurtosis = stats["skewness"], stats["kurtosis"]
    jarque_bera = stats["n_obs"] / 6 * (skewness ** 2 + excess_kurtosis ** 2 / 4)
    jarque_bera_p = float(np.exp(-jarque_bera / 2))  # chi-square(2) survival function
    export(ctx, locals())
