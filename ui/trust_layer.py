"""Trust calculations shared by every page: 90% ranges, grades, model risk, lookback sensitivity, ghost effect."""

import numpy as np

from trust import market_trust, scale_ranges
from ui.cache import cached_ghost, cached_lookback, cached_trust_ranges
from ui.context import export


def compute_trust(ctx):
    """90% ranges and A-D grades for every market-risk model, plus the model-risk diagnostics."""
    backtest_table = ctx.backtest_table
    confidence_level = ctx.confidence_level
    holding_period = ctx.holding_period
    investment_amount = ctx.investment_amount
    num_sims = ctx.num_sims
    point_name = ctx.point_name
    position_history = ctx.position_history
    quality = ctx.quality
    recommended_model = ctx.recommended_model
    returns = ctx.returns
    stock_by_date = ctx.stock_by_date
    var_selected = ctx.var_selected

    trust_ranges = cached_trust_ranges(returns, confidence_level, num_sims)
    ranges_table = scale_ranges(trust_ranges, var_selected, investment_amount)
    data_quality_min = float(min(q["score"] for q in quality.values())) if quality else float("nan")
    trust = market_trust(var_selected, ranges_table, backtest_table, recommended_model, point_name,
                         n_obs=len(returns), data_quality=data_quality_min, holding_period=holding_period,
                         sources=[ctx.data_note])
    trusted = trust["metrics"]
    headline_model = trust["headline_model"]
    lookback = cached_lookback(position_history, investment_amount, confidence_level, holding_period)
    window = stock_by_date.dropna()
    ghost = cached_ghost(window, position_history, confidence_level, investment_amount * np.sqrt(holding_period))
    export(ctx, locals())
