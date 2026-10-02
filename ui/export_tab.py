"""Excel Report tab: workbook and CSV downloads."""

import streamlit as st
from datetime import datetime
from excel_exporter import generate_excel_var_report


def render(ctx, tab5):
    """Excel Report tab: workbook and CSV downloads."""
    alignment = getattr(ctx, 'alignment', None)
    backtest_table = ctx.backtest_table
    backtest_window = ctx.backtest_window
    benchmark_name = ctx.benchmark_name
    beta = ctx.beta
    beta_down = ctx.beta_down
    company_name = ctx.company_name
    confidence_level = ctx.confidence_level
    confidence_levels = ctx.confidence_levels
    correlation = getattr(ctx, 'correlation', None)
    currency = ctx.currency
    data_note = ctx.data_note
    decomposition = getattr(ctx, 'decomposition', None)
    df = ctx.df
    diversification = getattr(ctx, 'diversification', None)
    holding_period = ctx.holding_period
    investment_amount = ctx.investment_amount
    is_portfolio = ctx.is_portfolio
    method_names = ctx.method_names
    recommendation = ctx.recommendation
    risk_free_pct = ctx.risk_free_pct
    stress_table = ctx.stress_table
    symbol = ctx.symbol
    var_by_level = ctx.var_by_level
    vol_shock_table = ctx.vol_shock_table
    worst_df = ctx.worst_df

    with tab5:
        st.markdown("### 📥 Download Excel Risk Report")
        st.write("A formatted `.xlsx` workbook with every model's VaR and ES, the out-of-sample backtest, β-adjusted stress tests, worst historical losses, the positions with each holding's data-quality score, and the raw price data.")

        excel_bytes = generate_excel_var_report(
            symbol=symbol,
            company_name=company_name,
            currency=currency,
            investment=investment_amount,
            confidence_level=confidence_level,
            holding_period=holding_period,
            df_data=df,
            var_by_level={m: {cl: var_by_level[cl][m] for cl in confidence_levels} for m in method_names},
            backtest_table=backtest_table,
            backtest_window=backtest_window,
            stress_table=stress_table,
            vol_shock_table=vol_shock_table,
            beta_down=beta_down,
            worst_df=worst_df,
            benchmark_name=benchmark_name,
            beta=beta,
            recommendation=recommendation,
            data_note=data_note,
            risk_free_rate=risk_free_pct / 100,
            portfolio={"decomposition": decomposition, "diversification": diversification, "correlation": correlation,
                       "alignment": alignment} if is_portfolio else None,
            data_layer={"positions": ctx.positions, "quality": ctx.quality, "volume_sources": ctx.volume_sources,
                        "prices_as_of": ctx.prices_as_of},
            trust={"ranges_table": ctx.ranges_table, "grades": ctx.trust["grades"], "model_risk": ctx.trust["model_risk"],
                   "lookback": ctx.lookback, "ghost": ctx.ghost, "block_length": ctx.trust_ranges["block_length"],
                   "headline_model": ctx.headline_model},
            liquidity={"metrics": ctx.liquidity_metrics, "holdings": ctx.liquidity_holdings, "amfi": ctx.liquidity_amfi,
                       "waterfall": ctx.liquidity_waterfall, "participation": ctx.participation,
                       "bangia_k": ctx.bangia_k, "impact_y": ctx.impact_y},
            credit={"metrics": ctx.credit_metrics, "view": ctx.credit_view, "results": ctx.credit_results,
                    "vol_choice": ctx.equity_vol_choice},
        )

        st.download_button(
            label=f"📥 Download Excel Report ({symbol})",
            data=excel_bytes,
            file_name=f"{symbol}_Risk_Analysis_Report_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary"
        )

        st.markdown("---")
        st.markdown("#### 📄 Raw Market Data Export (CSV)")
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"📄 Download Raw Prices & Returns ({symbol}_data.csv)",
            data=csv_bytes,
            file_name=f"{symbol}_market_data.csv",
            mime="text/csv"
        )
