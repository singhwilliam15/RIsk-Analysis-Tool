"""Price & Volatility tab."""

import numpy as np
import plotly.express as px
import streamlit as st
from var_calculator import ewma_volatility


def render(ctx, tab2):
    """Price & Volatility tab."""
    company_name = ctx.company_name
    curr_sym = ctx.curr_sym
    df = ctx.df
    is_portfolio = ctx.is_portfolio
    returns = ctx.returns
    symbol = ctx.symbol

    with tab2:
        st.markdown("### 📈 Price & Volatility Analytics")

        col_p1, col_p2 = st.columns(2)

        with col_p1:
            fig_price = px.line(df, x="Date", y="Close",
                                title="Portfolio Value Index (start = 100)" if is_portfolio else f"{company_name} ({symbol}) Closing Price History",
                                labels={"Close": "Index Level" if is_portfolio else f"Price ({curr_sym})"},
                                template="plotly_dark", color_discrete_sequence=["#4FD1C5"])
            st.plotly_chart(fig_price, width="stretch")

        with col_p2:
            fig_returns = px.line(df, x="Date", y="Returns", title="Daily Percentage Returns",
                                  labels={"Returns": "Daily Return"}, template="plotly_dark", color_discrete_sequence=["#63B3ED"])
            fig_returns.add_hline(y=0, line_dash="dash", line_color="#A0AEC0")
            st.plotly_chart(fig_returns, width="stretch")

        st.markdown("#### 🌊 Annualized Volatility: 30-Day Rolling vs EWMA")
        ewma_sigma, _ = ewma_volatility(returns)
        fig_vol = px.area(df, x="Date", y="Rolling_30d_Vol", title="Volatility clustering: calm and stressed periods",
                          labels={"Rolling_30d_Vol": "Annualized Volatility"}, template="plotly_dark", color_discrete_sequence=["#ED8936"])
        fig_vol.add_scatter(x=df.loc[ewma_sigma.index, "Date"], y=ewma_sigma * np.sqrt(252), mode="lines",
                            name="EWMA (λ = 0.94)", line=dict(color="#68D391", width=1.5))
        fig_vol.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig_vol, width="stretch")
