"""Page configuration, custom CSS and the page header."""

import streamlit as st


def setup_page():
    """Page configuration, custom CSS and the page header."""
    st.set_page_config(
        page_title="Risk Analysis Tool",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom Premium Styling
    st.markdown("""
    <style>
        /* Dark Theme Accent Styling */
        .stApp {
            background-color: #0E1117;
            color: #E0E0E0;
        }
        .metric-card {
            background: linear-gradient(135deg, #1A1F2C 0%, #11151C 100%);
            border: 1px solid #2D3748;
            border-radius: 10px;
            padding: 16px 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            margin-bottom: 12px;
        }
        .metric-label {
            font-size: 0.85rem;
            color: #A0AEC0;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 4px;
        }
        .metric-value {
            font-size: 1.6rem;
            font-weight: 700;
            color: #FFFFFF;
        }
        .metric-sub {
            font-size: 0.8rem;
            color: #68D391;
        }
        .metric-sub-red {
            font-size: 0.8rem;
            color: #FC8181;
        }
        .badge-green {
            background-color: #1C4532;
            color: #68D391;
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.9rem;
        }
        .badge-yellow {
            background-color: #5B4712;
            color: #F6AD55;
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.9rem;
        }
        .badge-red {
            background-color: #63171B;
            color: #FC8181;
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.9rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # App Header
    st.title("⚡ Risk Analysis Tool")
    st.caption("Market, liquidity, credit, concentration and event risk for an Indian or US stock or portfolio, linked "
               "through one stress engine, with a confidence range and trust grade on every headline number. "
               "Market risk is live: eight VaR / ES models, out-of-sample backtests, risk decomposition, stress testing "
               "and Excel export; the other pillars are being added.")
