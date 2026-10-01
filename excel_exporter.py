"""
Excel Exporter Module
Generates a formatted multi-sheet Excel risk report: model comparison, raw data,
out-of-sample backtest and stress tests.
"""

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
import pandas as pd
import numpy as np
import io
from datetime import datetime

# Styles & Palette
HEADER_FILL = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")  # Dark Blue
ACCENT_FILL = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")  # Light Blue
PASS_FILL = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
FAIL_FILL = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid")

BOLD_FONT = Font(name="Calibri", size=11, bold=True)
HEADER_FONT = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
TITLE_FONT = Font(name="Calibri", size=14, bold=True, color="1F4E78")
SUBTITLE_FONT = Font(name="Calibri", size=10, italic=True, color="595959")
REGULAR_FONT = Font(name="Calibri", size=11)


def _plain(value):
    """Convert numpy / pandas scalars to plain Python values openpyxl can write."""
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if hasattr(value, "item"):
        return value.item()
    return value


def _write_table(ws, top_row: int, columns, rows, first_col: int = 2):
    """
    Write a header row and data rows.
    `columns` is a list of (header, number_format) pairs; '@' means text.
    Returns the next free row.
    """
    rows = list(rows)
    for c, (header, _) in enumerate(columns, start=first_col):
        cell = ws.cell(row=top_row, column=c, value=header)
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(horizontal="center", wrap_text=True)
    for r, row in enumerate(rows, start=top_row + 1):
        for c, ((_, fmt), value) in enumerate(zip(columns, row), start=first_col):
            cell = ws.cell(row=r, column=c, value=_plain(value))
            cell.font = REGULAR_FONT
            if fmt != "@":
                cell.number_format = fmt
    return top_row + len(rows) + 2


def generate_excel_var_report(symbol: str, company_name: str, currency: str, investment: float,
                              confidence_level: float, holding_period: int, df_data: pd.DataFrame,
                              var_by_level: dict, backtest_table: pd.DataFrame, backtest_window: int,
                              stress_df: pd.DataFrame, worst_df: pd.DataFrame,
                              benchmark_name: str, beta: float, portfolio: dict = None) -> bytes:
    """
    Generate the Excel risk report and return it as .xlsx bytes.
    `var_by_level` maps model name -> {confidence level -> VaR result dict}.
    `portfolio` (optional) holds the component VaR table, diversification summary and correlation matrix.
    """
    money = f'"{currency}" #,##0'
    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    # -------------------------------------------------------------
    # SHEET 1: DASHBOARD
    # -------------------------------------------------------------
    ws_dash = wb.create_sheet(title="Dashboard")
    ws_dash["B2"] = f"⚡ VALUE AT RISK — RISK MANAGEMENT DASHBOARD ({symbol})"
    ws_dash["B2"].font = TITLE_FONT
    ws_dash["B3"] = f"Company: {company_name} | Currency: {currency} | Generated: {datetime.now():%Y-%m-%d %H:%M}"
    ws_dash["B3"].font = SUBTITLE_FONT

    ws_dash["B5"] = "POSITION SETTINGS"
    ws_dash["B5"].font = BOLD_FONT
    ws_dash["B5"].fill = ACCENT_FILL
    settings_data = [
        ("Position Value", investment, money),
        ("Selected Confidence Level", confidence_level, "0.0%"),
        ("Holding Period (Days)", holding_period, "0"),
        (f"Beta vs {benchmark_name}", beta if np.isfinite(beta) else "n/a", "0.00"),
    ]
    for i, (label, val, fmt) in enumerate(settings_data, start=6):
        ws_dash[f"B{i}"] = label
        ws_dash[f"C{i}"] = _plain(val)
        ws_dash[f"C{i}"].number_format = fmt
        ws_dash[f"C{i}"].font = BOLD_FONT

    ws_dash["B11"] = f"VAR AND EXPECTED SHORTFALL BY MODEL ({holding_period}-DAY HORIZON)"
    ws_dash["B11"].font = BOLD_FONT
    cl_pct = f"{int(confidence_level * 100)}%"
    _write_table(
        ws_dash, 12,
        [("Model", "@"), ("90% VaR", money), ("95% VaR", money), ("99% VaR", money), (f"{cl_pct} Expected Shortfall", money)],
        [
            (model, levels[0.90]["var_scaled_amount"], levels[0.95]["var_scaled_amount"],
             levels[0.99]["var_scaled_amount"], levels[confidence_level]["cvar_scaled_amount"])
            for model, levels in var_by_level.items()
        ],
    )

    # -------------------------------------------------------------
    # PORTFOLIO SHEET (portfolio mode only)
    # -------------------------------------------------------------
    if portfolio is not None:
        ws_port = wb.create_sheet(title="Portfolio Risk")
        ws_port["B2"] = f"🧩 PORTFOLIO RISK DECOMPOSITION ({cl_pct}, {holding_period}-DAY)"
        ws_port["B2"].font = TITLE_FONT
        ws_port["B3"] = "Component VaR is the Euler allocation of parametric VaR and sums to the portfolio total."
        ws_port["B3"].font = SUBTITLE_FONT

        div = portfolio["diversification"]
        summary_rows = [
            ("Portfolio Historical VaR", div["portfolio_var"], money),
            ("Sum of Standalone VaRs", div["undiversified_var"], money),
            ("Diversification Benefit", div["diversification_benefit"], money),
            ("Diversification Ratio", div["diversification_ratio"], "0.0%"),
            ("Average Pairwise Correlation", div["average_correlation"], "0.00"),
            ("Common Trading Days", div["common_days"], "0"),
        ]
        for i, (label, val, fmt) in enumerate(summary_rows, start=5):
            ws_port[f"B{i}"] = label
            ws_port[f"C{i}"] = _plain(val)
            ws_port[f"C{i}"].number_format = fmt
            ws_port[f"C{i}"].font = BOLD_FONT

        comp = portfolio["components"]
        comp_cols = [
            ("Ticker", "@"), ("Company", "@"), ("Weight", "0.0%"), ("Annualized Volatility", "0.0%"),
            ("Standalone VaR", money), ("Marginal VaR", "0.00%"), ("Component VaR", money),
            ("Contribution %", "0.0%"), ("Risk / Weight", "0.00"),
        ]
        next_row = _write_table(ws_port, 12, comp_cols, comp[[name for name, _ in comp_cols]].itertuples(index=False))
        total_row = next_row - 1
        ws_port.cell(row=total_row, column=2, value="TOTAL").font = BOLD_FONT
        for name in ("Weight", "Component VaR", "Contribution %"):
            col = 2 + [n for n, _ in comp_cols].index(name)
            letter = get_column_letter(col)
            cell = ws_port.cell(row=total_row, column=col, value=f"=SUM({letter}13:{letter}{12 + len(comp)})")
            cell.number_format = dict(comp_cols)[name]
            cell.font = BOLD_FONT

        ws_port.cell(row=total_row + 2, column=2, value="CORRELATION OF DAILY RETURNS").font = BOLD_FONT
        corr = portfolio["correlation"]
        _write_table(
            ws_port, total_row + 3,
            [("", "@")] + [(t, "0.00") for t in corr.columns],
            [(t, *row) for t, row in zip(corr.index, corr.to_numpy())],
        )

    # -------------------------------------------------------------
    # SHEET 2: BACKTESTING
    # -------------------------------------------------------------
    ws_back = wb.create_sheet(title="Backtesting")
    ws_back["B2"] = f"🔬 OUT-OF-SAMPLE BACKTEST — {cl_pct} VAR, ROLLING {backtest_window}-DAY WINDOW"
    ws_back["B2"].font = TITLE_FONT
    ws_back["B3"] = ("Kupiec: correct breach count. Christoffersen: breaches independent. "
                     "Conditional coverage: both. PASS requires every p-value >= 0.05.")
    ws_back["B3"].font = SUBTITLE_FONT
    backtest_cols = [
        ("Method", "@"), ("Test Days", "0"), ("Expected Breaches", "0.0"), ("Actual Breaches", "0"),
        ("Breach Rate", "0.00%"), ("Kupiec p-value", "0.000"), ("Independence p-value", "0.000"),
        ("Conditional Coverage p-value", "0.000"), ("Back-to-Back Breaches", "0"),
        ("Traffic Light", "@"), ("Verdict", "@"), ("Avg VaR", "0.00%"),
    ]
    _write_table(ws_back, 5, backtest_cols, backtest_table[[name for name, _ in backtest_cols]].itertuples(index=False))
    verdict_col = 2 + [name for name, _ in backtest_cols].index("Verdict")
    for r in range(6, 6 + len(backtest_table)):
        cell = ws_back.cell(row=r, column=verdict_col)
        cell.fill = PASS_FILL if cell.value == "PASS" else FAIL_FILL
        cell.font = BOLD_FONT

    # -------------------------------------------------------------
    # SHEET 3: STRESS TESTING
    # -------------------------------------------------------------
    ws_stress = wb.create_sheet(title="Stress Testing")
    ws_stress["B2"] = "⚡ STRESS TESTING — HISTORICAL CRISIS SCENARIOS"
    ws_stress["B2"].font = TITLE_FONT
    beta_text = f"{beta:.2f}" if np.isfinite(beta) else "1.00 (assumed)"
    ws_stress["B3"] = f"Market shocks scaled by beta to the {benchmark_name} (β = {beta_text}), capped at a 100% loss."
    ws_stress["B3"].font = SUBTITLE_FONT
    next_row = _write_table(
        ws_stress, 5,
        [("Scenario", "@"), ("Market Shock", "0.0%"), ("Beta", "0.00"), ("Stock Shock", "0.0%"),
         ("Portfolio Impact", money), ("Post-Shock Value", money), ("Recovery (Days)", "0"),
         ("Probability", "@"), ("Risk Level", "@")],
        stress_df[["Scenario", "Shock", "Beta", "Stock_Shock", "Portfolio_Impact", "Post_Shock_Value",
                   "Recovery_Days", "Probability", "Risk_Level"]].itertuples(index=False),
    )
    ws_stress.cell(row=next_row, column=2, value="WORST ACTUAL LOSSES IN SAMPLE").font = BOLD_FONT
    window_end = df_data.loc[worst_df["Window End"], "Date"].dt.strftime("%Y-%m-%d").to_numpy() if len(worst_df) else []
    _write_table(
        ws_stress, next_row + 1,
        [("Horizon", "@"), ("Worst Return", "0.00%"), ("Loss", money), ("Window End", "@")],
        zip(worst_df["Horizon"], worst_df["Worst Return"], worst_df["Loss"], window_end),
    )

    # -------------------------------------------------------------
    # SHEET 4: RAW DATA
    # -------------------------------------------------------------
    ws_raw = wb.create_sheet(title="Raw Data")
    ws_raw["B2"] = f"📈 HISTORICAL MARKET PRICE DATA ({symbol})"
    ws_raw["B2"].font = TITLE_FONT
    raw = df_data[["Date", "Close", "Returns", "Log_Returns", "Rolling_30d_Vol"]].copy()
    raw["Date"] = raw["Date"].dt.strftime("%Y-%m-%d")
    raw = raw.astype(object).where(raw.notna(), None)
    _write_table(
        ws_raw, 4,
        [("Date", "@"), ("Close Price", "#,##0.00"), ("Daily Return", "0.00%"), ("Log Return", "0.00%"),
         ("Rolling 30d Volatility", "0.00%")],
        raw.itertuples(index=False),
    )

    # Fit column widths to content, ignoring the long title and note rows
    for ws in wb.worksheets:
        for col in ws.iter_cols(min_row=4):
            max_len = max((len(str(cell.value)) for cell in col if cell.value is not None), default=0)
            ws.column_dimensions[get_column_letter(col[0].column)].width = min(max(max_len + 3, 12), 45)

    output = io.BytesIO()
    wb.save(output)
    return output.getvalue()
