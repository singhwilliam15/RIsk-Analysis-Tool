"""
Excel Exporter Module
Generates a complete multi-sheet Excel report matching VaR_Risk_Management_Tool.xlsx
"""

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
import pandas as pd
import numpy as np
import io

def generate_excel_var_report(symbol: str, company_name: str, currency: str, investment: float, 
                             confidence_level: float, holding_period: int, df_data: pd.DataFrame, 
                             stats: dict, var_hist: dict, var_param: dict, var_mc: dict, 
                             backtest: dict, stress_df: pd.DataFrame) -> bytes:
    """
    Generate complete Excel workbook mirroring the original VaR Risk Management Tool template.
    Returns bytes of the generated .xlsx file.
    """
    wb = openpyxl.Workbook()
    # Remove default sheet
    wb.remove(wb.active)
    
    # Styles & Palette
    header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid") # Dark Blue
    accent_fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid") # Light Blue
    highlight_fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid") # Soft Gold
    green_fill = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
    
    bold_font = Font(name="Calibri", size=11, bold=True)
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    title_font = Font(name="Calibri", size=14, bold=True, color="1F4E78")
    subtitle_font = Font(name="Calibri", size=10, italic=True, color="595959")
    regular_font = Font(name="Calibri", size=11)
    
    thin_border = Border(
        left=Side(style='thin', color='D9D9D9'),
        right=Side(style='thin', color='D9D9D9'),
        top=Side(style='thin', color='D9D9D9'),
        bottom=Side(style='thin', color='D9D9D9')
    )

    # -------------------------------------------------------------
    # SHEET 1: DASHBOARD
    # -------------------------------------------------------------
    ws_dash = wb.create_sheet(title="Dashboard")
    ws_dash.views.sheetView[0].showGridLines = True
    
    ws_dash["B2"] = f"⚡ VALUE AT RISK — RISK MANAGEMENT DASHBOARD ({symbol})"
    ws_dash["B2"].font = title_font
    ws_dash["B3"] = f"Company: {company_name} | Currency: {currency} | Date Generated: Auto-Automated Tool"
    ws_dash["B3"].font = subtitle_font
    
    # Portfolio Settings
    ws_dash["B5"] = "PORTFOLIO SETTINGS"
    ws_dash["B5"].font = bold_font
    ws_dash["B5"].fill = accent_fill
    
    settings_data = [
        ("Initial Investment", investment, f'"{currency}" #,##0'),
        ("Confidence Level", confidence_level, "0.0%"),
        ("Holding Period (Days)", holding_period, "0")
    ]
    
    for i, (label, val, fmt) in enumerate(settings_data, start=6):
        ws_dash[f"B{i}"] = label
        ws_dash[f"C{i}"] = val
        ws_dash[f"C{i}"].number_format = fmt
        ws_dash[f"B{i}"].font = regular_font
        ws_dash[f"C{i}"].font = bold_font

    # Risk Summary Comparison Table
    ws_dash["B10"] = "SUMMARY RISK COMPARISON (1-DAY VAR)"
    ws_dash["B10"].font = bold_font
    
    headers_comp = ["Methodology", "90% VaR", "95% VaR", "99% VaR", "95% Expected Shortfall (CVaR)"]
    for col_idx, h in enumerate(headers_comp, start=2):
        cell = ws_dash.cell(row=11, column=col_idx, value=h)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")
        
    comp_rows = [
        ("Historical VaR (Non-Parametric)", var_hist["var_daily_amount"], var_hist["var_daily_amount"], var_hist["var_daily_amount"], var_hist["cvar_daily_amount"]),
        ("Parametric VaR (Variance-Covariance)", var_param["var_daily_amount"], var_param["var_daily_amount"], var_param["var_daily_amount"], var_param["cvar_daily_amount"]),
        ("Monte Carlo VaR (Simulated Paths)", var_mc["var_daily_amount"], var_mc["var_daily_amount"], var_mc["var_daily_amount"], var_mc["cvar_daily_amount"])
    ]
    
    for r_idx, row_tuple in enumerate(comp_rows, start=12):
        ws_dash.cell(row=r_idx, column=2, value=row_tuple[0]).font = bold_font
        for c_idx in range(3, 7):
            cell = ws_dash.cell(row=r_idx, column=c_idx, value=row_tuple[c_idx-2])
            cell.number_format = f'"{currency}" #,##0'
            cell.font = regular_font
            cell.alignment = Alignment(horizontal="right")

    # -------------------------------------------------------------
    # SHEET 2: RAW DATA
    # -------------------------------------------------------------
    ws_raw = wb.create_sheet(title="Raw Data")
    ws_raw.views.sheetView[0].showGridLines = True
    
    ws_raw["B2"] = f"📈 HISTORICAL MARKET PRICE DATA ({symbol})"
    ws_raw["B2"].font = title_font
    
    headers_raw = ["Date", "Close Price", "Daily Return", "Log Return", "Rolling 30d Volatility"]
    for col_idx, h in enumerate(headers_raw, start=2):
        cell = ws_raw.cell(row=4, column=col_idx, value=h)
        cell.font = header_font
        cell.fill = header_fill
        
    for r_i, row in df_data.iterrows():
        r = r_i + 5
        ws_raw.cell(row=r, column=2, value=row['Date'].strftime('%Y-%m-%d')).alignment = Alignment(horizontal="center")
        ws_raw.cell(row=r, column=3, value=row['Close']).number_format = '#,##0.00'
        
        ret_cell = ws_raw.cell(row=r, column=4, value=row['Returns'] if pd.notnull(row['Returns']) else "")
        ret_cell.number_format = '0.00%'
        
        log_cell = ws_raw.cell(row=r, column=5, value=row['Log_Returns'] if pd.notnull(row['Log_Returns']) else "")
        log_cell.number_format = '0.00%'
        
        vol_cell = ws_raw.cell(row=r, column=6, value=row['Rolling_30d_Vol'] if pd.notnull(row['Rolling_30d_Vol']) else "")
        vol_cell.number_format = '0.00%'

    # -------------------------------------------------------------
    # SHEET 3: BACKTESTING
    # -------------------------------------------------------------
    ws_back = wb.create_sheet(title="Backtesting")
    ws_back.views.sheetView[0].showGridLines = True
    
    ws_back["B2"] = "🔬 VAR BACKTESTING — KUPIEC & BASEL TRAFFIC LIGHT TEST"
    ws_back["B2"].font = title_font
    
    back_items = [
        ("Total Historical Observations (T)", backtest["total_observations"], "0"),
        ("Model Confidence Level", backtest["confidence_level"], "0.0%"),
        ("Expected VaR Exceptions", backtest["expected_failures"], "0.0"),
        ("Actual VaR Exceptions", backtest["actual_breaches"], "0"),
        ("Breach Rate (%)", backtest["breach_rate"], "0.00%"),
        ("Kupiec Likelihood Ratio (LR) Stat", backtest["lr_stat"], "0.0000"),
        ("Chi-Square Critical Value (95%)", backtest["chi_sq_critical"], "0.0000"),
        ("Kupiec Test Result", backtest["test_result"], "@"),
        ("Basel II Traffic Light Status", backtest["traffic_light"], "@"),
        ("Status Description", backtest["status_desc"], "@")
    ]
    
    ws_back["B4"] = "METRIC"
    ws_back["C4"] = "VALUE"
    ws_back["B4"].font = header_font
    ws_back["C4"].font = header_font
    ws_back["B4"].fill = header_fill
    ws_back["C4"].fill = header_fill
    
    for i, (label, val, fmt) in enumerate(back_items, start=5):
        ws_back[f"B{i}"] = label
        cell_v = ws_back[f"C{i}"]
        cell_v.value = val
        cell_v.font = bold_font
        if fmt != "@":
            cell_v.number_format = fmt

    # Auto-fit Column Widths
    for ws in wb.worksheets:
        for col in ws.columns:
            max_len = max(len(str(cell.value or '')) for cell in col)
            col_letter = get_column_letter(col[0].column)
            ws.column_dimensions[col_letter].width = max(max_len + 3, 12)

    output = io.BytesIO()
    wb.save(output)
    return output.getvalue()
