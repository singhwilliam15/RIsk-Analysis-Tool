"""
Parsers for official NSE files and API responses, turning them into the disclosure templates (disclosures.py).
Pure functions on text and JSON, so they are tested offline on saved samples; the downloading lives in
scripts/fetch_disclosures.py.

- Shareholding pattern XBRL (SEBI LODR Reg. 31): promoter holding and promoter shares pledged.
- ASM / GSM surveillance lists (JSON).
- Credit-rating disclosures (SEBI LODR Reg. 30, NSE "Credit Rating" filings, JSON).
"""

import re

import numpy as np
import pandas as pd

from events import RATING_SCALE, SHORT_TERM_RATING, rating_rank

PROMOTER_CONTEXT = "ShareholdingOfPromoterAndPromoterGroup_ContextI"
TOTAL_CONTEXT = "ShareholdingPattern_ContextI"
ASM_URL = "https://www.nseindia.com/api/reportASM"
GSM_URL = "https://www.nseindia.com/api/reportGSM"


# ---------------------------------------------------------------
# Shareholding pattern XBRL
# ---------------------------------------------------------------

def xbrl_fact(text: str, tag: str, context: str):
    """The first value of `tag` in `context` (any namespace prefix), or None."""
    m = re.search(rf"<[A-Za-z\-]+:{tag}\b[^>]*contextRef=\"{re.escape(context)}\"[^>]*>([^<]*)<", text)
    return m.group(1).strip() if m else None


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def shareholding_pledge(text: str) -> dict:
    """
    Promoter holding and pledges from a shareholding-pattern XBRL filing, in percent (0–100):
    promoter holding of total shares, shares pledged as % of the promoter holding and of total shares, and the
    number pledged. Shares encumbered by non-disposal undertakings are not pledges and are left out. None when
    the filing does not carry the promoter-holding fact (an older taxonomy).
    """
    # Filings state percentages either as fractions (total shareholding "1") or in percent ("100.00", older
    # taxonomy versions): the filing's own total-shareholding fact gives the scale
    total_pct = _number(xbrl_fact(text, "ShareholdingAsAPercentageOfTotalNumberOfShares", TOTAL_CONTEXT))
    to_fraction = 0.01 if np.isfinite(total_pct) and total_pct > 1.5 else 1.0
    promoter = _number(xbrl_fact(text, "ShareholdingAsAPercentageOfTotalNumberOfShares", PROMOTER_CONTEXT)) * to_fraction
    total_shares = _number(xbrl_fact(text, "NumberOfShares", TOTAL_CONTEXT))
    if not np.isfinite(promoter):
        if not np.isfinite(total_shares):
            return None
        # A valid filing with no promoter group at all (e.g. HDFC Bank, Coforge): nothing held, nothing pledged
        report = re.search(r"<[A-Za-z\-]+:DateOfReport\b[^>]*>([^<]*)<", text)
        return {"promoter_holding_pct": 0.0, "pledged_pct_of_promoter": 0.0, "pledged_pct_of_total": 0.0,
                "pledged_shares": 0.0, "total_shares": total_shares,
                "report_date": pd.Timestamp(report.group(1)) if report else pd.NaT}
    pledged = _number(xbrl_fact(text, "NumberOfSharesEncumberedUnderPledged", PROMOTER_CONTEXT))
    of_promoter = _number(xbrl_fact(text, "EncumberedShareUnderPledgedAsPercentageOfTotalNumberOfShares", PROMOTER_CONTEXT)) * to_fraction
    of_total = _number(xbrl_fact(text, "EncumberedShareUnderPledgedAsPercentageOfTotalNumberOfShares", TOTAL_CONTEXT)) * to_fraction
    any_pledge = re.search(r"WhetherAnySharesHeldByPromotersAreEncumberedUnderPledged\b[^>]*>\s*true", text)
    if not np.isfinite(pledged) and not any_pledge:
        pledged, of_promoter, of_total = 0.0, 0.0, 0.0  # the filing says no promoter shares are pledged
    report = re.search(r"<[A-Za-z\-]+:DateOfReport\b[^>]*>([^<]*)<", text)
    return {"promoter_holding_pct": promoter * 100, "pledged_pct_of_promoter": of_promoter * 100,
            "pledged_pct_of_total": of_total * 100, "pledged_shares": pledged, "total_shares": total_shares,
            "report_date": pd.Timestamp(report.group(1)) if report else pd.NaT}


def pledge_row(symbol: str, filing: dict, parsed: dict) -> dict:
    """One pledges-template row from a shareholding master record (date, broadcastDate, xbrl) and its parsed XBRL."""
    return {"symbol": symbol, "quarter_end": pd.to_datetime(filing["date"], format="%d-%b-%Y").strftime("%Y-%m-%d"),
            "promoter_holding_pct": round(parsed["promoter_holding_pct"], 4),
            "pledged_pct_of_promoter": round(parsed["pledged_pct_of_promoter"], 4),
            "pledged_pct_of_total": round(parsed["pledged_pct_of_total"], 4),
            "pledged_shares": parsed["pledged_shares"],
            "disclosure_date": pd.to_datetime(filing["broadcastDate"][:11], format="%d-%b-%Y").strftime("%Y-%m-%d"),
            "source": filing["xbrl"]}


# ---------------------------------------------------------------
# Surveillance
# ---------------------------------------------------------------

def _gsm_stage(desc: str) -> str:
    """'ASM IBC Stage I and GSM Stage 0' → 'Stage 0'; 'Graded Surveillance Measure - Stage VI' → 'Stage VI'."""
    m = re.search(r"(?:GSM|Graded Surveillance Measure)\s*-?\s*Stage\s*([0-9]+|[IVX]+)\b", str(desc), re.I)
    return f"Stage {m.group(1).upper()}" if m else str(desc)


def surveillance_rows(asm: dict, gsm: list, list_date) -> pd.DataFrame:
    """
    Surveillance-template rows from NSE's ASM list (long- and short-term) and GSM list. The lists show today's
    stage only, so date_in is the list date and date_out is blank.
    """
    day = pd.Timestamp(list_date).strftime("%Y-%m-%d")
    rows = []
    for key, measure in (("longterm", "ASM-LT"), ("shortterm", "ASM-ST")):
        for r in (asm.get(key) or {}).get("data", []):
            rows.append({"symbol": r["symbol"], "measure": measure, "stage": r.get("asmSurvIndicator") or r.get("survDesc"),
                         "date_in": day, "date_out": "", "source": ASM_URL})
    for r in gsm or []:
        rows.append({"symbol": r["symbol"], "measure": "GSM", "stage": _gsm_stage(r.get("survDesc")),
                     "date_in": day, "date_out": "", "source": GSM_URL})
    return pd.DataFrame(rows, columns=["symbol", "measure", "stage", "date_in", "date_out", "source"])


# ---------------------------------------------------------------
# Credit-rating disclosures
# ---------------------------------------------------------------

AGENCIES = (("CRISIL", "CRISIL"), ("ICRA", "ICRA"), ("CARE", "CARE"), ("INDIA RATINGS", "India Ratings"),
            ("ACUIT", "Acuité"), ("BRICKWORK", "Brickwork"), ("INFOMERICS", "Infomerics"))


def agency_name(text: str) -> str:
    upper = str(text).upper()
    return next((short for key, short in AGENCIES if key in upper), str(text).strip())


def is_short_term(rating: str) -> bool:
    """Short-term scales (A1+ … A4, with agency prefixes) are not on the long-term AAA…D scale."""
    return bool(SHORT_TERM_RATING.search(str(rating).upper()))


def rating_action(record: dict):
    """
    The template action for an NSE credit-rating record: New → assigned, Reaffirm → reaffirmed, an explicit
    upgrade/downgrade/withdrawal/watch, and otherwise a change against the earlier rating in the same filing.
    None when the action cannot be read (the row is then skipped, not guessed).
    """
    action = str(record.get("RatingAction") or "").lower()
    other = str(record.get("SpecifyOthRatingActn") or "").lower()
    rating = str(record.get("CreditRating") or "").lower()
    text = f"{action} {other}"
    if "watch" in text or "watch" in rating:
        return "placed on watch"
    for word, mapped in (("upgrad", "upgraded"), ("downgrad", "downgraded"), ("withdr", "withdrawn"), ("suspend", "suspended")):
        if word in text:
            return mapped
    if action == "new" or "assign" in text:
        return "assigned"
    if action.startswith("reaffirm"):
        return "reaffirmed"
    now, before = rating_rank(record.get("CreditRating")), rating_rank(record.get("CreditRatingEarlier"))
    if now >= 0 and before >= 0 and not is_short_term(record.get("CreditRating")):
        return "upgraded" if now < before else "downgraded" if now > before else "reaffirmed"
    return None


def rating_rows(records: list, issuers: dict) -> tuple:
    """
    Ratings-template rows for the issuers in `issuers` ({ISIN issuer prefix such as 'INE351F': symbol}). Every
    instrument of a covered issuer is kept; the instrument is its ISIN. Returns (frame, skipped count).
    """
    rows, skipped = [], 0
    seen = set()
    for r in records:
        isin = str(r.get("ISIN") or "")
        symbol = issuers.get(isin[:7])
        if not symbol:
            continue
        action = rating_action(r)
        if action is None or not r.get("DateofCR") or not r.get("CreditRating"):
            skipped += 1
            continue
        key = (symbol, r.get("NameOfCRAgency"), isin, r.get("CreditRating"), r.get("DateofCR"))
        if key in seen:
            continue
        seen.add(key)
        rows.append({"symbol": symbol, "agency": agency_name(r.get("NameOfCRAgency")), "instrument": f"ISIN {isin}",
                     "rating": str(r["CreditRating"]).strip(), "outlook": (r.get("Outlook") or "").strip(),
                     "action": action, "action_date": pd.to_datetime(r["DateofCR"], format="%d-%m-%Y").strftime("%Y-%m-%d"),
                     "source": r.get("XbrlFileName") or ""})
    columns = ["symbol", "agency", "instrument", "rating", "outlook", "action", "action_date", "source"]
    return pd.DataFrame(rows, columns=columns), skipped


def latest_long_term_rating(ratings: pd.DataFrame, symbol: str, as_of=None):
    """The most recent long-term rating of `symbol` public by `as_of` (short-term scales and withdrawals skipped)."""
    if ratings is None or ratings.empty:
        return None
    rows = ratings[ratings["symbol"] == symbol]
    if as_of is not None:
        rows = rows[pd.to_datetime(rows["action_date"]) <= pd.Timestamp(as_of)]
    rows = rows[~rows["rating"].map(is_short_term) & (rows["rating"].map(rating_rank) >= 0)
                & (rows["action"] != "withdrawn")]
    if rows.empty:
        return None
    last = rows.sort_values("action_date").iloc[-1]
    return {"rating": last["rating"], "grade": RATING_SCALE[rating_rank(last["rating"])], "agency": last["agency"],
            "date": pd.Timestamp(last["action_date"])}
