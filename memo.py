"""
One-page CRO memo, as Markdown and as PDF (reportlab + matplotlib). The text is assembled from rules and the
computed figures; no language model is involved.

Sections: Bottom line · Pillar summary · Integrated stress · What the numbers miss · Limits · Actions ·
Data sources and caveats.
"""

import io
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle  # noqa: E402

MISSES = (
    "Estimation ranges cover sampling error, not a change of regime.",
    "Liquidity ignores block deals, free float and redemptions; impact uses an assumed constant.",
    "Merton PD is risk-neutral; off-balance-sheet, group and promoter debt are not captured.",
    "Event tiers see only filed disclosures; jump sizes are assumptions until the case studies test them.",
    "Linked stress uses past crises; a new kind of crisis can combine the pillars differently.",
)
STATUS_COLORS = {"green": colors.HexColor("#C6EFCE"), "amber": colors.HexColor("#FFEB9C"), "red": colors.HexColor("#FFC7CE")}


def bottom_line(content: dict) -> str:
    """One sentence from rules: limit status first, then the worst linked stress and the interaction effect."""
    reds = [r for r in content["limits"] if r["Status"] == "red"]
    ambers = [r for r in content["limits"] if r["Status"] == "amber"]
    stress = content["stress"]
    if reds:
        lead = f"{len(reds)} limit(s) breached ({', '.join(r['Limit'] for r in reds)})"
    elif ambers:
        lead = f"Within limits, but {len(ambers)} close to its limit ({', '.join(r['Limit'] for r in ambers)})"
    else:
        lead = "All limits are green"
    if stress:
        lead += f". In the worst linked scenario ({stress['name']}) the portfolio would lose {stress['linked_pct']:.1%}"
        lead += (", with no cross-pillar interaction" if abs(stress["interaction_pct"]) < 5e-5 else
                 f", of which the cross-pillar interaction is {stress['interaction_pct']:+.2%} of value")
    return lead + "."


def markdown(content: dict) -> str:
    lines = [f"# CRO memo: {content['name']}", "",
             f"*{content['as_of']:%d %b %Y} · {content['confidence']} · value {content['value']} · Risk Analysis Tool*", "",
             "## Bottom line", "", bottom_line(content), "", "## Pillar summary", "",
             "| Pillar | Headline | Range | Grade | Status |", "| --- | --- | --- | --- | --- |"]
    lines += [f"| {r['Pillar']} | {r['Headline']} | {r['Range']} | {r['Grade']} | {r['Status']} |" for r in content["pillars"]]
    lines += ["", "## Integrated stress", ""]
    s = content["stress"]
    if s:
        lines += [f"Worst scenario **{s['name']}**: linked total **{s['linked']}** against {s['plain']} for the market move "
                  f"alone. Shapley split: market {s['market']}, liquidity {s['liquidity']}, credit {s['credit']}, events "
                  f"{s['events']}; cross-pillar interaction {s['interaction']}."]
    if content.get("reverse"):
        lines += ["", content["reverse"]]
    lines += ["", "## What the numbers miss", ""] + [f"- {m}" for m in MISSES]
    lines += ["", "## Limits", "", "| Limit | Value | Limit | Utilisation | Status |", "| --- | --- | --- | --- | --- |"]
    lines += [f"| {r['Limit']} | {r['Shown']} | {r['Limit Shown']} | {r['Utilisation Shown']} | {r['Status']} |"
              for r in content["limits"]]
    lines += ["", "## Top risks", ""] + [f"{i}. {r}" for i, r in enumerate(content["risks"], 1)]
    lines += ["", "## Actions", ""] + [f"{i}. {a}" for i, a in enumerate(content["actions"], 1)]
    lines += ["", "## Data sources and caveats", ""] + [f"- {s}" for s in content["sources"]]
    lines += ["", f"*Generated {date.today():%d %b %Y} from rules; no language model. Methodology: docs/methodology.md.*"]
    return "\n".join(lines) + "\n"


def _chart(content: dict) -> bytes:
    s = content["stress"]
    fig, ax = plt.subplots(figsize=(6.2, 1.9), dpi=150)
    parts = [("Market", s["market_value"], "#3182CE"), ("Liquidity", s["liquidity_value"], "#DD6B20"),
             ("Credit", s["credit_value"], "#805AD5"), ("Events", s["events_value"], "#E53E3E")]
    left = 0.0
    for label, value, color in parts:
        ax.barh("Linked (Shapley split)", value, left=left, color=color, label=label)
        left += value
    ax.barh("Market move alone", s["plain_value"], color="#A0AEC0")
    ax.set_xlabel(f"Loss ({content['currency']})")
    ax.legend(ncol=4, fontsize=7, loc="lower right", frameon=False)
    ax.tick_params(labelsize=7)
    ax.set_title(f"Worst linked scenario: {s['name']}", fontsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def pdf(content: dict) -> bytes:
    """The memo on one A4 page."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=14 * mm, rightMargin=14 * mm, topMargin=12 * mm, bottomMargin=10 * mm)
    base = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=base["BodyText"], fontSize=7.5, leading=9.2)
    head = ParagraphStyle("head", parent=base["Heading3"], fontSize=9, spaceBefore=4, spaceAfter=2)
    title = ParagraphStyle("title", parent=base["Title"], fontSize=13, spaceAfter=2)
    small = ParagraphStyle("small", parent=body, fontSize=6.5, leading=8, textColor=colors.HexColor("#555555"))

    def table(rows, widths, status_col=None):
        t = Table([[Paragraph(str(c), small) for c in row] for row in rows], colWidths=widths, repeatRows=1)
        style = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E78")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                 ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#BBBBBB")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                 ("TOPPADDING", (0, 0), (-1, -1), 1), ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]
        if status_col is not None:
            for i, row in enumerate(rows[1:], start=1):
                color = STATUS_COLORS.get(str(row[status_col]).lower())
                if color:
                    style.append(("BACKGROUND", (status_col, i), (status_col, i), color))
        t.setStyle(TableStyle(style))
        return t

    story = [Paragraph(f"CRO memo: {content['name']}", title),
             Paragraph(f"{content['as_of']:%d %b %Y} · {content['confidence']} · value {content['value']} · Risk Analysis Tool", small),
             Paragraph("Bottom line", head), Paragraph(bottom_line(content), body), Paragraph("Pillar summary", head)]
    rows = [["Pillar", "Headline", "Range", "Grade", "Status"]] + [
        [r["Pillar"], r["Headline"], r["Range"], r["Grade"], r["Status"]] for r in content["pillars"]]
    story.append(table(rows, [28 * mm, 52 * mm, 58 * mm, 14 * mm, 28 * mm], status_col=4))
    story.append(Paragraph("Integrated stress", head))
    if content["stress"]:
        s = content["stress"]
        story.append(Paragraph(f"Linked total {s['linked']} vs {s['plain']} for the market move alone "
                               f"(cross-pillar interaction {s['interaction']}).", body))
        story.append(Image(io.BytesIO(_chart(content)), width=165 * mm, height=50 * mm))
    if content.get("reverse"):
        story.append(Paragraph(content["reverse"], body))
    story.append(Paragraph("What the numbers miss", head))
    story += [Paragraph(f"• {m}", body) for m in MISSES]
    story.append(Paragraph("Limits", head))
    rows = [["Limit", "Value", "Limit", "Use", "Status"]] + [
        [r["Limit"], r["Shown"], r["Limit Shown"], r["Utilisation Shown"], r["Status"]] for r in content["limits"]]
    story.append(table(rows, [76 * mm, 30 * mm, 30 * mm, 20 * mm, 24 * mm], status_col=4))
    story.append(Paragraph("Top risks", head))
    story += [Paragraph(f"{i}. {r}", body) for i, r in enumerate(content["risks"], 1)]
    story.append(Paragraph("Actions", head))
    story += [Paragraph(f"{i}. {a}", body) for i, a in enumerate(content["actions"], 1)]
    story.append(Paragraph("Data sources and caveats", head))
    story += [Paragraph(f"• {s}", small) for s in content["sources"]]
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(f"Generated {date.today():%d %b %Y} from rules; no language model. Methodology: docs/methodology.md.", small))
    doc.build(story)
    return buf.getvalue()
