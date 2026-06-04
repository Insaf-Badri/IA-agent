import sys
import json
from pathlib import Path
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER

OUTPUT_DIR = Path("data/output")
REPORT_DIR = Path("data/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── Colors ────────────────────────────────────────────────────────────────────
NAVY   = colors.HexColor("#1B3A6B")
RED    = colors.HexColor("#CF0A2C")
GREEN  = colors.HexColor("#155724")
ORANGE = colors.HexColor("#856404")
LIGHT  = colors.HexColor("#D6E4F0")
GRAY   = colors.HexColor("#555555")
WHITE  = colors.white

# ── Styles ────────────────────────────────────────────────────────────────────
def styles():
    return {
        "title":         ParagraphStyle("title",         fontSize=22, textColor=WHITE, fontName="Helvetica-Bold", alignment=TA_CENTER, spaceAfter=4),
        "subtitle":      ParagraphStyle("subtitle",      fontSize=10, textColor=WHITE, fontName="Helvetica",      alignment=TA_CENTER),
        "section":       ParagraphStyle("section",       fontSize=11, textColor=NAVY,  fontName="Helvetica-Bold", spaceBefore=12, spaceAfter=5),
        "body":          ParagraphStyle("body",          fontSize=9.5,textColor=GRAY,  fontName="Helvetica"),
        "decision":      ParagraphStyle("decision",      fontSize=16, fontName="Helvetica-Bold", alignment=TA_CENTER, spaceAfter=4),
        "justification": ParagraphStyle("justification", fontSize=10, textColor=GRAY,  fontName="Helvetica-Oblique", alignment=TA_CENTER, spaceBefore=4),
        "footer":        ParagraphStyle("footer",        fontSize=7.5,textColor=GRAY,  fontName="Helvetica", alignment=TA_CENTER),
    }

# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt(value, is_currency=False):
    if value is None or value == "":
        return "N/A"
    if is_currency:
        try:    return f"{float(str(value).replace(',', '')):,.0f} DZD"
        except: return str(value)
    return str(value)

def to_float(value):
    try:    return float(str(value).replace(",", "").replace(" ", ""))
    except: return None

def info_table(rows, s):
    t = Table(rows, colWidths=[62*mm, 112*mm])
    t.setStyle(TableStyle([
        ("FONTNAME",       (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME",       (1,0), (1,-1), "Helvetica"),
        ("FONTSIZE",       (0,0), (-1,-1), 9.5),
        ("TEXTCOLOR",      (0,0), (0,-1), NAVY),
        ("TEXTCOLOR",      (1,0), (1,-1), GRAY),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.white, colors.HexColor("#F5F8FC")]),
        ("GRID",           (0,0), (-1,-1), 0.4, colors.HexColor("#CCCCCC")),
        ("TOPPADDING",     (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",  (0,0), (-1,-1), 5),
        ("LEFTPADDING",    (0,0), (-1,-1), 8),
    ]))
    return t

# ── Build ─────────────────────────────────────────────────────────────────────
def build_from_data(data: dict, out_path: str):
    """Called directly by loan_graph.py — no JSON file needed."""
    s        = styles()
    laf      = data["documents"].get("loan_application_form", {}).get("fields", {})
    pay      = data["documents"].get("payslip",               {}).get("fields", {})
    bank     = data["documents"].get("bank_statement",        {}).get("fields", {})
    val      = data["validation"]
    fraud    = data.get("fraud_detection", {})
    decision = data["agent_decision"]
    case_id  = data["case_id"]

    doc   = SimpleDocTemplate(str(out_path), pagesize=A4,
                topMargin=10*mm, bottomMargin=15*mm,
                leftMargin=18*mm, rightMargin=18*mm)
    story = []

    # ── Header ────────────────────────────────────────────────────────────────
    header = Table(
        [[Paragraph("LOAN DECISION REPORT", s["title"])],
         [Paragraph("Banque Nationale d'Algérie — AI Credit Review Platform", s["subtitle"])],
         [Paragraph(f"Case: {case_id}  |  {datetime.now().strftime('%d/%m/%Y %H:%M')}", s["subtitle"])]],
        colWidths=[174*mm]
    )
    header.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), NAVY),
        ("TOPPADDING",    (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("LEFTPADDING",   (0,0), (-1,-1), 12),
    ]))
    story.append(header)
    story.append(Spacer(1, 6*mm))

    # ── Decision box ──────────────────────────────────────────────────────────
    rec       = decision.get("recommendation") or "N/A"
    dec_color = GREEN if rec == "Approve" else (ORANGE if "Conditions" in rec else RED)
    dec_label = "APPROVED" if rec == "Approve" else ("APPROVED WITH CONDITIONS" if "Conditions" in rec else "REJECTED")
    conf      = decision.get("confidence")
    conf_str  = f"{round(conf * 100)}%" if conf else "N/A"

    dec_box = Table(
        [[Paragraph(f'<font color="#{dec_color.hexval()[2:]}">{dec_label}</font>', s["decision"])],
         [Paragraph(f"Risk: <b>{decision.get('risk_level', 'N/A')}</b>  |  Confidence: <b>{conf_str}</b>", s["justification"])],
         [Paragraph(decision.get("justification") or "", s["justification"])]],
        colWidths=[174*mm]
    )
    dec_box.setStyle(TableStyle([
        ("BOX",           (0,0), (-1,-1), 1.5, dec_color),
        ("BACKGROUND",    (0,0), (-1,-1), colors.HexColor("#F8F9FA")),
        ("TOPPADDING",    (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("LEFTPADDING",   (0,0), (-1,-1), 12),
    ]))
    story.append(dec_box)
    story.append(Spacer(1, 5*mm))

    # ── Section 1: Applicant ──────────────────────────────────────────────────
    story.append(Paragraph("1. Applicant Information", s["section"]))
    story.append(info_table([
        ["Full Name",      fmt(laf.get("full_name"))],
        ["Date of Birth",  fmt(laf.get("date_of_birth"))],
        ["Gender",         fmt(laf.get("gender"))],
        ["Marital Status", fmt(laf.get("marital_status"))],
        ["Dependents",     fmt(laf.get("dependents"))],
        ["Education",      fmt(laf.get("education"))],
        ["Self Employed",  fmt(laf.get("self_employed"))],
        ["Property Area",  fmt(laf.get("property_area"))],
        ["Credit History", "Good (1)" if laf.get("credit_history") == 1 else "Bad (0)" if laf.get("credit_history") == 0 else "N/A"],
    ], s))
    story.append(Spacer(1, 4*mm))

    # ── Section 2: Financials ─────────────────────────────────────────────────
    story.append(Paragraph("2. Financial Summary", s["section"]))
    story.append(info_table([
        ["Applicant Income",    fmt(laf.get("applicant_income"),         True)],
        ["Co-Applicant Income", fmt(laf.get("coapplicant_income"),       True)],
        ["Gross Salary",        fmt(pay.get("gross_salary"),             True)],
        ["Net Salary",          fmt(pay.get("net_salary"),               True)],
        ["Employer",            fmt(pay.get("employer_name"))],
        ["Employment Type",     fmt(pay.get("employment_type"))],
        ["Avg Monthly Inflow",  fmt(bank.get("average_monthly_inflow"),  True)],
        ["Avg Monthly Outflow", fmt(bank.get("average_monthly_outflow"), True)],
        ["Closing Balance",     fmt(bank.get("closing_balance"),         True)],
    ], s))
    story.append(Spacer(1, 4*mm))

    # ── Section 3: Loan ───────────────────────────────────────────────────────
    story.append(Paragraph("3. Loan Request", s["section"]))
    loan_amt = to_float(laf.get("loan_amount_requested")) or 0
    term     = to_float(laf.get("loan_term_months")) or 1
    try:    monthly = f"{loan_amt / term:,.0f} DZD"
    except: monthly = "N/A"
    story.append(info_table([
        ["Loan Amount",          fmt(laf.get("loan_amount_requested"), True)],
        ["Loan Term",            f"{fmt(laf.get('loan_term_months'))} months"],
        ["Declared Purpose",     fmt(laf.get("declared_purpose"))],
        ["Est. Monthly Payment", monthly],
    ], s))
    story.append(Spacer(1, 4*mm))

    # ── Section 4: Validation ─────────────────────────────────────────────────
    story.append(Paragraph("4. Validation Results", s["section"]))
    val_status = (val.get("status") or "N/A").upper()
    val_color  = GREEN if val_status == "PASSED" else (ORANGE if val_status == "WARNING" else RED)
    issues     = val.get("issues") or []
    val_rows   = [["Validation Status",
                   Paragraph(f'<font color="#{val_color.hexval()[2:]}"><b>{val_status}</b></font>', s["body"])]]
    if issues:
        for issue in issues:
            val_rows.append(["Issue", Paragraph(f'<font color="#CF0A2C">{issue}</font>', s["body"])])
    else:
        val_rows.append(["", Paragraph('<font color="#155724">All checks passed</font>', s["body"])])
    story.append(info_table(val_rows, s))
    story.append(Spacer(1, 4*mm))

    # ── Section 5: Fraud ──────────────────────────────────────────────────────
    story.append(Paragraph("5. Fraud Detection", s["section"]))
    is_fraud        = fraud.get("is_fraud", False)
    fraud_score     = fraud.get("fraud_score")
    fraud_signals   = fraud.get("signals") or []
    fraud_just      = fraud.get("justification") or ""
    fraud_color     = RED if is_fraud else GREEN
    fraud_label     = "FRAUD SUSPECTED" if is_fraud else "No Fraud Detected"
    fraud_score_str = f"{round(fraud_score * 100)}%" if fraud_score is not None else "N/A"

    fraud_rows = [
        ["Status",      Paragraph(f'<font color="#{fraud_color.hexval()[2:]}"><b>{fraud_label}</b></font>', s["body"])],
        ["Fraud Score", Paragraph(f'<font color="#{fraud_color.hexval()[2:]}"><b>{fraud_score_str}</b></font>', s["body"])],
        ["Threshold",   "40%"],
    ]
    if fraud_signals:
        for signal in fraud_signals:
            fraud_rows.append(["Signal", Paragraph(f'<font color="#CF0A2C">{signal}</font>', s["body"])])
    else:
        fraud_rows.append(["Signals", Paragraph('<font color="#155724">No suspicious signals detected</font>', s["body"])])
    if fraud_just:
        fraud_rows.append(["Assessment", Paragraph(fraud_just, s["body"])])
    story.append(info_table(fraud_rows, s))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 5*mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=NAVY))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        f"Auto-generated by the AI Credit Review Platform. "
        f"This report assists the credit officer and does not replace human judgment. "
        f"{datetime.now().strftime('%d/%m/%Y %H:%M')}",
        s["footer"]
    ))

    doc.build(story)
    print(f"Report saved → {out_path}")
    


if __name__ == "__main__":
    case_id = sys.argv[1] if len(sys.argv) > 1 else None
    if not case_id:
        print("Usage: python report.py CASE-2025-0001")
        sys.exit(1)
    build_from_data(case_id)
    