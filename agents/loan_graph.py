# agents/loan_graph.py
import os
import json

import joblib
import pandas as pd
import pdfplumber
from pathlib import Path
from typing import TypedDict, Literal
from groq import Groq
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from Backend.db      import load_case, save_case, increment_retry
from Backend.storage import download_pdf, upload_report

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL  = "llama-3.1-8b-instant"

BASE_DIR       = Path(__file__).parent.parent
MODEL_PATH     = BASE_DIR / "ML" / "models" / "fraud_model.pkl"
FRAUD_THRESHOLD       = 0.40
FRAUD_AUTO_REJECT     = 0.85   # your rule: auto-reject above this
CONFIDENCE_MIN        = 0.70   # your rule: human review below this
MAX_RETRIES           = 1      # your rule: retry extract once then fail

FEATURE_COLUMNS = [
    "income_declared", "income_detected", "address_mismatch",
    "device_location", "document_authenticity_score",
    "account_balance_pattern", "employment_mismatch", "rapid_loan_requests",
]

# ══════════════════════════════════════════════════════
# STATE — replaces your JSON files entirely
# ══════════════════════════════════════════════════════

class LoanState(TypedDict):
    case_id:          str
    documents:        dict    # extract output
    validation:       dict    # validate output
    fraud:            dict    # fraud output
    decision:         dict    # decide output
    report_path:      str     # MinIO path to PDF
    status:           str
    retry_count:      int
    error:            str

# ══════════════════════════════════════════════════════
# HELPERS (shared across nodes)
# ══════════════════════════════════════════════════════

def to_float(v, default=0.0):
    try:    return float(str(v).replace(",", "").replace(" ", ""))
    except: return default

def read_pdf(local_path: str) -> str | None:
    try:
        with pdfplumber.open(local_path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception as e:
        print(f"  ✗ PDF read error: {e}")
        return None

SCHEMAS = {
    "loan_application_form": {
        "full_name":"","date_of_birth":"","gender":"","marital_status":"",
        "dependents":0,"education":"","self_employed":False,
        "applicant_income":0,"coapplicant_income":0,
        "loan_amount_requested":0,"loan_term_months":0,
        "property_area":"","declared_purpose":"","credit_history":0
    },
    "payslip": {
        "employee_name":"","employer_name":"","employment_type":"",
        "pay_period":"","gross_salary":0,"net_salary":0,
        "deductions":0,"currency":""
    },
    "bank_statement": {
        "account_holder_name":"","bank_name":"","statement_period":"",
        "average_monthly_inflow":0,"average_monthly_outflow":0,
        "closing_balance":0,"currency":""
    }
}

# ══════════════════════════════════════════════════════
# NODE 1 — EXTRACT
# ══════════════════════════════════════════════════════

def extract_node(state: LoanState) -> LoanState:
    case_id     = state["case_id"]
    # ✅ FIX: read current count first, then increment so the router
    #         always sees the updated value on the next validation pass.
    retry_count = state.get("retry_count", 0)
    print(f"\n[Extract] Starting — {case_id} (attempt {retry_count + 1})")
    documents = {}

    for doc_type in ["loan_application_form", "payslip", "bank_statement"]:
        local_path = download_pdf(case_id, doc_type)
        if not local_path:
            print(f"  ✗ Missing: {doc_type}.pdf")
            documents[doc_type] = {"present": False, "fields": {}}
            continue

        raw_text = read_pdf(local_path)
        if not raw_text:
            documents[doc_type] = {"present": False, "fields": {}}
            continue

        schema = SCHEMAS[doc_type]
        prompt = (
            f"Extract these fields from the document. "
            f"Return ONLY a JSON object, no explanation, no markdown.\n"
            f"Fields: {json.dumps(schema, indent=2)}\n"
            f"Document:\n---\n{raw_text}\n---"
        )
        resp = client.chat.completions.create(
            model=MODEL, max_tokens=800,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = resp.choices[0].message.content.strip().strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
        try:
            fields = json.loads(raw)
        except json.JSONDecodeError:
            fields = {}

        documents[doc_type] = {
            "present":     True,
            "source_file": f"{doc_type}.pdf",
            "fields":      fields
        }
        print(f"  ✓ Extracted {len(fields)} fields for {doc_type}")

    new_state = {
        **state,
        "documents":   documents,
        "status":      "extracted",
        "retry_count": retry_count + 1,   # ✅ FIX: increment persisted in state
    }
    data = {**load_case(case_id), **new_state}
    save_case(case_id, data)
    return new_state

# ══════════════════════════════════════════════════════
# NODE 2 — VALIDATE
# ══════════════════════════════════════════════════════

def validate_node(state: LoanState) -> LoanState:
    case_id = state["case_id"]
    print(f"\n[Validate] Starting — {case_id}")
    docs    = state["documents"]
    laf     = docs.get("loan_application_form", {}).get("fields", {})
    pay     = docs.get("payslip", {}).get("fields", {})
    bank    = docs.get("bank_statement", {}).get("fields", {})

    issues   = []
    warnings = []

    # ── Critical checks (trigger re-extract if failed) ─────────────────
    required = ["full_name", "applicant_income", "loan_amount_requested",
                "loan_term_months", "credit_history"]
    missing  = [f for f in required if not laf.get(f) and laf.get(f) != 0]
    if missing:
        issues.append(f"Missing required fields: {', '.join(missing)}")

    if not docs.get("loan_application_form", {}).get("present"):
        issues.append("Loan application form not found")
    if not docs.get("payslip", {}).get("present"):
        issues.append("Payslip not found")
    if not docs.get("bank_statement", {}).get("present"):
        issues.append("Bank statement not found")

    # ── Value checks ────────────────────────────────────────────────────
    income = to_float(laf.get("applicant_income"))
    loan   = to_float(laf.get("loan_amount_requested"))
    term   = to_float(laf.get("loan_term_months"))

    if income <= 0:
        issues.append("Applicant income is zero or missing")
    if loan <= 0:
        issues.append("Loan amount is zero or missing")
    if term <= 0:
        issues.append("Loan term is zero or missing")

    # ── Warning checks ──────────────────────────────────────────────────
    gross = to_float(pay.get("gross_salary"))
    if gross > 0 and income > 0:
        gap = abs(income - gross) / gross
        if gap > 0.30:
            warnings.append(f"Income discrepancy: declared {income:,.0f} vs payslip {gross:,.0f}")

    closing = to_float(bank.get("closing_balance"))
    if closing < loan * 0.05:
        warnings.append("Low savings relative to loan amount")

    status = "failed" if issues else ("warning" if warnings else "passed")

    validation = {
        "status":   status,
        "issues":   issues,
        "warnings": warnings,
    }
    print(f"  Validation: {status.upper()} | issues={len(issues)} warnings={len(warnings)}")

    new_state = {**state, "validation": validation, "status": f"validated_{status}"}
    data      = {**load_case(case_id), **new_state}
    save_case(case_id, data)
    return new_state

# ══════════════════════════════════════════════════════
# NODE 3 — FRAUD
# ══════════════════════════════════════════════════════

def fraud_node(state: LoanState) -> LoanState:
    case_id = state["case_id"]
    print(f"\n[Fraud] Starting — {case_id}")
    docs = state["documents"]
    laf  = docs.get("loan_application_form", {}).get("fields", {})
    pay  = docs.get("payslip",               {}).get("fields", {})
    bank = docs.get("bank_statement",        {}).get("fields", {})

    # ── Compute features ────────────────────────────────────────────────
    income_declared = to_float(laf.get("applicant_income"))
    income_detected = to_float(pay.get("gross_salary"))

    loan_name = (laf.get("full_name")           or "").lower().strip()
    pay_name  = (pay.get("employee_name")        or "").lower().strip()
    bank_name = (bank.get("account_holder_name") or "").lower().strip()
    address_mismatch = 1 if (
        (loan_name and pay_name  and loan_name != pay_name) or
        (loan_name and bank_name and loan_name != bank_name)
    ) else 0

    required    = ["full_name","applicant_income","loan_amount_requested",
                   "loan_term_months","credit_history"]
    num_missing = sum(1 for f in required if not laf.get(f))
    doc_auth    = round(1.0 - (num_missing / len(required)), 2)

    closing = to_float(bank.get("closing_balance"))
    inflow  = to_float(bank.get("average_monthly_inflow"))
    balance_pattern = round(min(closing / inflow, 2.0) if inflow > 0 else 0.0, 4)

    form_se      = str(laf.get("self_employed","")).lower() in ["true","yes","1"]
    pay_type     = (pay.get("employment_type") or "").lower()
    emp_mismatch = 1 if (
        (form_se and "permanent" in pay_type) or
        (not form_se and "self" in pay_type)
    ) else 0

    features = {
        "income_declared":             income_declared,
        "income_detected":             income_detected,
        "address_mismatch":            address_mismatch,
        "device_location":             0,
        "document_authenticity_score": doc_auth,
        "account_balance_pattern":     balance_pattern,
        "employment_mismatch":         emp_mismatch,
        "rapid_loan_requests":         0,
    }

    # ── Run ML model ─────────────────────────────────────────────────────
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"fraud_model.pkl not found at {MODEL_PATH}")
    model    = joblib.load(MODEL_PATH)
    df       = pd.DataFrame([features], columns=FEATURE_COLUMNS)
    score    = round(float(model.predict_proba(df)[0][1]), 4)
    is_fraud = score >= FRAUD_THRESHOLD

    # ── Signals ──────────────────────────────────────────────────────────
    signals = []
    inc_gap = abs(income_declared - income_detected)
    inc_pct = inc_gap / income_detected if income_detected > 0 else 0
    if inc_pct > 0.15:
        signals.append(f"Income gap: declared {income_declared:,.0f} vs detected {income_detected:,.0f} ({inc_pct:.0%})")
    if address_mismatch:
        signals.append("Name mismatch across documents")
    if doc_auth < 0.6:
        signals.append(f"Low document authenticity score: {doc_auth}")
    if balance_pattern < 0.1:
        signals.append(f"Near-zero savings vs loan: balance pattern {balance_pattern}")
    if emp_mismatch:
        signals.append("Employment type mismatch")

    # ── LLM justification ────────────────────────────────────────────────
    prompt = (
        f"You are a fraud analyst. Summarize this fraud assessment in 2 sentences.\n"
        f"Fraud score: {score:.0%} (threshold: {FRAUD_THRESHOLD:.0%})\n"
        f"Flagged: {is_fraud}\nSignals: {signals if signals else 'None'}\n"
        f"income_declared={income_declared}, income_detected={income_detected}, "
        f"address_mismatch={address_mismatch}, doc_auth={doc_auth}"
    )
    resp = client.chat.completions.create(
        model=MODEL, max_tokens=150,
        messages=[{"role": "user", "content": prompt}]
    )
    summary = resp.choices[0].message.content.strip()

    flag = "FRAUD SUSPECTED" if is_fraud else "No fraud detected"
    print(f"  Score: {score:.0%} | {flag}")

    fraud = {
        "is_fraud":      is_fraud,
        "fraud_score":   score,
        "signals":       signals,
        "threshold":     FRAUD_THRESHOLD,
        "justification": summary,
    }
    new_state = {**state, "fraud": fraud, "status": "fraud_checked"}
    data      = {**load_case(case_id), **new_state}
    save_case(case_id, data)
    return new_state

# ══════════════════════════════════════════════════════
# NODE 4 — DECIDE
# ══════════════════════════════════════════════════════

def decide_node(state: LoanState) -> LoanState:
    case_id = state["case_id"]
    print(f"\n[Decide] Starting — {case_id}")
    docs  = state["documents"]
    val   = state["validation"]
    fraud = state["fraud"]
    laf   = docs.get("loan_application_form", {}).get("fields", {})
    pay   = docs.get("payslip",               {}).get("fields", {})
    bank  = docs.get("bank_statement",        {}).get("fields", {})

    income      = to_float(laf.get("applicant_income"))
    co_income   = to_float(laf.get("coapplicant_income"))
    total_inc   = income + co_income
    loan        = to_float(laf.get("loan_amount_requested"))
    term        = to_float(laf.get("loan_term_months")) or 1
    net_salary  = to_float(pay.get("net_salary"))
    closing     = to_float(bank.get("closing_balance"))
    credit_hist = to_float(laf.get("credit_history"), 0)
    dti         = (loan / term) / total_inc if total_inc > 0 else 1

    reasons = []

    # ── Hard rejects ─────────────────────────────────────────────────────
    if credit_hist == 0:
        reasons.append("Bad credit history")
    if dti > 0.60:
        reasons.append(f"DTI ratio critical: {dti:.0%} (max 60%)")
    if len(reasons) >= 2:
        recommendation, risk, confidence = "Reject", "High", 0.85
    else:
        # ── Conditional approvals ─────────────────────────────────────
        conditions = []
        if val.get("status") == "warning":
            conditions.append("Validation warnings present")
        if 0.35 < dti <= 0.60:
            conditions.append(f"DTI ratio elevated: {dti:.0%}")
        if credit_hist == 0:
            conditions.append("Poor credit history")
        if closing < loan * 0.05:
            conditions.append("Low savings relative to loan")
        if net_salary > 0 and loan / (net_salary * 12) > 4:
            conditions.append(f"Loan is {loan/(net_salary*12):.1f}x annual net salary")

        if conditions:
            reasons = conditions
            recommendation, risk, confidence = "Approve with Conditions", "Medium", 0.75
        else:
            reasons = ["All checks passed"]
            recommendation, risk, confidence = "Approve", "Low", 0.90

    # ── LLM justification ────────────────────────────────────────────────
    prompt = (
        f"You are a senior credit analyst. Write a 2-sentence justification.\n"
        f"Decision: {recommendation}\nRisk: {risk}\n"
        f"Reasons: {', '.join(reasons)}\n"
        f"Applicant: {laf.get('full_name')}\n"
        f"Income: {income:,.0f} + co-applicant {co_income:,.0f} DZD/month\n"
        f"Loan: {loan:,.0f} DZD over {int(term)} months | DTI: {dti:.0%}\n"
        f"Credit history: {credit_hist} | Fraud score: {fraud.get('fraud_score',0):.0%}\n"
        f"Write exactly 2 clear professional sentences. No bullet points."
    )
    resp = client.chat.completions.create(
        model=MODEL, max_tokens=150,
        messages=[{"role": "user", "content": prompt}]
    )
    justification = resp.choices[0].message.content.strip()

    print(f"  Decision: {recommendation} | Risk: {risk} | Confidence: {confidence:.0%}")

    decision = {
        "recommendation": recommendation,
        "risk_level":     risk,
        "confidence":     confidence,
        "justification":  justification,
        "reasons":        reasons,
    }
    new_state = {**state, "decision": decision, "status": "decided"}
    data      = {**load_case(case_id), **new_state}
    save_case(case_id, data)
    return new_state

# ══════════════════════════════════════════════════════
# NODE 5 — AUTO-REJECT (high fraud, skip decide)
# ══════════════════════════════════════════════════════

def auto_reject_node(state: LoanState) -> LoanState:
    case_id = state["case_id"]
    print(f"\n[Auto-Reject] Fraud score {state['fraud']['fraud_score']:.0%} — auto-rejecting")

    decision = {
        "recommendation": "Reject",
        "risk_level":     "Very High",
        "confidence":     0.95,
        "justification":  (
            f"Application automatically rejected due to high fraud score "
            f"({state['fraud']['fraud_score']:.0%}) exceeding the auto-reject "
            f"threshold ({FRAUD_AUTO_REJECT:.0%}). "
            f"Signals: {'; '.join(state['fraud'].get('signals', ['None']))}."
        ),
        "reasons": ["Fraud score above auto-reject threshold"],
    }
    new_state = {**state, "decision": decision, "status": "auto_rejected"}
    data      = {**load_case(case_id), **new_state}
    save_case(case_id, data)
    return new_state

# ══════════════════════════════════════════════════════
# NODE 6 — REPORT
# ══════════════════════════════════════════════════════

def report_node(state: LoanState) -> LoanState:
    case_id = state["case_id"]
    print(f"\n[Report] Building PDF — {case_id}")

    import tempfile
    from agents.report import build_from_data

    data = {
        "case_id":         case_id,
        "documents":       state["documents"],
        "validation":      state["validation"],
        "fraud_detection": state["fraud"],
        "agent_decision":  state["decision"],
        "status":          state["status"],
    }

    tmp_path   = str(Path(tempfile.gettempdir()) / f"{case_id}_report.pdf")
    build_from_data(data, tmp_path)

    minio_path = upload_report(case_id, tmp_path)
    print(f"  Report uploaded → {minio_path}")

    new_state = {**state, "report_path": minio_path, "status": "completed"}
    db_data   = {**load_case(case_id), **new_state}
    save_case(case_id, db_data)
    return new_state

# ══════════════════════════════════════════════════════
# NODE 7 — HUMAN REVIEW (pause point)
# ══════════════════════════════════════════════════════

def human_review_node(state: LoanState) -> LoanState:
    case_id = state["case_id"]
    print(f"\n[Human Review] Case {case_id} requires human review")
    new_state = {**state, "status": "awaiting_human_review"}
    data      = {**load_case(case_id), **new_state}
    save_case(case_id, data)
    return new_state

# ══════════════════════════════════════════════════════
# NODE 8 — FAILED
# ══════════════════════════════════════════════════════

def failed_node(state: LoanState) -> LoanState:
    case_id = state["case_id"]
    print(f"\n[Failed] Case {case_id} — extraction failed after {MAX_RETRIES + 1} attempts")
    new_state = {**state, "status": "failed"}
    data      = {**load_case(case_id), **new_state}
    save_case(case_id, data)
    return new_state

# ══════════════════════════════════════════════════════
# ROUTING FUNCTIONS
# ══════════════════════════════════════════════════════

def route_after_validate(state: LoanState) -> Literal["fraud", "extract", "failed"]:
    status = state["validation"].get("status")
    if status == "failed":
        # ✅ FIX: retry_count was already incremented inside extract_node,
        #         so we compare against MAX_RETRIES directly — no mutation here.
        if state["retry_count"] <= MAX_RETRIES:
            print(f"  → Validation failed, retrying extraction "
                  f"(attempt {state['retry_count'] + 1}/{MAX_RETRIES + 1})")
            return "extract"
        else:
            print(f"  → Retry exhausted → failed")
            return "failed"
    return "fraud"   # passed or warning → continue

def route_after_fraud(state: LoanState) -> Literal["decide", "auto_reject"]:
    score = state["fraud"].get("fraud_score", 0)
    if score > FRAUD_AUTO_REJECT:
        print(f"  → Fraud score {score:.0%} > {FRAUD_AUTO_REJECT:.0%} → auto-reject")
        return "auto_reject"
    return "decide"

def route_after_decide(state: LoanState) -> Literal["report", "human_review"]:
    confidence = state["decision"].get("confidence", 1.0)
    if confidence < CONFIDENCE_MIN:
        print(f"  → Confidence {confidence:.0%} < {CONFIDENCE_MIN:.0%} → human review")
        return "human_review"
    return "report"

# ══════════════════════════════════════════════════════
# BUILD THE GRAPH
# ══════════════════════════════════════════════════════

def build_loan_graph():
    graph = StateGraph(LoanState)

    # Register nodes
    graph.add_node("extract",      extract_node)
    graph.add_node("validate",     validate_node)
    graph.add_node("fraud",        fraud_node)
    graph.add_node("decide",       decide_node)
    graph.add_node("auto_reject",  auto_reject_node)
    graph.add_node("report",       report_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("failed",       failed_node)

    # Entry
    graph.set_entry_point("extract")

    # Fixed edges
    graph.add_edge("extract", "validate")

    # Conditional edges
    graph.add_conditional_edges("validate", route_after_validate, {
        "fraud":   "fraud",
        "extract": "extract",   # retry loop
        "failed":  "failed",
    })
    graph.add_conditional_edges("fraud", route_after_fraud, {
        "decide":      "decide",
        "auto_reject": "auto_reject",
    })
    graph.add_conditional_edges("decide", route_after_decide, {
        "report":       "report",
        "human_review": "human_review",
    })

    # Terminal nodes
    graph.add_edge("auto_reject",  "report")   # still generate report on reject
    graph.add_edge("report",       END)
    graph.add_edge("human_review", END)
    graph.add_edge("failed",       END)

    return graph.compile()

# Singleton — loaded once at worker startup
loan_agent = build_loan_graph()