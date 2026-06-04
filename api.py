# Backend/api.py
from dotenv import load_dotenv
load_dotenv(override=False)
from pathlib import Path
load_dotenv(Path(__file__).resolve().parent / ".env")
import io
from fastapi import BackgroundTasks, FastAPI, UploadFile, File, HTTPException
from Backend.storage import get_minio
from Backend.db      import load_case, init_db
from celery.result   import AsyncResult
from Backend.tasks import process_loan_case, celery_app  
from fastapi import FastAPI, BackgroundTasks


app = FastAPI(title="Loan Review Agent API", version="2.0")

load_dotenv(Path(__file__).resolve().parent / ".env")

@app.on_event("startup")
def startup():
    init_db()
    from Backend.storage import init_buckets
    init_buckets()

# ── 1. Upload PDFs ─────────────────────────────────────────────────────────
@app.post("/api/loans/{case_id}/upload/{doc_type}")
async def upload_document(case_id: str, doc_type: str,
                           file: UploadFile = File(...)):
    allowed = ["loan_application_form", "payslip", "bank_statement"]
    if doc_type not in allowed:
        raise HTTPException(400, f"doc_type must be one of {allowed}")
    minio = get_minio()
    data  = await file.read()
    minio.put_object("loan-documents", f"{case_id}/{doc_type}.pdf",
                     io.BytesIO(data), len(data),
                     content_type="application/pdf")
    return {"uploaded": f"{case_id}/{doc_type}.pdf"}

# ── 2. Trigger agent pipeline ───────────────────────────────────────────────
@app.post("/api/loans/{case_id}/process")
def trigger_review(case_id: str):
    task = process_loan_case.delay(case_id)
    return {"case_id": case_id, "job_id": task.id, "status": "queued"}

# ── 3. Pipeline status ──────────────────────────────────────────────────────

@app.get("/api/loans/{case_id}/status")
def get_status(job_id: str):
    result = AsyncResult(job_id, app=celery_app)  # ← pass app=celery_app
    return {"job_id": job_id, "state": result.state, "detail": result.info}
# ── 4. Final result ─────────────────────────────────────────────────────────
@app.get("/api/loans/{case_id}/result")
def get_result(case_id: str):
    data = load_case(case_id)
    if data["status"] == "new":
        raise HTTPException(404, "Case not found")
    return data

# ── 5. Human review decision (POST approval after human_review pause) ───────
@app.post("/api/loans/{case_id}/human-decision")
def human_decision(case_id: str, approved: bool, reviewer_notes: str = ""):
    data = load_case(case_id)
    if data["status"] != "awaiting_human_review":
        raise HTTPException(400, "Case is not awaiting human review")
    data["decision"]["human_override"]  = "Approve" if approved else "Reject"
    data["decision"]["reviewer_notes"]  = reviewer_notes
    data["status"] = "completed_human_review"
    from Backend.db import save_case
    save_case(case_id, data)
    return {"case_id": case_id, "human_decision": data["decision"]["human_override"]}

@app.post("/submit-case")
def submit_case(case_id: str, documents: dict, background_tasks: BackgroundTasks):
    # Fire off the Celery task asynchronously
    process_loan_case.delay(case_id, documents)
    return {"status": "processing", "case_id": case_id}