# Backend/tasks.py
import os
import httpx
from celery import Celery
from agents.loan_graph import loan_agent, LoanState
from .db import load_case, save_case
from dotenv import load_dotenv
from langsmith import trace
from langgraph_sdk import get_sync_client
from langgraph.pregel.remote import RemoteGraph


celery_app = Celery(
    "loan_review",
    broker=os.getenv("REDIS_URL",  "redis://redis:6379/0"),
    backend=os.getenv("REDIS_URL", "redis://redis:6379/0"),
)

LG_URL = os.getenv("LANGGRAPH_SERVER_URL", "http://host.docker.internal:2024")

def get_graph():
    return RemoteGraph("loan_pipeline", url=LG_URL)

@celery_app.task(bind=True, name="process_loan_case")
def process_loan_case(self, case_id: str):
    existing = load_case(case_id)

    initial: LoanState = {
        "case_id":     case_id,
        "documents":   existing.get("documents",  {}),
        "validation":  existing.get("validation", {}),
        "fraud":       existing.get("fraud",       {}),
        "decision":    existing.get("decision",    {}),
        "report_path": "",
        "status":      "started",
        "retry_count": existing.get("retry_count", 0) + 1,
        "error":       "",
    }

    self.update_state(state="STARTED", meta={"case_id": case_id})

    with trace(
        name="process_loan_case",
        run_type="chain",
        tags=["loan", "celery"],
        metadata={"case_id": case_id, "retry_count": initial["retry_count"]},
    ) as run:
        try:
            with httpx.Client(timeout=120) as client:

                # 1. Create a thread
                thread = client.post(f"{LG_URL}/threads", json={}).json()
                thread_id = thread["thread_id"]

                # 2. Start the run — Studio sees this immediately
                run_resp = client.post(
                    f"{LG_URL}/threads/{thread_id}/runs",
                    json={
                        "assistant_id": "loan_pipeline",
                        "input": initial,
                        "config": {
                            "tags": ["loan", "celery"],
                            "metadata": {
                                "case_id":     case_id,
                                "retry_count": initial["retry_count"],
                            },
                        },
                    }
                ).json()
                run_id = run_resp["run_id"]

                # 3. Wait for completion
                client.get(
                    f"{LG_URL}/threads/{thread_id}/runs/{run_id}/join"
                )

                # 4. Get final state
                final = client.get(
                    f"{LG_URL}/threads/{thread_id}/state"
                ).json()["values"]

            result = {
                "case_id":  case_id,
                "status":   final["status"],
                "decision": final["decision"],
                "report":   final.get("report_path", ""),
            }

            save_case(case_id, {**existing, **result})
            run.end(outputs=result)
            return result

        except Exception as e:
            error_payload = {**existing, "status": "error", "error": str(e)}
            save_case(case_id, error_payload)
            run.end(error=str(e))
            raise