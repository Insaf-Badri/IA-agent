import os
import json
from pathlib import Path
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from datetime import datetime, date

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

print("DATABASE_URL =", os.getenv("DATABASE_URL"))

def get_conn():
    return psycopg2.connect(os.getenv("DATABASE_URL"))

def json_serial(obj):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

def safe_json(val):
    """Safely convert any value to a JSON string for psycopg2."""
    return json.dumps(val if val is not None else {}, default=json_serial)

def init_db():
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS loans (
            id          SERIAL PRIMARY KEY,
            case_id     VARCHAR(50) UNIQUE NOT NULL,
            status      VARCHAR(40) DEFAULT 'new',
            documents   JSONB DEFAULT '{}',
            validation  JSONB DEFAULT '{}',
            fraud       JSONB DEFAULT '{}',
            decision    JSONB DEFAULT '{}',
            raw_data    JSONB DEFAULT '{}',
            retry_count INTEGER DEFAULT 0,
            created_at  TIMESTAMP DEFAULT NOW(),
            updated_at  TIMESTAMP DEFAULT NOW()
        )
    """)
    conn.commit()
    cur.close()
    conn.close()

def load_case(case_id: str) -> dict:
    conn = get_conn()
    cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT * FROM loans WHERE case_id = %s", [case_id])
    row  = cur.fetchone()
    cur.close()
    conn.close()
    if not row:
        return {
            "case_id":     case_id,
            "documents":   {},
            "validation":  {},
            "fraud":       {},
            "decision":    {},
            "status":      "new",
            "retry_count": 0,
        }
    result = dict(row)
    for key in ["documents", "validation", "fraud", "decision", "raw_data"]:
        if isinstance(result.get(key), str):
            result[key] = json.loads(result[key])
    return result

def save_case(case_id: str, data: dict):
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("""
        INSERT INTO loans
            (case_id, status, documents, validation, fraud, decision, raw_data, retry_count, updated_at)
        VALUES
            (%s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s, NOW())
        ON CONFLICT (case_id) DO UPDATE SET
            status      = EXCLUDED.status,
            documents   = EXCLUDED.documents,
            validation  = EXCLUDED.validation,
            fraud       = EXCLUDED.fraud,
            decision    = EXCLUDED.decision,
            raw_data    = EXCLUDED.raw_data,
            retry_count = EXCLUDED.retry_count,
            updated_at  = NOW()
    """, [
        case_id,
        data.get("status", "new"),
        safe_json(data.get("documents",  {})),
        safe_json(data.get("validation", {})),
        safe_json(data.get("fraud",      {})),
        safe_json(data.get("decision",   {})),
        safe_json(data.get("raw_data",   {})),
        data.get("retry_count", 0),
    ])
    conn.commit()
    cur.close()
    conn.close()

def increment_retry(case_id: str):
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute(
        "UPDATE loans SET retry_count = retry_count + 1 WHERE case_id = %s",
        [case_id]
    )
    conn.commit()
    cur.close()
    conn.close()