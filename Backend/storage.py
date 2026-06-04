# Backend/storage.py
import os
import tempfile
from pathlib import Path
from minio import Minio
from minio.error import S3Error

BUCKETS = ["loan-documents", "loan-reports"]

DOWNLOAD_DIR = Path(tempfile.gettempdir()) / "loan_agent_docs"
DOWNLOAD_DIR.mkdir(exist_ok=True)

def get_minio():
    endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    print(f"🔌 Connecting to MinIO at: {endpoint}")
    return Minio(
        os.getenv("MINIO_ENDPOINT", "localhost:9000"),
        access_key=os.getenv("MINIO_ACCESS_KEY", "insaf"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "insaf123"),
        secure=False
    )

def init_buckets():
    client = get_minio()
    for bucket in BUCKETS:
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
            print(f"✅ Created bucket: {bucket}")

def download_pdf(case_id: str, doc_type: str) -> str | None:
    client      = get_minio()
    object_name = f"{case_id}/{doc_type}.pdf"
    local_path  = DOWNLOAD_DIR / f"{case_id}_{doc_type}.pdf"
    try:
        client.fget_object("loan-documents", object_name, str(local_path))
        return str(local_path)
    except S3Error as e:
        print(f"  ✗ MinIO download failed for {object_name}: {e}")
        return None

def upload_report(case_id: str, local_path: str) -> str:
    client      = get_minio()
    object_name = f"{case_id}_report.pdf"
    client.fput_object("loan-reports", object_name, local_path,
                       content_type="application/pdf")
    return f"loan_reports/{object_name}"