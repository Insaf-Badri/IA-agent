# Replace your local storage.py with the new OBS version
import os
import tempfile
import boto3
from pathlib import Path
from botocore.exceptions import ClientError

BUCKETS = ["loan-documents", "loan-reports"]
DOWNLOAD_DIR = Path(tempfile.gettempdir()) / "loan_agent_docs"
DOWNLOAD_DIR.mkdir(exist_ok=True)

def get_obs():
    endpoint = os.getenv("HW_OBS_ENDPOINT", "https://obs.ap-southeast-1.myhuaweicloud.com")
    print(f"🔌 Connecting to OBS at: {endpoint}")
    return boto3.client(
        's3',
        endpoint_url=endpoint,
        aws_access_key_id=os.getenv("HW_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("HW_SECRET_KEY"),
        region_name='ap-southeast-1'
    )

# Backward compatibility alias
get_minio = get_obs

def init_buckets():
    client = get_obs()
    for bucket in BUCKETS:
        try:
            client.head_bucket(Bucket=bucket)
            print(f"✅ Bucket exists: {bucket}")
        except ClientError:
            client.create_bucket(Bucket=bucket)
            print(f"✅ Created bucket: {bucket}")

def download_pdf(case_id: str, doc_type: str) -> str | None:
    client      = get_obs()
    object_name = f"{case_id}/{doc_type}.pdf"
    local_path  = DOWNLOAD_DIR / f"{case_id}_{doc_type}.pdf"
    try:
        client.download_file(
            os.getenv("HW_OBS_BUCKET", "loan-documents"),
            object_name,
            str(local_path)
        )
        return str(local_path)
    except ClientError as e:
        print(f"  ✗ OBS download failed for {object_name}: {e}")
        return None

def upload_report(case_id: str, local_path: str) -> str:
    client      = get_obs()
    object_name = f"{case_id}_report.pdf"
    client.upload_file(
        local_path,
        "loan-reports",
        object_name,
        ExtraArgs={'ContentType': 'application/pdf'}
    )
    return f"loan-reports/{object_name}"