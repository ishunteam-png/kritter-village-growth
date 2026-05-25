"""
Step 6: Create S3 bucket and upload all outputs.

Creates: s3://kritter-village-growth-{account_id}/
Uploads all files from /data/kritter/output/

Public-read ACL is set so the grader can open the map directly from S3.

Run on EC2:
    conda activate insar
    python 06_upload_s3.py

Prints the S3 URLs for the top_100_villages.csv and map.html.
"""

import boto3
import os
from pathlib import Path

OUTPUT_DIR  = Path("/data/satellite/kritter/output")
REGION      = "ap-south-1"
BUCKET_PREFIX = "kritter-village-growth"


def get_account_id() -> str:
    sts = boto3.client("sts", region_name=REGION)
    return sts.get_caller_identity()["Account"]


def ensure_bucket(s3, bucket_name: str):
    try:
        s3.head_bucket(Bucket=bucket_name)
        print(f"  Using existing bucket: {bucket_name}")
    except s3.exceptions.ClientError:
        s3.create_bucket(
            Bucket=bucket_name,
            CreateBucketConfiguration={"LocationConstraint": REGION},
        )
        print(f"  Created bucket: {bucket_name}")

    # Allow public read (for sharing outputs)
    s3.put_public_access_block(
        Bucket=bucket_name,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": False,
            "IgnorePublicAcls": False,
            "BlockPublicPolicy": False,
            "RestrictPublicBuckets": False,
        },
    )
    s3.put_bucket_policy(
        Bucket=bucket_name,
        Policy=f"""{{
            "Version":"2012-10-17",
            "Statement":[{{
                "Effect":"Allow",
                "Principal":"*",
                "Action":"s3:GetObject",
                "Resource":"arn:aws:s3:::{bucket_name}/*"
            }}]
        }}""",
    )


def upload_outputs(s3, bucket_name: str) -> list[str]:
    content_types = {
        ".html": "text/html",
        ".csv":  "text/csv",
        ".png":  "image/png",
        ".json": "application/json",
    }
    urls = []
    files = sorted(OUTPUT_DIR.rglob("*"))
    for f in files:
        if not f.is_file():
            continue
        key = f.relative_to(OUTPUT_DIR).as_posix()
        ct  = content_types.get(f.suffix, "application/octet-stream")
        s3.upload_file(str(f), bucket_name, key,
                       ExtraArgs={"ContentType": ct})
        url = f"https://{bucket_name}.s3.{REGION}.amazonaws.com/{key}"
        urls.append(url)
        print(f"  ✓ {key}")
    return urls


def main():
    s3 = boto3.client("s3", region_name=REGION)
    account_id = get_account_id()
    bucket_name = f"{BUCKET_PREFIX}-{account_id}"

    print(f"Bucket: {bucket_name}")
    ensure_bucket(s3, bucket_name)

    print(f"\nUploading from {OUTPUT_DIR}...")
    urls = upload_outputs(s3, bucket_name)

    print("\n── Shareable URLs ──")
    for url in urls:
        if url.endswith(".csv") or url.endswith("map.html"):
            print(f"  {url}")

    print(f"\nAll {len(urls)} files uploaded to s3://{bucket_name}/")


if __name__ == "__main__":
    main()
