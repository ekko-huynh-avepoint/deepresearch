import os
import boto3
from typing import Optional
import logging
import shutil
# Get S3 configuration from environment variables
S3_ENDPOINT = os.environ.get("S3_ENDPOINT")
S3_BUCKET = os.environ.get("S3_BUCKET")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY")

logger = logging.getLogger(__name__)


def upload_to_s3(data: bytes, s3_key: str) -> Optional[str]:
    """Upload binary data to S3 and return the URL"""
    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=S3_ENDPOINT,
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_KEY,
        )
        s3.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=data)

        # Ensure the endpoint URL ends with a slash for proper URL formation
        endpoint = S3_ENDPOINT if S3_ENDPOINT.endswith('/') else f"{S3_ENDPOINT}/"
        s3_url = f"{endpoint}/{S3_BUCKET}/{s3_key}"
        logger.info(f"Successfully uploaded to S3: {s3_url}")
        return s3_url

    except Exception as e:
        logger.error(f"Failed to upload to S3: {e}")
        return None


def upload_directory_to_s3(directory_path: str, s3_prefix: str = "") -> dict:
    """Upload all files in a directory to S3 and return a map of filenames to S3 URLs"""
    if not os.path.exists(directory_path):
        logger.error(f"Directory not found: {directory_path}")
        return {"error": f"Directory not found: {directory_path}"}

    result = {}

    try:
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Create an S3 key relative to the directory path
                relative_path = os.path.relpath(file_path, directory_path)
                s3_key = os.path.join(s3_prefix, relative_path).replace("\\", "/")

                # Read file content and upload to S3
                with open(file_path, 'rb') as f:
                    file_data = f.read()
                    s3_url = upload_to_s3(file_data, s3_key)
                    if s3_url:
                        result[relative_path] = s3_url

    except Exception as e:
        logger.error(f"Error uploading directory to S3: {e}")
        return {"error": str(e)}

    return result