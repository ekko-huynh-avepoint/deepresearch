# import os
# import uuid
# import hashlib
# import logging
# import datetime
# import asyncio
# import boto3
# from src.Service.research_manager import ResearchManager
# from src.knowledge_storm.reports import generate_research_report
# from typing import Optional
# from dotenv import load_dotenv
#
# load_dotenv()
#
# # Setup logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
#
# S3_ENDPOINT = os.environ.get("S3_ENDPOINT_URL", "https://bucket-production-eb7a.up.railway.app:443")
# S3_BUCKET = os.environ.get("S3_BUCKET", "mcp")
# S3_ACCESS_KEY_ID = os.environ.get("S3_ACCESS_KEY_ID", "NBI3QwaI3kVyoY0j09LISiUzlqwjzc1H")
# S3_SECRET_ACCESS_KEY = os.environ.get("S3_SECRET_ACCESS_KEY", "fO6H2A5A8PicMfa3YolUu6jlS1xY6iiyejkzYRS8uIKtyIWO")
#
# DATA_DIR = "data"
# os.makedirs(DATA_DIR, exist_ok=True)
#
# research_manager = ResearchManager()
#
#
# def generate_random_hash(length=8):
#     """Generate a random hash for unique file naming"""
#     random_uuid = uuid.uuid4()
#     hash_object = hashlib.md5(str(random_uuid).encode())
#     return hash_object.hexdigest()[:length]
#
#
# def normalize_path(path):
#     """Normalize path to use forward slashes consistently across platforms"""
#     return path.replace('\\', '/')
#
#
# def remove_directory_contents(directory):
#     """Remove all files and subdirectories in a directory using only os module"""
#     normalized_dir = normalize_path(directory)
#     for root, dirs, files in os.walk(normalized_dir, topdown=False):
#         # First remove all files in current directory
#         for file in files:
#             file_path = normalize_path(os.path.join(root, file))
#             try:
#                 os.remove(file_path)
#             except OSError as e:
#                 logger.warning(f"Error removing file {file_path}: {e}")
#
#         # Then remove all empty directories
#         for dir in dirs:
#             dir_path = normalize_path(os.path.join(root, dir))
#             try:
#                 os.rmdir(dir_path)
#             except OSError as e:
#                 logger.warning(f"Error removing directory {dir_path}: {e}")
#
#     # Try to remove the main directory itself
#     try:
#         os.rmdir(normalized_dir)
#     except OSError:
#         logger.warning(f"Could not completely remove directory: {normalized_dir}")
#
#
# def reliable_upload_to_s3(data: bytes, s3_key: str) -> Optional[str]:
#     try:
#         s3 = boto3.client(
#             "s3",
#             endpoint_url=S3_ENDPOINT,
#             aws_access_key_id=S3_ACCESS_KEY_ID,
#             aws_secret_access_key=S3_SECRET_ACCESS_KEY,
#         )
#
#         s3.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=data)
#
#         if S3_ENDPOINT.endswith('/'):
#             endpoint = S3_ENDPOINT
#         else:
#             endpoint = f"{S3_ENDPOINT}/"
#
#         # Construct the S3 URL
#         s3_url = f"{endpoint}{S3_BUCKET}/{s3_key}"
#         logger.info(f"Successfully uploaded to S3: {s3_url}")
#         return s3_url
#
#     except Exception as e:
#         logger.error(f"Failed to upload to S3: {e}", exc_info=True)
#         return None
#
#
# async def deep_research(query: str):
#     # Use normalized paths - this is critical for cross-platform compatibility
#     output_dir = normalize_path("./results/")
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir, exist_ok=True)
#         logger.info(f"Created output directory: {output_dir}")
#
#     research_manager.set_topic(query)
#
#     topic_name = query.replace(" ", "_")
#
#     # Skip the groq subdirectory - this was causing problems
#     results_dir = normalize_path(os.path.join(output_dir, topic_name))
#
#     # Ensure results directory exists
#     if not os.path.exists(results_dir):
#         os.makedirs(results_dir, exist_ok=True)
#         logger.info(f"Created results directory: {results_dir}")
#
#     pdf_path = None
#     download_url = None
#
#     try:
#         # Run research directly without groq subdirectory
#         research_manager.run_groq(output_dir=normalize_path(output_dir))
#         logger.info(f"Generating PDF report for {topic_name}")
#
#         # Explicitly normalize the path for the PDF generation
#         pdf_path = generate_research_report(normalize_path(results_dir), topic_name)
#         pdf_path = normalize_path(pdf_path)  # Ensure path has forward slashes
#
#         if not pdf_path or not os.path.exists(pdf_path):
#             raise FileNotFoundError(f"PDF report was not generated: {pdf_path}")
#
#         logger.info(f"PDF report generated successfully: {pdf_path}")
#
#         unique_hash = generate_random_hash()
#         date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
#
#         pdf_filename = os.path.basename(pdf_path)
#         filename_parts = os.path.splitext(pdf_filename)
#         unique_pdf_name = f"{filename_parts[0]}_{date}_{unique_hash}{filename_parts[1]}"
#
#         # Ensure forward slashes in S3 key
#         s3_key = f"research_reports/{topic_name}/{unique_pdf_name}"
#
#         with open(pdf_path, 'rb') as pdf_file:
#             pdf_data = pdf_file.read()
#             if not pdf_data:
#                 raise ValueError("PDF file is empty")
#
#             logger.info(f"PDF file read successfully, size: {len(pdf_data)} bytes")
#
#             download_url = reliable_upload_to_s3(pdf_data, s3_key)
#
#             if not download_url:
#                 raise Exception("Failed to upload PDF to S3, no download URL returned")
#
#         logger.info(f"Cleaning up research documents in {results_dir}")
#         if os.path.exists(results_dir):
#             remove_directory_contents(results_dir)
#             logger.info(f"Research documents cleanup completed")
#
#         return {
#             "status": "success",
#             "summary": (
#                 f"✅ Research completed using backend.\n"
#                 f"PDF report generated: {unique_pdf_name}\n"
#                 f"Download link: {download_url}\n"
#             ),
#             "pdf_download_url": download_url
#         }
#     except Exception as err:
#         logger.error(f"Error in deep_research: {err}", exc_info=True)
#         return {
#             "status": "error",
#             "summary": f"AI self-heal: Encountered error during deep research: {err}",
#         }
#
#
#
# async def main():
#     # Test deep_research with a sample query
#     query = "artificial intelligence trends"
#
#     logger.info(f"Current Date/Time (UTC): {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#     logger.info(f"Current user: ekko-huynh-avepoint")
#     logger.info(f"Starting deep research test with query: '{query}'")
#
#     try:
#         result = await deep_research(query)
#         logger.info(f"Research result: {result}")
#     except Exception as e:
#         logger.error(f"Test failed with error: {e}", exc_info=True)
#
#
# if __name__ == "__main__":
#     logger.info("Starting deep research test")
#     asyncio.run(main())