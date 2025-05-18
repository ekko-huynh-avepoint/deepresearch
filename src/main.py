import os
import uuid
import hashlib
import logging
import datetime
import boto3
import sys
import json
import time
from mcp.server.fastmcp import FastMCP
from src.Service.research_manager import ResearchManager
from src.knowledge_storm.reports import generate_research_report
from typing import Optional, List, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("research_server.log")
    ]
)
logger = logging.getLogger(__name__)

# Server configuration
SERVER_NAME = "ResearchServer"
SERVER_HOST = os.environ.get("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("SERVER_PORT", 3555))
IP_ADDRESS = os.environ.get("IP_ADDRESS", "127.0.0.1")

# S3 configuration
S3_ENDPOINT = os.environ.get("S3_ENDPOINT_URL")
S3_BUCKET = os.environ.get("S3_BUCKET", "mcp")
S3_ACCESS_KEY_ID = os.environ.get("S3_ACCESS_KEY_ID")
S3_SECRET_ACCESS_KEY = os.environ.get("S3_SECRET_ACCESS_KEY")

# Important paths - Use absolute paths with container environment in mind
# Base directory should be /app in container environment
BASE_DIR = "/app"
if not os.path.exists(BASE_DIR):
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    logger.info(f"Base directory /app not found, using current directory: {BASE_DIR}")

DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "src", "results")

# Ensure directories exist
for directory in [DATA_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)
    logger.info(f"Ensured directory exists: {directory}")

# Initialize server
mcp = FastMCP(name=SERVER_NAME, host=SERVER_HOST, port=SERVER_PORT)

# Initialize research manager
research_manager = ResearchManager()

# Log configuration details at startup
logger.info(f"Server initialized: {SERVER_NAME} on {SERVER_HOST}:{SERVER_PORT}")
logger.info(f"Environment: BASE_DIR={BASE_DIR}, RESULTS_DIR={RESULTS_DIR}")
logger.info(f"S3 Configuration: Endpoint={S3_ENDPOINT}, Bucket={S3_BUCKET}")
logger.info(f"S3 Credentials Available: Access Key={'✓' if S3_ACCESS_KEY_ID else '✗'}, "
            f"Secret Key={'✓' if S3_SECRET_ACCESS_KEY else '✗'}")


def generate_random_hash(length=8):
    """Generate a random hash for unique file naming"""
    random_uuid = uuid.uuid4()
    hash_object = hashlib.md5(str(random_uuid).encode())
    return hash_object.hexdigest()[:length]


def verify_research_files(topic_dir: str) -> Dict[str, bool]:
    """Verify research files exist in the specified directory"""
    required_files = [
        "raw_search_results.json",
        "url_to_info.json",
        "storm_gen_outline.txt",
        "direct_gen_outline.txt",
        "storm_gen_article.txt",
        "storm_gen_article_polished.txt"
    ]

    results = {}
    for file in required_files:
        file_path = os.path.join(topic_dir, file)
        exists = os.path.exists(file_path)
        results[file] = exists
        logger.info(f"Research file check: {file_path} - {'FOUND' if exists else 'NOT FOUND'}")

    return results


def remove_directory_contents(directory):
    """Remove all files and subdirectories in a directory"""
    if not os.path.exists(directory):
        logger.warning(f"Directory does not exist, cannot clean: {directory}")
        return

    logger.info(f"Cleaning directory: {directory}")
    try:
        for root, dirs, files in os.walk(directory, topdown=False):
            # First remove all files in current directory
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    logger.debug(f"Removed file: {file_path}")
                except OSError as e:
                    logger.warning(f"Failed to remove file {file_path}: {e}")

            # Then remove all empty directories
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                try:
                    os.rmdir(dir_path)
                    logger.debug(f"Removed directory: {dir_path}")
                except OSError as e:
                    logger.warning(f"Failed to remove directory {dir_path}: {e}")
    except Exception as e:
        logger.error(f"Error cleaning directory {directory}: {e}", exc_info=True)


def reliable_upload_to_s3(data: bytes, s3_key: str) -> Optional[str]:
    """Upload file to S3 with enhanced error handling and retries"""
    if not all([S3_ENDPOINT, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY, S3_BUCKET]):
        logger.error(f"S3 configuration is incomplete. Cannot upload to S3.")
        return None

    logger.info(f"Attempting S3 upload with key: {s3_key}, file size: {len(data)} bytes")

    # Try up to 3 times
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            s3 = boto3.client(
                "s3",
                endpoint_url=S3_ENDPOINT,
                aws_access_key_id=S3_ACCESS_KEY_ID,
                aws_secret_access_key=S3_SECRET_ACCESS_KEY,
            )

            # Upload the file
            s3.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=data)

            # Construct the S3 URL
            if S3_ENDPOINT.endswith('/'):
                endpoint = S3_ENDPOINT
            else:
                endpoint = f"{S3_ENDPOINT}/"

            s3_url = f"{endpoint}{S3_BUCKET}/{s3_key}"
            logger.info(f"Successfully uploaded to S3: {s3_url}")
            return s3_url

        except Exception as e:
            logger.warning(f"S3 upload attempt {attempt} failed: {e}")
            if attempt == max_retries:
                logger.error(f"Failed to upload to S3 after {max_retries} attempts", exc_info=True)
                return None


def create_fallback_research_files(topic_dir: str, topic: str):
    """Create fallback research files if the original research process fails"""
    logger.info(f"Creating fallback research files in {topic_dir}")

    # Ensure the directory exists
    os.makedirs(topic_dir, exist_ok=True)

    # Create minimal versions of required files
    files_to_create = {
        "raw_search_results.json": json.dumps({"urls": []}),
        "url_to_info.json": json.dumps({}),
        "storm_gen_outline.txt": f"# {topic} Research Outline\n\n1. Introduction\n2. Summary\n3. Conclusion",
        "direct_gen_outline.txt": f"# {topic} Research Outline\n\n1. Introduction\n2. Summary\n3. Conclusion",
        "storm_gen_article.txt": f"# {topic} Research\n\nThis is a placeholder article for {topic}.",
        "storm_gen_article_polished.txt": f"# {topic} Research\n\nThis is a placeholder article for {topic}."
    }

    for filename, content in files_to_create.items():
        file_path = os.path.join(topic_dir, filename)
        with open(file_path, 'w') as f:
            f.write(content)
        logger.info(f"Created fallback file: {file_path}")


@mcp.tool()
async def deep_research(query: str):
    """Perform deep research on a given query"""
    logger.info(f"Starting deep research on: {query}")

    # Define directories using absolute paths
    topic_name = query.replace(" ", "_")
    groq_dir = os.path.join(RESULTS_DIR, "groq")
    topic_dir = os.path.join(groq_dir, topic_name)

    # Create directories with verbose logging
    for dir_path in [RESULTS_DIR, groq_dir, topic_dir]:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Ensured directory exists: {dir_path}")

    logger.info(f"Directory structure: RESULTS_DIR={RESULTS_DIR}, groq_dir={groq_dir}, topic_dir={topic_dir}")

    # List any existing files in the directory
    if os.path.exists(topic_dir) and os.listdir(topic_dir):
        logger.info(f"Existing files in {topic_dir}: {os.listdir(topic_dir)}")

    pdf_path = None
    download_url = None
    research_successful = False

    try:
        # Set research topic
        research_manager.set_topic(query)
        logger.info(f"Research topic set to: {query}")

        # Run research with absolute output directory path
        logger.info(f"Running Groq research with output_dir: {RESULTS_DIR}")
        try:
            research_manager.run_groq(output_dir=RESULTS_DIR)
            logger.info(f"Research completed, checking for output files")

            # Verify files exist
            time.sleep(1)  # Small delay to ensure file system updates
            file_check = verify_research_files(topic_dir)
            research_successful = all(file_check.values())

            if not research_successful:
                logger.warning(f"Research process didn't generate all required files: {file_check}")
                # Create fallback files
                create_fallback_research_files(topic_dir, query)
                logger.info("Created fallback research files")

        except Exception as e:
            logger.error(f"Error during Groq research: {e}", exc_info=True)
            # Create fallback files
            create_fallback_research_files(topic_dir, query)
            logger.info("Created fallback research files after research error")

        # Generate report
        logger.info(f"Generating PDF report for {topic_name} in directory {topic_dir}")
        try:
            pdf_path = generate_research_report(topic_dir, topic_name)
            logger.info(f"PDF path returned: {pdf_path}")
        except Exception as e:
            logger.error(f"Error in generate_research_report: {e}", exc_info=True)
            return {
                "status": "error",
                "summary": f"Failed to generate PDF report: {e}"
            }

        if not pdf_path:
            logger.error("PDF path is None - report generation failed")
            return {
                "status": "error",
                "summary": "PDF report generation failed - no path returned"
            }

        # Verify PDF exists
        if not os.path.exists(pdf_path):
            logger.error(f"Generated PDF not found at path: {pdf_path}")

            # List directory contents to help debug
            parent_dir = os.path.dirname(pdf_path)
            if os.path.exists(parent_dir):
                files = os.listdir(parent_dir)
                logger.info(f"Files in {parent_dir}: {files}")
            else:
                logger.error(f"Parent directory {parent_dir} does not exist")

            return {
                "status": "error",
                "summary": f"PDF report was not found at expected path: {pdf_path}"
            }

        # PDF exists, get info
        file_size = os.path.getsize(pdf_path)
        logger.info(f"PDF report generated successfully: {pdf_path}, size: {file_size} bytes")

        # Create unique filename
        unique_hash = generate_random_hash()
        date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")

        pdf_filename = os.path.basename(pdf_path)
        filename_parts = os.path.splitext(pdf_filename)
        unique_pdf_name = f"{filename_parts[0]}_{date}_{unique_hash}{filename_parts[1]}"

        s3_key = f"research_reports/{topic_name}/{unique_pdf_name}".replace("\\", "/")

        # Upload to S3
        try:
            with open(pdf_path, 'rb') as pdf_file:
                pdf_data = pdf_file.read()
                if not pdf_data or len(pdf_data) == 0:
                    logger.error(f"PDF file is empty: {pdf_path}")
                    return {
                        "status": "error",
                        "summary": f"Generated PDF file is empty: {pdf_path}"
                    }

                logger.info(f"PDF file read successfully, size: {len(pdf_data)} bytes")

                download_url = reliable_upload_to_s3(pdf_data, s3_key)

                if not download_url:
                    logger.error("Failed to upload PDF to S3, no download URL returned")
                    return {
                        "status": "error",
                        "summary": "Failed to upload PDF to S3 service"
                    }
        except Exception as e:
            logger.error(f"Error reading or uploading PDF: {e}", exc_info=True)
            return {
                "status": "error",
                "summary": f"Error processing PDF file: {e}"
            }

        # Success response with warning if fallback files were used
        status_prefix = "" if research_successful else "⚠️ NOTE: Research process used fallback files. "
        return {
            "status": "success",
            "summary": (
                f"✅ {status_prefix}Research completed.\n"
                f"PDF report generated: {unique_pdf_name}\n"
                f"Download link: {download_url}\n"
            ),
            "pdf_download_url": download_url,
            "used_fallback": not research_successful
        }
    except Exception as err:
        logger.error(f"Error in deep_research: {err}", exc_info=True)
        return {
            "status": "error",
            "summary": f"Research process error: {err}",
        }
    finally:
        # Don't clean up right away - keep files for debugging
        # We'll let the next run clean them up
        logger.info(f"Keeping research documents in {topic_dir} for debugging")


if __name__ == "__main__":
    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Starting {SERVER_NAME} at {start_time}")
    logger.info(f"Current user: ekko-huynh-avepoint")

    try:
        mcp.run(transport="sse")
    except KeyboardInterrupt:
        logger.info(f"\n{SERVER_NAME} stopped by user.")
    except Exception as e:
        logger.error(f"{SERVER_NAME} exited with error: {e}", exc_info=True)
    finally:
        end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"{SERVER_NAME} has shut down at {end_time}.")