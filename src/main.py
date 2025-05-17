import os
import uuid
import hashlib
import logging
import datetime
from mcp.server.fastmcp import FastMCP
from src.Service.research_manager import ResearchManager
from src.Service.s3_tool import upload_to_s3
from src.knowledge_storm.reports import generate_research_report

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SERVER_NAME = "ResearchServer"
SERVER_HOST = os.environ.get("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("SERVER_PORT", 3555))
IP_ADDRESS = os.environ.get("IP_ADDRESS", "127.0.0.1")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

mcp = FastMCP(name=SERVER_NAME, host=SERVER_HOST, port=SERVER_PORT)

research_manager = ResearchManager()


def generate_random_hash(length=8):
    random_uuid = uuid.uuid4()
    hash_object = hashlib.md5(str(random_uuid).encode())
    return hash_object.hexdigest()[:length]


def remove_directory_contents(directory):
    """Remove all files and subdirectories in a directory using only os module"""
    for root, dirs, files in os.walk(directory, topdown=False):
        # First remove all files in current directory
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)
        # Then remove all empty directories
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            try:
                os.rmdir(dir_path)
            except OSError:
                # Directory might not be empty yet
                pass
    # Try to remove the main directory itself
    try:
        os.rmdir(directory)
    except OSError:
        logger.warning(f"Could not completely remove directory: {directory}")


@mcp.tool()
async def deep_research(
        query: str,
        output_dir: str = None
):
    if output_dir is None:
        output_dir = f"./results/"

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")

    research_manager.set_topic(query)

    # Clean topic name for directory naming
    topic_name = query.replace(" ", "_")
    results_dir = os.path.join(output_dir, "groq", topic_name)

    # Ensure results directory exists
    groq_dir = os.path.join(output_dir, "groq")
    if not os.path.exists(groq_dir):
        os.makedirs(groq_dir, exist_ok=True)
        logger.info(f"Created groq directory: {groq_dir}")

    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
        logger.info(f"Created results directory: {results_dir}")

    pdf_path = None
    download_url = None

    try:
        # Run research using Groq
        research_manager.run_groq(output_dir=output_dir)

        # Generate PDF report
        logger.info(f"Generating PDF report for {topic_name}")
        pdf_path = generate_research_report(results_dir, topic_name)

        # Generate unique hash and get current time
        unique_hash = generate_random_hash()

        date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")

        # Upload PDF to S3 for download link with unique hash and timestamp
        pdf_filename = os.path.basename(pdf_path)
        filename_parts = os.path.splitext(pdf_filename)
        unique_pdf_name = f"{filename_parts[0]}_{date}_{unique_hash}{filename_parts[1]}"

        s3_key = f"research_reports/{topic_name}/{unique_pdf_name}"

        with open(pdf_path, 'rb') as pdf_file:
            pdf_data = pdf_file.read()
            download_url = upload_to_s3(pdf_data, s3_key)

        return {
            "status": "success",
            "summary": (
                f"✅ Research completed using backend.\n"
                f"PDF report generated: {unique_pdf_name}\n"
                f"Download link: {download_url}\n"
            ),
            "output_dir": output_dir,
            "query": query,
            "pdf_download_url": download_url
        }
    except Exception as err:
        logger.error(f"Error in deep_research: {err}", exc_info=True)
        return {
            "status": "error",
            "summary": f"AI self-heal: Encountered error during deep research: {err}",
        }
    finally:
        # Clean up the research documents regardless of success or failure
        logger.info(f"Cleaning up research documents in {results_dir}")
        if os.path.exists(results_dir):
            remove_directory_contents(results_dir)
            logger.info(f"Research documents cleanup completed")


if __name__ == "__main__":
    logger.info(f"Starting {SERVER_NAME} on {SERVER_HOST}:{SERVER_PORT}")
    logger.info(f"Current user: ekko-huynh-avepoint")

    try:
        mcp.run(transport="sse")
    except KeyboardInterrupt:
        logger.info(f"\n{SERVER_NAME} stopped by user.")
    except Exception as e:
        logger.error(f"{SERVER_NAME} exited with error: {e}", exc_info=True)
    finally:
        logger.info(f"{SERVER_NAME} has shut down.")