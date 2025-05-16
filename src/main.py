import os
import logging
from datetime import datetime
from mcp.server.fastmcp import FastMCP
from src.Service.research_manager import ResearchManager

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

@mcp.tool()
async def deep_research(
    query: str,
    model: str = "groq",
    output_dir: str = None
):
    try:
        if output_dir is None:
            output_dir = f"./results/{model}"

        research_manager.set_topic(query)

        research_manager.run_groq(output_dir=output_dir)

        # if model == "gpt":
        #     research_manager.run_gpt(output_dir=output_dir)
        # elif model == "groq":
        #     research_manager.run_groq(output_dir=output_dir)
        # elif model == "ollama":
        #     research_manager.run_ollama(output_dir=output_dir)
        # else:
        #     return {
        #         "status": "error",
        #         "summary": f"Unknown model: {model}. Choose from 'gpt', 'groq', 'ollama'."
        #     }

        return {
            "status": "success",
            "summary": (
                f"✅ Research completed using '{model}' backend.\n"
                f"Results saved in: {output_dir}/\n"
            ),
            "output_dir": output_dir,
            "model": model,
            "query": query
        }
    except Exception as err:
        logger.error(f"Error in deep_research: {err}", exc_info=True)
        return {
            "status": "error",
            "summary": f"AI self-heal: Encountered error during deep research: {err}",
        }


if __name__ == "__main__":
    logger.info(f"Starting {SERVER_NAME} on {SERVER_HOST}:{SERVER_PORT}")
    logger.info(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Current user: ekko-huynh-avepoint")

    try:
        mcp.run(transport="sse")
    except KeyboardInterrupt:
        logger.info(f"\n{SERVER_NAME} stopped by user.")
    except Exception as e:
        logger.error(f"{SERVER_NAME} exited with error: {e}", exc_info=True)
    finally:
        logger.info(f"{SERVER_NAME} has shut down.")