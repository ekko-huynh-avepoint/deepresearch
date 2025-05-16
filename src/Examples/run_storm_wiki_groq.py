import os
import re
from argparse import ArgumentParser

from src.knowledge_storm import (
    STORMWikiRunnerArguments,
    STORMWikiRunner,
    STORMWikiLMConfigs, logger,
)
from src.knowledge_storm.lm import GroqModel
from src.knowledge_storm.rm import (
    BraveRM,
    DuckDuckGoSearchRM,
    FirecrawlRM,
    VectorRM,
)
from src.knowledge_storm.utils import QdrantVectorStoreManager  # <<-- Add QdrantVectorStoreManager
from qdrant_client import QdrantClient  # <<-- Add QdrantClient
from langchain_ollama import OllamaEmbeddings  # <<-- Add OllamaEmbeddings
from dotenv import load_dotenv

load_dotenv()


def sanitize_topic(topic):
    topic = topic.replace(" ", "_")
    topic = re.sub(r"[^a-zA-Z0-9_-]", "", topic)
    if not topic:
        topic = "unnamed_topic"
    return topic

def setup_vectorrm(args):
    embedding_model = os.getenv("EMB_MODEL")
    ollama_base_url = os.getenv("OLLAMA_EMB", "http://localhost:11434")
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")

    emb_model = OllamaEmbeddings(model=embedding_model, base_url=ollama_base_url)
    client = QdrantClient(url=qdrant_url)
    qdrant_manager = QdrantVectorStoreManager(
        client=client,
        collection_name=args.collection_name,
        model=emb_model,
    )

    if getattr(args, "csv_file_path", None):
        qdrant_manager.create_or_update_vector_store(
            file_path=args.csv_file_path,
            content_column="content",
            title_column="title",
            url_column="url",
            desc_column="description",
            batch_size=64
        )

    return VectorRM(
        collection_name=args.collection_name,
        qdrant_vs=qdrant_manager.get_qdrant(),
        embed_model=emb_model,
        k=getattr(args, "search_top_k", 3),
    )


def main(args):
    lm_configs = STORMWikiLMConfigs()

    if not os.environ.get("GROQ_API_KEY"):
        raise ValueError(
            "GROQ_API_KEY environment variable is not set. Please set it in your secrets.toml file."
        )

    groq_kwargs = {
        "api_key": os.environ.get("GROQ_API_KEY"),
        "api_base": "https://api.groq.com/openai/v1",
        "temperature": args.temperature,
        "top_p": args.top_p,
    }

    conv_simulator_lm = GroqModel(
        model="gemma2-9b-it", max_tokens=500, **groq_kwargs
    )
    question_asker_lm = GroqModel(
        model="gemma2-9b-it", max_tokens=500, **groq_kwargs
    )
    outline_gen_lm = GroqModel(model="gemma2-9b-it", max_tokens=400, **groq_kwargs)
    article_gen_lm = GroqModel(model="gemma2-9b-it", max_tokens=700, **groq_kwargs)
    article_polish_lm = GroqModel(
        model="gemma2-9b-it", max_tokens=4000, **groq_kwargs
    )

    lm_configs.set_conv_simulator_lm(conv_simulator_lm)
    lm_configs.set_question_asker_lm(question_asker_lm)
    lm_configs.set_outline_gen_lm(outline_gen_lm)
    lm_configs.set_article_gen_lm(article_gen_lm)
    lm_configs.set_article_polish_lm(article_polish_lm)

    engine_args = STORMWikiRunnerArguments(
        output_dir=args.output_dir,
        max_conv_turn=args.max_conv_turn,
        max_perspective=args.max_perspective,
        search_top_k=args.search_top_k,
        max_thread_num=args.max_thread_num,
    )

    # Use VectorRM or Web retrievers
    match args.retriever:
        case "vector":
            rm = setup_vectorrm(args)
        case "brave":
            rm = BraveRM(
                brave_search_api_key=os.environ.get("BRAVE_API_KEY"),
                k=engine_args.search_top_k,
            )
        case "tavily":
            rm = TavilySearchRM(
                tavily_search_api_key=os.environ.get("TAVILY_API_KEY"),
                k=engine_args.search_top_k,
                include_raw_content=True,
            )
        case "duckduckgo":
            rm = DuckDuckGoSearchRM(
                k=engine_args.search_top_k, safe_search="On", region="us-en"
            )
        case "firecrawl":
            rm = FirecrawlRM(
                search_link=os.environ.get("LINK_SEARCH"),
                k=engine_args.search_top_k,
            )
        case _:
            raise ValueError(
                f'Invalid retriever: {args.retriever}. "'
            )
    rm = FirecrawlRM(
        search_link=os.environ.get("LINK_SEARCH"),
        k=engine_args.search_top_k,
    )


    runner = STORMWikiRunner(engine_args, lm_configs, rm)

    topic = input("Topic: ")
    sanitized_topic = sanitize_topic(topic)

    try:
        runner.run(
            topic=sanitized_topic,
            do_research=True,
            do_generate_outline=True,
            do_generate_article=True,
            do_polish_article=True,
            remove_duplicate=args.remove_duplicate,
        )
        runner.post_run()
        runner.summary()
    except Exception as e:
        logger.exception(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="./results/groq", help="Directory to store the outputs.")
    parser.add_argument("--max-thread-num", type=int, default=3, help="Maximum number of threads to use.")
    parser.add_argument("--retriever", type=str, default="tavily",
                        choices=["vector", "brave", "tavily", "duckduckgo", "firecrawl"],
                        help="The search engine API to use for retrieving information.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature to use.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter.")
    parser.add_argument("--do-research", action="store_true", help="If True, simulate conversation to research the topic.")
    parser.add_argument("--do-generate-outline", action="store_true", help="If True, generate an outline for the topic.")
    parser.add_argument("--do-generate-article", action="store_true", help="If True, generate an article for the topic.")
    parser.add_argument("--do-polish-article", action="store_true", help="If True, polish the article by adding a summarization section and (optionally) removing duplicate content.")
    parser.add_argument("--max-conv-turn", type=int, default=3, help="Maximum number of questions in conversational question asking.")
    parser.add_argument("--max-perspective", type=int, default=3, help="Maximum number of perspectives to consider in perspective-guided question asking.")
    parser.add_argument("--search-top-k", type=int, default=3, help="Top k search results to consider for each search query.")
    parser.add_argument("--retrieve-top-k", type=int, default=3, help="Top k collected references for each section title.")
    parser.add_argument("--remove-duplicate", action="store_true", help="If True, remove duplicate content from the article.")
    # --- VectorRM specific args ---
    parser.add_argument("--collection-name", type=str, default="ekko_vectordb", help="Qdrant collection name.")
    parser.add_argument("--csv-file-path", type=str, default=None, help="Path to CSV for vector store population.")

    main(parser.parse_args())