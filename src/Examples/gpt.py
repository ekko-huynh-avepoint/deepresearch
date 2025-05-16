import os
import json
from argparse import ArgumentParser, Namespace
from dotenv import load_dotenv
from typing import Optional, Any, List

from src.knowledge_storm.lm import OpenAIModel, GroqModel, OllamaClient
from src.knowledge_storm.logging_wrapper import LoggingWrapper
from src.knowledge_storm.collaborative_storm.engine import (
    CollaborativeStormLMConfigs,
    RunnerArgument,
    CoStormRunner,
)
from src.knowledge_storm.collaborative_storm.modules.callback import (
    LocalConsolePrintCallBackHandler,
)
from src.knowledge_storm.rm import (
    BraveRM, DuckDuckGoSearchRM, TavilySearchRM, VectorRM, FirecrawlRM
)
from src.knowledge_storm.utils import QdrantVectorStoreManager
from qdrant_client import QdrantClient
from langchain_ollama import OllamaEmbeddings

load_dotenv()

def get_lm_configs(args: Optional[Namespace] = None, provider: str = "openai") -> CollaborativeStormLMConfigs:
    """
    Factory for creating language model configs for OpenAI, Ollama, or Groq.
    """
    if provider == "openai":
        openai_kwargs = {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "api_provider": "openai",
            "temperature": 1.0,
            "top_p": 0.9,
            "api_base": None,
        }
        model_name = "gpt-4o-mini"
        lm = lambda max_tokens: OpenAIModel(model=model_name, max_tokens=max_tokens, **openai_kwargs)
        config = CollaborativeStormLMConfigs()
        config.set_question_answering_lm(lm(1000))
        config.set_discourse_manage_lm(lm(500))
        config.set_utterance_polishing_lm(lm(2000))
        config.set_warmstart_outline_gen_lm(lm(500))
        config.set_question_asking_lm(lm(300))
        config.set_knowledge_base_lm(lm(1000))
        return config

    elif provider == "ollama":
        ollama_kwargs = {
            "model": getattr(args, "model", "llama3"),
            "port": getattr(args, "port", 11434),
            "url": getattr(args, "url", "http://localhost:11434"),
            "stop": ("\n\n---",),
        }
        config = CollaborativeStormLMConfigs()
        return config

    elif provider == "groq":
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            raise RuntimeError("GROQ_API_KEY environment variable is not set.")
        groq_kwargs = {
            "api_key": groq_api_key,
            "api_base": "https://api.groq.com/openai/v1",
            "temperature": getattr(args, "temperature", 1.0) if args else 1.0,
            "top_p": getattr(args, "top_p", 0.9) if args else 0.9,
        }
        model_name = "meta-llama/llama-4-scout-17b-16e-instruct"
        lm = lambda max_tokens: GroqModel(model=model_name, max_tokens=max_tokens, **groq_kwargs)
        config = CollaborativeStormLMConfigs()
        return config

    else:
        raise ValueError(f"Unknown provider: {provider}")

# ----------------------- Retriever Setup -----------------------

def setup_vectorrm(args: Namespace) -> VectorRM:
    embedding_model = os.getenv("EMB_MODEL", "all-minilm")
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
        k=getattr(args, "retrieve_top_k", 3),
    )

def get_web_rm(args: Namespace) -> Any:
    k = getattr(args, "retrieve_top_k", 3)
    if args.web_source == "brave":
        return BraveRM(brave_search_api_key=os.environ.get("BRAVE_API_KEY"), k=k)
    elif args.web_source == "duckduckgo":
        return DuckDuckGoSearchRM(k=k, safe_search="On", region="us-en")
    elif args.web_source == "tavily":
        return TavilySearchRM(
            tavily_search_api_key=os.environ.get("TAVILY_API_KEY"),
            k=k,
            include_raw_content=True,
        )
    elif args.web_source == "firecrawl":
        return FirecrawlRM(
            search_link=os.environ.get("LINK_SEARCH"),
            k=k,
        )
    else:
        raise ValueError(f"Unknown web source: {args.web_source}")

class HybridRetriever:
    def __init__(self, vector_rm, web_rm, k=5):
        self.vector_rm = vector_rm
        self.web_rm = web_rm
        self.k = k

    def retrieve(self, query: str) -> List[Any]:
        results_vector = self.vector_rm.retrieve(query)
        results_web = self.web_rm.retrieve(query)
        # TODO: Optionally deduplicate/merge by score/url/etc.
        return (results_vector + results_web)[:self.k]

def get_retriever(args: Namespace) -> Any:
    if args.retriever == "vector":
        return setup_vectorrm(args)
    elif args.retriever == "web":
        return get_web_rm(args)
    elif args.retriever == "hybrid":
        return HybridRetriever(setup_vectorrm(args), get_web_rm(args), k=args.retrieve_top_k)
    else:
        raise ValueError(f"Unknown retriever mode: {args.retriever}")

# ----------------------- Main Workflow -----------------------

def main(args: Namespace):
    lm_config = get_lm_configs(args, provider="openai")
    retriever = get_retriever(args)
    topic = input("Topic: ").strip()
    runner_argument = RunnerArgument(
        topic=topic,
        retrieve_top_k=10,
        max_search_queries=2,
        total_conv_turn=20,
        max_search_thread=5,
        max_search_queries_per_turn=3,
        warmstart_max_num_experts=3,
        warmstart_max_turn_per_experts=2,
        warmstart_max_thread=3,
        max_thread_num=10,
        max_num_round_table_experts=2,
        moderator_override_N_consecutive_answering_turn=3,
        node_expansion_trigger_count=10,
    )
    logging_wrapper = LoggingWrapper(lm_config)
    callback_handler = LocalConsolePrintCallBackHandler() if args.enable_log_print else None

    cr = CoStormRunner(
        lm_config=lm_config,
        runner_argument=runner_argument,
        logging_wrapper=logging_wrapper,
        rm=retriever,
        callback_handler=callback_handler,
    )



    cr.warm_start()
    for _ in range(1):
        conv_turn = cr.step()
        print(f"**{conv_turn.role}**: {conv_turn.utterance}\n")
    your_utterance = input("Your utterance: ").strip()
    cr.step(user_utterance=your_utterance)
    conv_turn = cr.step()
    print(f"**{conv_turn.role}**: {conv_turn.utterance}\n")

    cr.knowledge_base.reogranize()
    article = cr.generate_report()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "report.md"), "w") as f:
        f.write(article)
    instance_copy = cr.to_dict()
    with open(os.path.join(args.output_dir, "instance_dump.json"), "w") as f:
        json.dump(instance_copy, f, indent=2)
    log_dump = cr.dump_logging_and_reset()
    with open(os.path.join(args.output_dir, "log.json"), "w") as f:
        json.dump(log_dump, f, indent=2)

# ----------------------- CLI -----------------------

if __name__ == "__main__":
    parser = ArgumentParser(description="Collaborative Storm CLI - Optimized")
    parser.add_argument("--collection-name", type=str, default="ekko_vectordb", help="Qdrant collection name.")
    parser.add_argument("--output-dir", type=str, default="./results/co-storm", help="Directory to store the outputs.")
    parser.add_argument("--retriever", type=str, choices=["vector", "web", "hybrid"], default="web",
                        help="Retrieval source: vector, web, or hybrid.")
    parser.add_argument("--web-source", type=str, choices=["brave", "tavily", "duckduckgo", "firecrawl"],
                        default="tavily", help="Web search engine to use.")
    parser.add_argument("--csv-file-path", type=str, default=None, help="Path to CSV for vector store population.")
    parser.add_argument("--enable_log_print", action="store_true", help="Enable console log printing.")
    parser.add_argument("--retrieve-top-k", type=int, default=3, help="Top-k results to retrieve.")
    # Additional provider-specific options can be added here if needed.
    main(parser.parse_args())