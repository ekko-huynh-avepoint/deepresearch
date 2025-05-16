import os
from dotenv import load_dotenv

from src.knowledge_storm import (
    STORMWikiRunnerArguments,
    STORMWikiRunner,
    STORMWikiLMConfigs,
)
from src.knowledge_storm.lm import AzureOpenAIModel
from src.knowledge_storm.rm import SearXNG
from src.knowledge_storm.encoder import get_encoder

load_dotenv()

def build_azure_openai_model(model_name: str, max_tokens: int, **openai_kwargs):
    """Factory for consistent AzureOpenAIModel creation."""
    return AzureOpenAIModel(
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_version=os.getenv("AZURE_API_VERSION"),
        model=model_name,
        max_tokens=max_tokens,
        **openai_kwargs,
    )

def main(args):
    # ---- Embedding setup at the start ----
    encoder = get_encoder()
    print(f"Embedding backend initialized: {encoder}")

    openai_kwargs = {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "temperature": 1.0,
        "top_p": 0.9,
    }
    gpt_4_model_name = "gpt-4o-mini"

    # Create all language model instances
    conv_simulator_lm = build_azure_openai_model(gpt_4_model_name, 1000, **openai_kwargs)
    question_asker_lm = build_azure_openai_model(gpt_4_model_name, 1000, **openai_kwargs)
    outline_gen_lm = build_azure_openai_model(gpt_4_model_name, 500, **openai_kwargs)
    article_gen_lm = build_azure_openai_model(gpt_4_model_name, 16384, **openai_kwargs)
    article_polish_lm = build_azure_openai_model(gpt_4_model_name, 16384, **openai_kwargs)

    # Set up LM configs
    lm_configs = STORMWikiLMConfigs()
    lm_configs.set_conv_simulator_lm(conv_simulator_lm)
    lm_configs.set_question_asker_lm(question_asker_lm)
    lm_configs.set_outline_gen_lm(outline_gen_lm)
    lm_configs.set_article_gen_lm(article_gen_lm)
    lm_configs.set_article_polish_lm(article_polish_lm)

    # Setup engine arguments
    max_threads = max(1, (os.cpu_count() or 4) - 2)
    engine_args = STORMWikiRunnerArguments(
        output_dir=args.output_dir,
        max_conv_turn=5,
        max_perspective=5,
        search_top_k=5,
        max_thread_num=max_threads,
    )

    # List of engines to disable for research-only retrieval
    research_only = [
        "soundcloud", "bandcamp", "deezer", "bilibili", "spotify", "mixcloud",
        "1337x", "piratebay", "nyaa", "kickass", "btdigg", "bt4g",
        "solidtorrents", "torznab", "moviepilot", "peertube", "rumble", "openclipart",
        "emojipedia", "findthatmeme", "frinkiac", "podcastindex", "radio_browser", "apple_music", "steam"
    ]
    retriever = SearXNG(
        searxng_api_url=os.getenv("SEARXNG_URL"),
        disabled_engines=research_only
    )

    runner = STORMWikiRunner(engine_args, lm_configs, retriever)

    topic = input("Topic: ")
    runner.run(
        topic=topic,
        do_research=True,
        do_generate_outline=True,
        do_generate_article=True,
        do_polish_article=True,
    )
    runner.post_run()
    runner.summary()

if __name__ == "__main__":
    class Args:
        def __init__(self):
            self.output_dir = "./results/gpt"
    main(Args())