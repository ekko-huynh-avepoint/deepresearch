import os
from argparse import ArgumentParser
from src.knowledge_storm.lm import OllamaClient
from src.knowledge_storm.rm import (
    SearXNG,
)
from src.knowledge_storm import (
    STORMWikiRunnerArguments,
    STORMWikiRunner,
    STORMWikiLMConfigs,
)


def main(args):
    lm_configs = STORMWikiLMConfigs()


    speed_ollama_kwargs = {
        "model": "llama3.1:8b-instruct-q8_0",
        "port": "11434",
        "url": "http://172.31.50.10",
        "stop": (
            "\n\n---",
        ),
    }

    think_ollama_kwargs = {
        "model": "qwen3:32b",
        "port": "11434",
        "url": "http://172.31.50.10",
        "stop": (
            "\n\n---",
        ),
    }

    conv_simulator_lm = OllamaClient(max_tokens=500, **speed_ollama_kwargs)
    question_asker_lm = OllamaClient(max_tokens=500, **speed_ollama_kwargs)

    outline_gen_lm = OllamaClient(max_tokens=400, **think_ollama_kwargs)
    article_gen_lm = OllamaClient(max_tokens=700, **think_ollama_kwargs)

    article_polish_lm = OllamaClient(max_tokens=4000, **think_ollama_kwargs)

    lm_configs.set_conv_simulator_lm(conv_simulator_lm)
    lm_configs.set_question_asker_lm(question_asker_lm)
    lm_configs.set_outline_gen_lm(outline_gen_lm)
    lm_configs.set_article_gen_lm(article_gen_lm)
    lm_configs.set_article_polish_lm(article_polish_lm)

    engine_args = STORMWikiRunnerArguments(
        output_dir=args.output_dir,
        max_conv_turn=3,
        max_perspective=3,
        search_top_k=3,
        max_thread_num=3,
    )

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
            self.output_dir = "./results/ollama"
    main(Args())
