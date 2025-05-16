import os
from dotenv import load_dotenv

from src.knowledge_storm import (
    STORMWikiRunnerArguments,
    STORMWikiRunner,
    STORMWikiLMConfigs,
)
from src.knowledge_storm.lm import (
    AzureOpenAIModel,
    GroqModel,
    OllamaClient,
)
from src.knowledge_storm.rm import SearXNG
from src.knowledge_storm.encoder import get_encoder

load_dotenv()

class ResearchManager:
    def __init__(self):
        self.encoder = None
        self.topic = None

    def set_topic(self, topic):
        self.topic = topic

    def _get_encoder(self):
        if self.encoder is None:
            self.encoder = get_encoder()
            print(f"Embedding backend initialized: {self.encoder}")
        return self.encoder

    def _get_research_only_engines(self):
        return [
            "soundcloud", "bandcamp", "deezer", "bilibili", "spotify", "mixcloud",
            "1337x", "piratebay", "nyaa", "kickass", "btdigg", "bt4g",
            "solidtorrents", "torznab", "moviepilot", "peertube", "rumble", "openclipart",
            "emojipedia", "findthatmeme", "frinkiac", "podcastindex", "radio_browser", "apple_music", "steam"
        ]

    def _get_retriever(self):
        research_only = self._get_research_only_engines()
        return SearXNG(
            searxng_api_url=os.getenv("SEARXNG_URL"),
            disabled_engines=research_only
        )

    def run_groq(self, output_dir="./results/groq"):
        self._get_encoder()
        novita_kwargs = {
            "api_key": os.getenv("NOVITA_API_KEY"),
            "api_base": "https://api.novita.ai/v3/openai",
            "temperature": 1.0,
            "top_p": 0.8,
        }
        groq_kwargs = {
            "api_key": os.environ.get("GROQ_API_KEY"),
            "api_base": "https://api.groq.com/openai/v1",
            "temperature": 1.0,
            "top_p": 0.8,
        }

        lm_configs = STORMWikiLMConfigs()
        lm_configs.set_conv_simulator_lm(GroqModel("llama-3.1-8b-instant", max_tokens=800, **groq_kwargs))
        lm_configs.set_question_asker_lm(GroqModel("llama-3.1-8b-instant", max_tokens=800, **groq_kwargs))
        for stage in ["set_outline_gen_lm", "set_article_gen_lm", "set_article_polish_lm"]:
            getattr(lm_configs, stage)(
                GroqModel("qwen/qwen3-235b-a22b-fp8", max_tokens=96000, **novita_kwargs)
            )

        max_threads = max(1, (os.cpu_count() or 4) - 2)
        engine_args = STORMWikiRunnerArguments(
            output_dir=output_dir,
            max_conv_turn=5,
            max_perspective=5,
            search_top_k=5,
            max_thread_num=max_threads,
        )

        retriever = self._get_retriever()
        runner = STORMWikiRunner(engine_args, lm_configs, retriever)
        runner.run(
            topic=self.topic,
            do_research=True,
            do_generate_outline=True,
            do_generate_article=True,
            do_polish_article=True,
        )
        runner.post_run()
        runner.summary()

    def run_ollama(self, output_dir="./results/ollama"):
        lm_configs = STORMWikiLMConfigs()

        speed_ollama_kwargs = {
            "model": "llama3.1:8b-instruct-q8_0",
            "port": "11434",
            "url": "http://172.31.50.10",
            "stop": ("\n\n---",),
        }
        think_ollama_kwargs = {
            "model": "qwen3:32b",
            "port": "11434",
            "url": "http://172.31.50.10",
            "stop": ("\n\n---",),
        }

        lm_configs.set_conv_simulator_lm(OllamaClient(max_tokens=500, **speed_ollama_kwargs))
        lm_configs.set_question_asker_lm(OllamaClient(max_tokens=500, **speed_ollama_kwargs))
        lm_configs.set_outline_gen_lm(OllamaClient(max_tokens=400, **think_ollama_kwargs))
        lm_configs.set_article_gen_lm(OllamaClient(max_tokens=700, **think_ollama_kwargs))
        lm_configs.set_article_polish_lm(OllamaClient(max_tokens=4000, **think_ollama_kwargs))

        engine_args = STORMWikiRunnerArguments(
            output_dir=output_dir,
            max_conv_turn=3,
            max_perspective=3,
            search_top_k=3,
            max_thread_num=3,
        )

        retriever = self._get_retriever()
        runner = STORMWikiRunner(engine_args, lm_configs, retriever)
        runner.run(
            topic=self.topic,
            do_research=True,
            do_generate_outline=True,
            do_generate_article=True,
            do_polish_article=True,
        )
        runner.post_run()
        runner.summary()

    def _build_azure_openai_model(self, model_name: str, max_tokens: int, **openai_kwargs):
        return AzureOpenAIModel(
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            api_version=os.getenv("AZURE_API_VERSION"),
            model=model_name,
            max_tokens=max_tokens,
            **openai_kwargs,
        )

    def run_gpt(self, output_dir="./results/gpt"):
        self._get_encoder()
        openai_kwargs = {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "temperature": 1.0,
            "top_p": 0.9,
        }
        gpt_4_model_name = "gpt-4o-mini"

        lm_configs = STORMWikiLMConfigs()
        lm_configs.set_conv_simulator_lm(self._build_azure_openai_model(gpt_4_model_name, 1000, **openai_kwargs))
        lm_configs.set_question_asker_lm(self._build_azure_openai_model(gpt_4_model_name, 1000, **openai_kwargs))
        lm_configs.set_outline_gen_lm(self._build_azure_openai_model(gpt_4_model_name, 500, **openai_kwargs))
        lm_configs.set_article_gen_lm(self._build_azure_openai_model(gpt_4_model_name, 16384, **openai_kwargs))
        lm_configs.set_article_polish_lm(self._build_azure_openai_model(gpt_4_model_name, 16384, **openai_kwargs))

        max_threads = max(1, (os.cpu_count() or 4) - 2)
        engine_args = STORMWikiRunnerArguments(
            output_dir=output_dir,
            max_conv_turn=5,
            max_perspective=5,
            search_top_k=5,
            max_thread_num=max_threads,
        )

        retriever = self._get_retriever()
        runner = STORMWikiRunner(engine_args, lm_configs, retriever)
        runner.run(
            topic=self.topic,
            do_research=True,
            do_generate_outline=True,
            do_generate_article=True,
            do_polish_article=True,
        )
        runner.post_run()
        runner.summary()