import json
import os
import time
from dataclasses import dataclass, field
from typing import Union

import dspy

from .modules.article_generation import StormArticleGenerationModule
from .modules.article_polish import StormArticlePolishingModule
from .modules.callback import BaseCallbackHandler
from .modules.knowledge_curation import StormKnowledgeCurationModule
from .modules.outline_generation import StormOutlineGenerationModule
from .modules.persona_generator import StormPersonaGenerator
from .modules.storm_dataclass import StormInformationTable, StormArticle
from ..interface import Engine, LMConfigs, Retriever
from ..lm import OpenAIModel
from ..utils import FileIOHelper, makeStringRed, truncate_filename

from concurrent.futures import ThreadPoolExecutor, as_completed

import hashlib

def hash_filename(text, max_len=40):
    """Create a shorter filename using hash for long topics."""
    if len(text) > max_len:
        h = hashlib.md5(text.encode()).hexdigest()[:8]
        return f"{text[:max_len - 9]}_{h}"  # -9 accounts for underscore + hash
    return text


def sanitize_for_json(obj):
    """Recursively convert non-serializable fields to strings or remove them."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(i) for i in obj]
    try:
        json.dumps(obj)
        return obj
    except (TypeError, OverflowError):
        return str(obj)

class STORMWikiLMConfigs(LMConfigs):
    """Configurations for LLM used in different parts of STORM."""

    def __init__(self):
        self.conv_simulator_lm = None
        self.question_asker_lm = None
        self.outline_gen_lm = None
        self.article_gen_lm = None
        self.article_polish_lm = None

    def init_azure_model(
        self,
        azure_api_key: str,
        api_base: str,
        api_version: str,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ):
        """Azure-based LLM initialization ONLY."""
        azure_kwargs = {
            "api_key": azure_api_key,
            "temperature": temperature,
            "top_p": top_p,
            "api_base": api_base,
            "api_version": api_version,
        }
        self.conv_simulator_lm = OpenAIModel(
            model="gpt-4o-mini", max_tokens=500, **azure_kwargs
        )
        self.question_asker_lm = OpenAIModel(
            model="gpt-4o-mini", max_tokens=500, **azure_kwargs
        )
        self.outline_gen_lm = OpenAIModel(
            model="gpt-4o-mini", max_tokens=400, **azure_kwargs
        )
        self.article_gen_lm = OpenAIModel(
            model="gpt-4o-mini", max_tokens=700, **azure_kwargs
        )
        self.article_polish_lm = OpenAIModel(
            model="gpt-4o-mini", max_tokens=4000, **azure_kwargs
        )

    def set_conv_simulator_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.conv_simulator_lm = model

    def set_question_asker_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.question_asker_lm = model

    def set_outline_gen_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.outline_gen_lm = model

    def set_article_gen_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.article_gen_lm = model

    def set_article_polish_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.article_polish_lm = model

@dataclass
class STORMWikiRunnerArguments:
    """Arguments for controlling the STORM Wiki pipeline."""
    output_dir: str = field(metadata={"help": "Output directory for the results."})
    max_conv_turn: int = field(default=3, metadata={"help": "Maximum number of questions in conversational question asking."})
    max_perspective: int = field(default=3, metadata={"help": "Maximum number of perspectives to consider in perspective-guided question asking."})
    max_search_queries_per_turn: int = field(default=3, metadata={"help": "Maximum number of search queries to consider in each turn."})
    disable_perspective: bool = field(default=False, metadata={"help": "If True, disable perspective-guided question asking."})
    search_top_k: int = field(default=3, metadata={"help": "Top k search results to consider for each search query."})
    retrieve_top_k: int = field(default=3, metadata={"help": "Top k collected references for each section title."})
    max_thread_num: int = field(default=10, metadata={"help": "Maximum number of threads to use. Consider reducing it if keep getting 'Exceed rate limit' error when calling LM API."})

class STORMWikiRunner(Engine):
    """STORM Wiki pipeline runner."""

    def __init__(
        self, args: STORMWikiRunnerArguments, lm_configs: STORMWikiLMConfigs, rm
    ):
        super().__init__(lm_configs=lm_configs)
        self.args = args
        self.lm_configs = lm_configs

        self.retriever = Retriever(rm=rm, max_thread=self.args.max_thread_num)
        storm_persona_generator = StormPersonaGenerator(
            self.lm_configs.question_asker_lm
        )
        self.storm_knowledge_curation_module = StormKnowledgeCurationModule(
            retriever=self.retriever,
            persona_generator=storm_persona_generator,
            conv_simulator_lm=self.lm_configs.conv_simulator_lm,
            question_asker_lm=self.lm_configs.question_asker_lm,
            max_search_queries_per_turn=self.args.max_search_queries_per_turn,
            search_top_k=self.args.search_top_k,
            max_conv_turn=self.args.max_conv_turn,
            max_thread_num=self.args.max_thread_num,
        )
        self.storm_outline_generation_module = StormOutlineGenerationModule(
            outline_gen_lm=self.lm_configs.outline_gen_lm
        )
        self.storm_article_generation = StormArticleGenerationModule(
            article_gen_lm=self.lm_configs.article_gen_lm,
            retrieve_top_k=self.args.retrieve_top_k,
            max_thread_num=self.args.max_thread_num,
        )
        self.storm_article_polishing_module = StormArticlePolishingModule(
            article_gen_lm=self.lm_configs.article_gen_lm,
            article_polish_lm=self.lm_configs.article_polish_lm,
        )

        self.lm_configs.init_check()
        self.apply_decorators()

    def run_knowledge_curation_module(
        self,
        ground_truth_url: str = "None",
        callback_handler: BaseCallbackHandler = None,
    ) -> StormInformationTable:
        t0 = time.time()
        information_table, conversation_log = self.storm_knowledge_curation_module.research(
            topic=self.topic,
            ground_truth_url=ground_truth_url,
            callback_handler=callback_handler,
            max_perspective=self.args.max_perspective,
            disable_perspective=False,
            return_conversation_log=True,
        )
        t1 = time.time()
        FileIOHelper.dump_json(
            conversation_log,
            os.path.join(self.article_output_dir, "conversation_log.json"),
        )
        information_table.dump_url_to_info(
            os.path.join(self.article_output_dir, "raw_search_results.json")
        )
        t2 = time.time()
        print(f"[INFO] Knowledge curation: research={t1-t0:.2f}s, dumps={t2-t1:.2f}s")
        return information_table

    def run_outline_generation_module(
        self,
        information_table: StormInformationTable,
        callback_handler: BaseCallbackHandler = None,
    ) -> StormArticle:
        t0 = time.time()
        outline, draft_outline = self.storm_outline_generation_module.generate_outline(
            topic=self.topic,
            information_table=information_table,
            return_draft_outline=True,
            callback_handler=callback_handler,
        )
        t1 = time.time()
        outline.dump_outline_to_file(
            os.path.join(self.article_output_dir, "storm_gen_outline.txt")
        )
        draft_outline.dump_outline_to_file(
            os.path.join(self.article_output_dir, "direct_gen_outline.txt")
        )
        t2 = time.time()
        print(f"[INFO] Outline generation: generate={t1-t0:.2f}s, dumps={t2-t1:.2f}s")
        return outline

    def run_article_generation_module(
        self,
        outline: StormArticle,
        information_table=StormInformationTable,
        callback_handler: BaseCallbackHandler = None,
    ) -> StormArticle:
        t0 = time.time()
        draft_article = self.storm_article_generation.generate_article(
            topic=self.topic,
            information_table=information_table,
            article_with_outline=outline,
            callback_handler=callback_handler,
        )
        t1 = time.time()
        with ThreadPoolExecutor(max_workers=2) as executor:
            paths = [
                (draft_article.dump_article_as_plain_text,
                 os.path.join(self.article_output_dir, "storm_gen_article.txt")),
                (draft_article.dump_reference_to_file,
                 os.path.join(self.article_output_dir, "url_to_info.json"))
            ]
            futures = [executor.submit(func, path) for func, path in paths]
            for _ in as_completed(futures):
                pass
        t2 = time.time()
        print(f"[INFO] Article generation: generate={t1-t0:.2f}s, dumps={t2-t1:.2f}s")
        return draft_article

    def run_article_polishing_module(
        self, draft_article: StormArticle, remove_duplicate: bool = False
    ) -> StormArticle:
        t0 = time.time()
        polished_article = self.storm_article_polishing_module.polish_article(
            topic=self.topic,
            draft_article=draft_article,
            remove_duplicate=remove_duplicate,
        )
        t1 = time.time()
        FileIOHelper.write_str(
            polished_article.to_string(),
            os.path.join(self.article_output_dir, "storm_gen_article_polished.txt"),
        )
        t2 = time.time()
        print(f"[INFO] Article polishing: polish={t1-t0:.2f}s, dump={t2-t1:.2f}s")
        return polished_article

    def post_run(self):
        """
        Post-run operations, including:
        1. Dumping the run configuration.
        2. Dumping the LLM call history.
        """
        config_log = self.lm_configs.log()
        FileIOHelper.dump_json(
            config_log, os.path.join(self.article_output_dir, "run_config.json")
        )

        llm_call_history = self.lm_configs.collect_and_reset_lm_history()
        with open(
            os.path.join(self.article_output_dir, "llm_call_history.jsonl"), "w"
        ) as f:
            for call in llm_call_history:
                if "kwargs" in call:
                    call.pop("kwargs")
                sanitized_call = sanitize_for_json(call)
                f.write(json.dumps(sanitized_call) + "\n")

    def _load_information_table_from_local_fs(self, information_table_local_path):
        assert os.path.exists(information_table_local_path), makeStringRed(
            f"{information_table_local_path} not exists. Please set --do-research argument to prepare the conversation_log.json for this topic."
        )
        return StormInformationTable.from_conversation_log_file(
            information_table_local_path
        )

    def _load_outline_from_local_fs(self, topic, outline_local_path):
        assert os.path.exists(outline_local_path), makeStringRed(
            f"{outline_local_path} not exists. Please set --do-generate-outline argument to prepare the storm_gen_outline.txt for this topic."
        )
        return StormArticle.from_outline_file(topic=topic, file_path=outline_local_path)

    def _load_draft_article_from_local_fs(
        self, topic, draft_article_path, url_to_info_path
    ):
        assert os.path.exists(draft_article_path), makeStringRed(
            f"{draft_article_path} not exists. Please set --do-generate-article argument to prepare the storm_gen_article.txt for this topic."
        )
        assert os.path.exists(url_to_info_path), makeStringRed(
            f"{url_to_info_path} not exists. Please set --do-generate-article argument to prepare the url_to_info.json for this topic."
        )
        article_text = FileIOHelper.load_str(draft_article_path)
        references = FileIOHelper.load_json(url_to_info_path)
        return StormArticle.from_string(
            topic_name=topic, article_text=article_text, references=references
        )

    def run(
        self,
        topic: str,
        ground_truth_url: str = "",
        do_research: bool = True,
        do_generate_outline: bool = True,
        do_generate_article: bool = True,
        do_polish_article: bool = True,
        remove_duplicate: bool = False,
        callback_handler: BaseCallbackHandler = BaseCallbackHandler(),
    ):
        """
        Run the STORM pipeline.
        """
        assert (
            do_research
            or do_generate_outline
            or do_generate_article
            or do_polish_article
        ), makeStringRed(
            "No action is specified. Please set at least one of --do-research, --do-generate-outline, --do-generate-article, --do-polish-article"
        )

        self.topic = topic
        sanitized_topic = topic.replace(" ", "_").replace("/", "_")
        self.article_dir_name = hash_filename(sanitized_topic, 40)
        self.article_output_dir = os.path.join(self.args.output_dir, self.article_dir_name)
        os.makedirs(self.article_output_dir, exist_ok=True)

        # research module
        information_table: StormInformationTable = None
        if do_research:
            information_table = self.run_knowledge_curation_module(
                ground_truth_url=ground_truth_url, callback_handler=callback_handler
            )
        # outline generation module
        outline: StormArticle = None
        if do_generate_outline:
            if information_table is None:
                information_table = self._load_information_table_from_local_fs(
                    os.path.join(self.article_output_dir, "conversation_log.json")
                )
            outline = self.run_outline_generation_module(
                information_table=information_table, callback_handler=callback_handler
            )

        # article generation module
        draft_article: StormArticle = None
        if do_generate_article:
            if information_table is None:
                information_table = self._load_information_table_from_local_fs(
                    os.path.join(self.article_output_dir, "conversation_log.json")
                )
            if outline is None:
                outline = self._load_outline_from_local_fs(
                    topic=topic,
                    outline_local_path=os.path.join(
                        self.article_output_dir, "storm_gen_outline.txt"
                    ),
                )
            draft_article = self.run_article_generation_module(
                outline=outline,
                information_table=information_table,
                callback_handler=callback_handler,
            )

        # article polishing module
        if do_polish_article:
            if draft_article is None:
                draft_article_path = os.path.join(
                    self.article_output_dir, "storm_gen_article.txt"
                )
                url_to_info_path = os.path.join(
                    self.article_output_dir, "url_to_info.json"
                )
                draft_article = self._load_draft_article_from_local_fs(
                    topic=topic,
                    draft_article_path=draft_article_path,
                    url_to_info_path=url_to_info_path,
                )
            self.run_article_polishing_module(
                draft_article=draft_article, remove_duplicate=remove_duplicate
            )