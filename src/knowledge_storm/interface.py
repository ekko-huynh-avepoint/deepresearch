import concurrent.futures
import dspy
import functools
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, List, Optional, Union, TYPE_CHECKING

from .utils import ArticleTextProcessing

logging.basicConfig(
    level=logging.INFO, format="%(name)s : %(levelname)-8s : %(message)s"
)
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .logging_wrapper import LoggingWrapper


class InformationTable(ABC):
    """Data class for information collected during KnowledgeCuration.
    Subclass to extend (e.g., add guided dialogue history).
    """
    @abstractmethod
    def retrieve_information(**kwargs):
        pass


class Information:
    def __init__(self, url, description, snippets, title, meta=None):
        self.url = url
        self.description = description
        self.snippets = snippets
        self.title = title
        self.meta = meta or {}
        self.citation_uuid = -1

    def __eq__(self, other):
        if not isinstance(other, Information):
            return False
        return (
            self.url == other.url
            and set(self.snippets) == set(other.snippets)
            and self._meta_str() == other._meta_str()
        )

    def __hash__(self):
        # MD5-based hash for consistency; includes meta string for uniqueness
        return int(
            self._md5_hash((self.url, tuple(sorted(self.snippets)), self._meta_str())), 16
        )

    def _meta_str(self):
        return f"Question: {self.meta.get('question', '')}, Query: {self.meta.get('query', '')}"

    @staticmethod
    def _md5_hash(value):
        if isinstance(value, (dict, list, tuple)):
            value = json.dumps(value, sort_keys=True)
        return hashlib.md5(str(value).encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, info_dict):
        info = cls(
            url=info_dict["url"],
            description=info_dict["description"],
            snippets=info_dict["snippets"],
            title=info_dict["title"],
            meta=info_dict.get("meta", None),
        )
        info.citation_uuid = int(info_dict.get("citation_uuid", -1))
        return info

    def to_dict(self):
        return {
            "url": self.url,
            "description": self.description,
            "snippets": self.snippets,
            "title": self.title,
            "meta": self.meta,
            "citation_uuid": self.citation_uuid,
        }


class ArticleSectionNode:
    """Dataclass for an article section (tree node)."""
    def __init__(self, section_name: str, content=None):
        self.section_name = section_name
        self.content = content
        self.children = []
        self.preference = None

    def add_child(self, new_child_node, insert_to_front=False):
        if insert_to_front:
            self.children.insert(0, new_child_node)
        else:
            self.children.append(new_child_node)

    def remove_child(self, child):
        self.children.remove(child)


class Article(ABC):
    def __init__(self, topic_name):
        self.root = ArticleSectionNode(topic_name)

    def find_section(self, node: ArticleSectionNode, name: str) -> Optional[ArticleSectionNode]:
        """Return node for given section name (recursive)."""
        if node.section_name == name:
            return node
        for child in node.children:
            result = self.find_section(child, name)
            if result:
                return result
        return None

    @abstractmethod
    def to_string(self) -> str:
        """Export article to string."""

    def get_outline_tree(self):
        """Return hierarchical dict representing outline structure."""
        def build_tree(node) -> Dict[str, Dict]:
            return {
                child.section_name: build_tree(child)
                for child in node.children
            }
        return build_tree(self.root)

    def get_first_level_section_names(self) -> List[str]:
        return [i.section_name for i in self.root.children]

    @classmethod
    @abstractmethod
    def from_string(cls, topic_name: str, article_text: str):
        """Instantiate article from string."""

    def prune_empty_nodes(self, node=None):
        if node is None:
            node = self.root
        node.children[:] = [
            child for child in node.children if self.prune_empty_nodes(child)
        ]
        if (not node.content) and not node.children:
            return None
        return node


class Retriever:
    """Base class for retriever modules."""
    def __init__(self, rm: dspy.Retrieve, max_thread: int = 1):
        self.max_thread = max_thread
        self.rm = rm

    def collect_and_reset_rm_usage(self):
        # Only collect usage if rm has get_usage_and_reset
        usage = getattr(self.rm, "get_usage_and_reset", lambda: {})()
        return usage if usage else {}

    def retrieve(self, query: Union[str, List[str]], exclude_urls=None) -> List[Information]:
        if exclude_urls is None:
            exclude_urls = []
        queries = query if isinstance(query, list) else [query]
        def process_query(q):
            retrieved_data_list = self.rm(query_or_queries=[q], exclude_urls=exclude_urls)
            results = []
            for data in retrieved_data_list:
                data["snippets"] = [
                    ArticleTextProcessing.remove_citations(snip)
                    for snip in data["snippets"]
                ]
                info = Information.from_dict(data)
                info.meta["query"] = q
                results.append(info)
            return results

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_thread) as executor:
            results = executor.map(process_query, queries)
        to_return = []
        for result in results:
            to_return.extend(result)
        return to_return


class KnowledgeCurationModule(ABC):
    """Interface for knowledge curation stage."""
    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    @abstractmethod
    def research(self, topic) -> InformationTable:
        pass


class OutlineGenerationModule(ABC):
    """Interface for outline generation stage."""
    @abstractmethod
    def generate_outline(self, topic: str, information_table: InformationTable, **kwargs) -> Article:
        pass


class ArticleGenerationModule(ABC):
    """Interface for article generation stage."""
    @abstractmethod
    def generate_article(
        self,
        topic: str,
        information_table: InformationTable,
        article_with_outline: Article,
        **kwargs,
    ) -> Article:
        pass


class ArticlePolishingModule(ABC):
    """Interface for article polishing stage."""
    @abstractmethod
    def polish_article(self, topic: str, draft_article: Article, **kwargs) -> Article:
        pass


def log_execution_time(func):
    """Decorator to log execution time of a function."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        elapsed = time.time() - start_time
        logger.info(f"{func.__name__} executed in {elapsed:.4f} seconds")
        if hasattr(self, "time"):
            self.time[func.__name__] = elapsed
        return result
    return wrapper


class LMConfigs(ABC):
    """Abstract base class for language model configs."""
    def init_check(self):
        for attr in self.__dict__:
            if "_lm" in attr and getattr(self, attr) is None:
                logging.warning(f"Language model for {attr} is not initialized. Please call set_{attr}()")

    def collect_and_reset_lm_history(self):
        history = []
        for attr in self.__dict__:
            lm = getattr(self, attr)
            if "_lm" in attr and hasattr(lm, "history"):
                history.extend(lm.history)
                lm.history = []
        return history

    def collect_and_reset_lm_usage(self):
        combined_usage = []
        for attr in self.__dict__:
            lm = getattr(self, attr)
            if "_lm" in attr and hasattr(lm, "get_usage_and_reset"):
                combined_usage.append(lm.get_usage_and_reset())
        model_name_to_usage = {}
        for usage in combined_usage:
            for model_name, tokens in usage.items():
                if model_name not in model_name_to_usage:
                    model_name_to_usage[model_name] = tokens
                else:
                    model_name_to_usage[model_name]["prompt_tokens"] += tokens["prompt_tokens"]
                    model_name_to_usage[model_name]["completion_tokens"] += tokens["completion_tokens"]
        return model_name_to_usage

    def log(self):
        return OrderedDict(
            {
                attr: getattr(self, attr).kwargs
                for attr in self.__dict__
                if "_lm" in attr and hasattr(getattr(self, attr), "kwargs")
            }
        )


class Engine(ABC):
    def __init__(self, lm_configs: LMConfigs):
        self.lm_configs = lm_configs
        self.time = {}
        self.lm_cost = {}
        self.rm_cost = {}

    def log_execution_time_and_lm_rm_usage(self, func):
        """Decorator to log execution time, LM and RM usage."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            self.time[func.__name__] = elapsed
            logger.info(f"{func.__name__} executed in {elapsed:.4f} seconds")
            self.lm_cost[func.__name__] = self.lm_configs.collect_and_reset_lm_usage()
            if hasattr(self, "retriever"):
                self.rm_cost[func.__name__] = self.retriever.collect_and_reset_rm_usage()
            return result
        return wrapper

    def apply_decorators(self):
        """Apply decorators to run_ methods."""
        for method_name in dir(self):
            if callable(getattr(self, method_name)) and method_name.startswith("run_"):
                original = getattr(self, method_name)
                decorated = self.log_execution_time_and_lm_rm_usage(original)
                setattr(self, method_name, decorated)

    @abstractmethod
    def run_knowledge_curation_module(self, **kwargs) -> Optional[InformationTable]:
        pass

    @abstractmethod
    def run_outline_generation_module(self, **kwargs) -> Article:
        pass

    @abstractmethod
    def run_article_generation_module(self, **kwargs) -> Article:
        pass

    @abstractmethod
    def run_article_polishing_module(self, **kwargs) -> Article:
        pass

    @abstractmethod
    def run(self, **kwargs):
        pass

    def summary(self):
        print("***** Execution time *****")
        for k, v in self.time.items():
            print(f"{k}: {v:.4f} seconds")
        print("***** Token usage of language models: *****")
        for k, v in self.lm_cost.items():
            print(f"{k}")
            for model_name, tokens in v.items():
                print(f"    {model_name}: {tokens}")
        print("***** Number of queries of retrieval models: *****")
        for k, v in self.rm_cost.items():
            print(f"{k}: {v}")

    def reset(self):
        self.time.clear()
        self.lm_cost.clear()
        self.rm_cost.clear()


class Agent(ABC):
    from .dataclass import KnowledgeBase, ConversationTurn

    def __init__(self, topic: str, role_name: str, role_description: str):
        self.topic = topic
        self.role_name = role_name
        self.role_description = role_description

    def get_role_description(self):
        return f"{self.role_name}: {self.role_description}" if self.role_description else self.role_name

    @abstractmethod
    def generate_utterance(
        self,
        knowledge_base: KnowledgeBase,
        conversation_history: List[ConversationTurn],
        logging_wrapper: "LoggingWrapper",
        **kwargs,
    ):
        pass