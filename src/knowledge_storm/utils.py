import concurrent.futures
import httpx
import json
import logging
import os
import pickle
import re
import regex
from typing import List, Dict, Optional, Any

from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from trafilatura import extract
from qdrant_client import models
from .lm import OpenAIModel
import pandas as pd
from tqdm import tqdm

logging.getLogger("httpx").setLevel(logging.WARNING)


def truncate_filename(filename: str, max_length: int = 125) -> str:
    """Truncate filename to max_length for filesystem safety."""
    if len(filename) > max_length:
        truncated = filename[:max_length]
        logging.warning(f"Filename is too long. Truncated to {truncated}.")
        return truncated
    return filename



def makeStringRed(message: str) -> str:
    return f"\033[91m {message}\033[00m"


class QdrantVectorStoreManager:
    def __init__(self, client, collection_name: str, model):
        self.client = client
        self.collection_name = collection_name
        self.model = model
        self.qdrant = self._check_create_collection()

    def _check_create_collection(self):
        if self.client is None:
            raise ValueError("Qdrant client is not initialized.")

        # Check if the collection exists and delete it if you want to force recreate
        if self.client.collection_exists(self.collection_name):
            print(f"Collection {self.collection_name} exists. Deleting for recreation...")
            self.client.delete_collection(self.collection_name)
            print(f"Collection {self.collection_name} deleted.")

        print(f"Creating collection {self.collection_name}...")
        try:
            dummy_vector = self.model.embed_query("test")
            vector_size = len(dummy_vector)
        except Exception:
            vector_size = 1024
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )

        return QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.model,
        )

    def create_or_update_vector_store(
        self,
        file_path: str,
        content_column: str,
        title_column: str = "title",
        url_column: str = "url",
        desc_column: str = "description",
        batch_size: int = 64,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ):


        if not self.collection_name:
            raise ValueError("Please provide a collection name.")
        if not file_path or not file_path.endswith(".csv"):
            raise ValueError("Please provide a valid CSV file path.")
        if not content_column or not url_column:
            raise ValueError("Please provide required column names.")

        df = pd.read_csv(file_path)
        for col in [content_column, url_column]:
            if col not in df.columns:
                raise ValueError(f"Column {col} not found in the CSV file.")

        documents = [
            {
                "page_content": row[content_column],
                "metadata": {
                    "title": row.get(title_column, ""),
                    "url": row[url_column],
                    "description": row.get(desc_column, ""),
                },
            }
            for row in df.to_dict(orient="records")
        ]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
            separators=[
                "\n\n", "\n", ".", "\uff0e", "\u3002", ",", "\uff0c", "\u3001", " ", "\u200B", ""
            ],
        )
        split_documents = text_splitter.split_documents(documents)

        for i in tqdm(range(0, len(split_documents), batch_size)):
            self.qdrant.add_documents(
                documents=split_documents[i:i+batch_size],
                batch_size=batch_size,
            )

        # Close the qdrant client connection if available
        if hasattr(self.qdrant, "client") and hasattr(self.qdrant.client, "close"):
            self.qdrant.client.close()

    def get_qdrant(self):
        return self.qdrant

    def get_client(self):
        return self.client

    def get_embeddings(self):
        return self.model


class ArticleTextProcessing:
    @staticmethod
    def limit_word_count_preserve_newline(input_string: str, max_word_count: int) -> str:
        """Limit string to max words, preserving line integrity."""
        word_count = 0
        lines = []
        for line in input_string.split("\n"):
            line_words = line.split()
            if word_count + len(line_words) <= max_word_count:
                lines.append(line)
                word_count += len(line_words)
            else:
                allowed = max_word_count - word_count
                if allowed > 0:
                    lines.append(" ".join(line_words[:allowed]))
                break
        return "\n".join(lines).strip()

    @staticmethod
    def remove_citations(s: str) -> str:
        """Remove [1], [2,3] etc. citation patterns."""
        return re.sub(r"\[\d+(?:,\s*\d+)*]", "", s)

    @staticmethod
    def parse_citation_indices(s: str) -> List[int]:
        """Extract citation indices as integers."""
        return [int(index[1:-1]) for index in re.findall(r"\[\d+]", s)]

    @staticmethod
    def remove_uncompleted_sentences_with_citations(text: str) -> str:
        """Remove incomplete sentences and group citations properly."""
        def replace_with_individual_brackets(match):
            numbers = match.group(1).split(", ")
            return " ".join(f"[{n}]" for n in numbers)
        def deduplicate_group(match):
            citations = match.group(0)
            unique = sorted(set(re.findall(r"\[\d+]", citations)), key=lambda x: int(x.strip("[]")))
            return "".join(unique)

        text = re.sub(r"\[([0-9, ]+)]", replace_with_individual_brackets, text)
        text = re.sub(r"(\[\d+])+", deduplicate_group, text)
        eos_pattern = r"([.!?])\s*(\[\d+\])?\s*"
        matches = list(re.finditer(eos_pattern, text))
        if matches:
            text = text[:matches[-1].end()].strip()
        return text

    @staticmethod
    def clean_up_citation(conv: Any) -> Any:
        for turn in getattr(conv, "dlg_history", []):
            for marker in ("References:", "Sources:"):
                if marker in turn.agent_utterance:
                    turn.agent_utterance = turn.agent_utterance[: turn.agent_utterance.find(marker)]
            turn.agent_utterance = turn.agent_utterance.replace("Answer:", "").strip()
            try:
                max_ref_num = max([int(x) for x in re.findall(r"\[(\d+)]", turn.agent_utterance)])
            except Exception:
                max_ref_num = 0
            if max_ref_num > len(getattr(turn, "search_results", [])):
                for i in range(len(turn.search_results), max_ref_num + 1):
                    turn.agent_utterance = turn.agent_utterance.replace(f"[{i}]", "")
            turn.agent_utterance = ArticleTextProcessing.remove_uncompleted_sentences_with_citations(turn.agent_utterance)
        return conv

    @staticmethod
    def clean_up_outline(outline: str, topic: str = "") -> str:
        output_lines = []
        current_level = 0
        for line in outline.split("\n"):
            stripped = line.strip()
            if topic and f"# {topic.lower()}" in stripped.lower():
                output_lines = []
            if stripped.startswith("#"):
                current_level = stripped.count("#")
                output_lines.append(stripped)
            elif stripped.startswith("-"):
                output_lines.append("#" * (current_level + 1) + " " + stripped[1:].strip())
        outline = "\n".join(output_lines)
        # Remove references/appendix sections and citations
        patterns_to_remove = [
            r"#[#]? See also.*?(?=##|$)", r"#[#]? See Also.*?(?=##|$)",
            r"#[#]? Notes.*?(?=##|$)", r"#[#]? References.*?(?=##|$)",
            r"#[#]? External links.*?(?=##|$)", r"#[#]? External Links.*?(?=##|$)",
            r"#[#]? Bibliography.*?(?=##|$)", r"#[#]? Further reading*?(?=##|$)",
            r"#[#]? Further Reading*?(?=##|$)", r"#[#]? Summary.*?(?=##|$)",
            r"#[#]? Appendices.*?(?=##|$)", r"#[#]? Appendix.*?(?=##|$)"
        ]
        for pat in patterns_to_remove:
            outline = re.sub(pat, "", outline, flags=re.DOTALL)
        outline = re.sub(r"\[.*?]", "", outline)
        return outline

    @staticmethod
    def clean_up_section(text: str) -> str:
        """Remove incomplete sentences and summary sections."""
        paragraphs = text.split("\n")
        output = []
        summary_sec_flag = False
        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
            if not p.startswith("#"):
                p = ArticleTextProcessing.remove_uncompleted_sentences_with_citations(p)
            if summary_sec_flag:
                if p.startswith("#"):
                    summary_sec_flag = False
                else:
                    continue
            if p.startswith(("Overall", "In summary", "In conclusion")):
                continue
            if "# Summary" in p or "# Conclusion" in p:
                summary_sec_flag = True
                continue
            output.append(p)
        return "\n\n".join(output)

    @staticmethod
    def update_citation_index(s: str, citation_map: Dict[int, int]) -> str:
        for original in citation_map:
            s = s.replace(f"[{original}]", f"__PLACEHOLDER_{original}__")
        for original, unify in citation_map.items():
            s = s.replace(f"__PLACEHOLDER_{original}__", f"[{unify}]")
        return s

    @staticmethod
    def parse_article_into_dict(input_string: str) -> Dict:
        lines = [line for line in input_string.split("\n") if line.strip()]
        root = {"content": "", "subsections": {}}
        current_path = [(root, -1)]
        for line in lines:
            if line.startswith("#"):
                level = line.count("#")
                title = line.strip("# ").strip()
                new_section = {"content": "", "subsections": {}}
                while current_path and current_path[-1][1] >= level:
                    current_path.pop()
                current_path[-1][0]["subsections"][title] = new_section
                current_path.append((new_section, level))
            else:
                current_path[-1][0]["content"] += line + "\n"
        return root["subsections"]


class FileIOHelper:

    @staticmethod
    def dump_json(obj: Any, file_name: str, encoding: str = "utf-8") -> None:
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, "w", encoding=encoding) as fw:
            json.dump(obj, fw, default=FileIOHelper.handle_non_serializable)

    @staticmethod
    def handle_non_serializable(obj: Any) -> str:
        return "non-serializable contents"

    @staticmethod
    def load_json(file_name: str, encoding: str = "utf-8") -> Any:
        with open(file_name, "r", encoding=encoding) as fr:
            return json.load(fr)

    @staticmethod
    def write_str(s: str, path: str, encoding: str = "utf-8") -> None:
        with open(path, "w", encoding=encoding) as f:
            f.write(s)

    @staticmethod
    def load_str(path: str, encoding: str = "utf-8") -> str:
        with open(path, "r", encoding=encoding) as f:
            return f.read()

    @staticmethod
    def dump_pickle(obj: Any, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    @staticmethod
    def load_pickle(path: str) -> Any:
        with open(path, "rb") as f:
            return pickle.load(f)


class WebPageHelper:
    """Helper class for processing web pages (inspired by Stanford Oval WikiChat)."""
    def __init__(
        self,
        min_char_count: int = 150,
        snippet_chunk_size: int = 1000,
        max_thread_num: int = 10,
    ):
        self.httpx_client = httpx.Client(verify=False)
        self.min_char_count = min_char_count
        self.max_thread_num = max_thread_num
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=snippet_chunk_size,
            chunk_overlap=0,
            length_function=len,
            is_separator_regex=False,
            separators=[
                "\n\n", "\n", ".", "\uff0e", "\u3002", ",", "\uff0c", "\u3001", " ", "\u200B", ""
            ],
        )

    def download_webpage(self, url: str) -> Optional[bytes]:
        try:
            res = self.httpx_client.get(url, timeout=4)
            if res.status_code >= 400:
                res.raise_for_status()
            return res.content
        except httpx.HTTPError as exc:
            print(f"Error while requesting {exc.request.url!r} - {exc!r}")
            return None

    def urls_to_articles(self, urls: List[str]) -> Dict[str, Dict[str, str]]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_thread_num) as executor:
            htmls = list(executor.map(self.download_webpage, urls))
        articles = {}
        for h, u in zip(htmls, urls):
            if h is None:
                continue
            article_text = extract(
                h,
                include_tables=False,
                include_comments=False,
                output_format="txt",
            )
            if article_text and len(article_text) > self.min_char_count:
                articles[u] = {"text": article_text}
        return articles

    def urls_to_snippets(self, urls: List[str]) -> Dict[str, Dict[str, List[str]]]:
        articles = self.urls_to_articles(urls)
        for u in articles:
            articles[u]["snippets"] = self.text_splitter.split_text(articles[u]["text"])
        return articles


def user_input_appropriateness_check(user_input: str) -> str:
    my_openai_model = OpenAIModel(
        model="gpt-4o-mini",
        max_tokens=200,
        temperature=0.0,
        top_p=0.9,
    )

    if len(user_input.split()) > 200:
        return "The input is too long. Please make your input topic more concise!"

    if not re.match(r'^[a-zA-Z0-9\s\-",.?\']*$', user_input):
        return ("The input contains invalid characters. The input should only contain "
                "a-z, A-Z, 0-9, space, -/\"/,./?/'.")

    prompt = (
        "Here is a topic input into a knowledge curation engine that can write a Wikipedia-like article for the topic. "
        "Please judge whether it is appropriate or not for the engine to curate information for this topic based on English search engine. "
        "The following types of inputs are inappropriate:\n"
        "1. Inputs that may be related to illegal, harmful, violent, racist, or sexual purposes.\n"
        "2. Inputs that are given using languages other than English. Currently, the engine can only support English.\n"
        "3. Inputs that are related to personal experience or personal information. Currently, the engine can only use information from the search engine.\n"
        "4. Inputs that are not aimed at topic research or inquiry. For example, asks requiring detailed execution, such as calculations, programming, or specific service searches fall outside the engine's scope of capabilities.\n"
        'If the topic is appropriate for the engine to process, output "Yes."; otherwise, output "No. The input violates reason [1/2/3/4]".\n'
        f"User input: {user_input}"
    )
    reject_reason_info = {
        1: ("Sorry, this input may be related to sensitive topics. Please try another topic. "
            "(Our input filtering uses OpenAI GPT-4o-mini, which may result in false positives. "
            "We apologize for any inconvenience.)"),
        2: ("Sorry, the current engine can only support English. Please try another topic. "
            "(Our input filtering uses OpenAI GPT-4o-mini, which may result in false positives. "
            "We apologize for any inconvenience.)"),
        3: ("Sorry, the current engine cannot process topics related to personal experience. Please try another topic. "
            "(Our input filtering uses OpenAI GPT-4o-mini, which may result in false positives. "
            "We apologize for any inconvenience.)"),
        4: ("Sorry, STORM cannot follow arbitrary instruction. Please input a topic you want to learn about. "
            "(Our input filtering uses OpenAI GPT-4o-mini, which may result in false positives. "
            "We apologize for any inconvenience.)"),
    }

    try:
        response = my_openai_model(prompt)[0].replace("[", "").replace("]", "")
        if response.startswith("No"):
            match = regex.search(r"reason\s(\d+)", response)
            if match:
                reject_reason = int(match.group(1))
                return reject_reason_info.get(reject_reason, "Sorry, the input is inappropriate. Please try another topic!")
            return "Sorry, the input is inappropriate. Please try another topic!"
    except Exception:
        return "Sorry, the input is inappropriate. Please try another topic!"
    return "Approved"


def purpose_appropriateness_check(user_input: str) -> str:
    my_openai_model = OpenAIModel(
        model="gpt-4o-mini",
        max_tokens=200,
        temperature=0.0,
        top_p=0.9,
    )

    prompt = (
        "Here is a purpose input into a report generation engine that can create a long-form report on any topic of interest. "
        "Please judge whether the provided purpose is valid for using this service. "
        "Try to judge if given purpose is non-sense like random words or just try to get around the sanity check. "
        "You should not make the rule too strict.\n\n"
        'If the purpose is valid, output "Yes."; otherwise, output "No" followed by reason.\n'
        f"User input: {user_input}"
    )
    try:
        response = my_openai_model(prompt)[0].replace("[", "").replace("]", "")
        if response.startswith("No"):
            return "Please provide a more detailed explanation on your purpose of requesting this article."
    except Exception:
        return "Please provide a more detailed explanation on your purpose of requesting this article."
    return "Approved"