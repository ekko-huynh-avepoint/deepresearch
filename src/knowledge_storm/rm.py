import logging
import os
from typing import Callable, Union, List

import backoff
import dspy
import requests

from dsp import backoff_hdlr, giveup_hdlr

from .utils import WebPageHelper

class BingSearch(dspy.Retrieve):
    def __init__(
        self,
        bing_search_api_key=None,
        k=3,
        is_valid_source: Callable = None,
        min_char_count: int = 150,
        snippet_chunk_size: int = 1000,
        webpage_helper_max_threads=10,
        mkt="en-US",
        language="en",
        **kwargs,
    ):
        """
        Params:
            min_char_count: Minimum character count for the article to be considered valid.
            snippet_chunk_size: Maximum character count for each snippet.
            webpage_helper_max_threads: Maximum number of threads to use for webpage helper.
            mkt, language, **kwargs: Bing search API parameters.
            - Reference: https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/reference/query-parameters
        """
        super().__init__(k=k)
        if not bing_search_api_key and not os.environ.get("BING_SEARCH_API_KEY"):
            raise RuntimeError(
                "You must supply bing_search_subscription_key or set environment variable BING_SEARCH_API_KEY"
            )
        elif bing_search_api_key:
            self.bing_api_key = bing_search_api_key
        else:
            self.bing_api_key = os.environ["BING_SEARCH_API_KEY"]
        self.endpoint = "https://api.bing.microsoft.com/v7.0/search"
        self.params = {"mkt": mkt, "setLang": language, "count": k, **kwargs}
        self.webpage_helper = WebPageHelper(
            min_char_count=min_char_count,
            snippet_chunk_size=snippet_chunk_size,
            max_thread_num=webpage_helper_max_threads,
        )
        self.usage = 0

        # If not None, is_valid_source shall be a function that takes a URL and returns a boolean.
        if is_valid_source:
            self.is_valid_source = is_valid_source
        else:
            self.is_valid_source = lambda x: True

    def get_usage_and_reset(self):
        usage = self.usage
        self.usage = 0

        return {"BingSearch": usage}

    def forward(
        self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = []
    ):
        """Search with Bing for self.k top passages for query or queries

        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.
            exclude_urls (List[str]): A list of urls to exclude from the search results.

        Returns:
            a list of Dicts, each dict has keys of 'description', 'snippets' (list of strings), 'title', 'url'
        """
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        self.usage += len(queries)

        url_to_results = {}

        headers = {"Ocp-Apim-Subscription-Key": self.bing_api_key}

        for query in queries:
            try:
                results = requests.get(
                    self.endpoint, headers=headers, params={**self.params, "q": query}
                ).json()

                for d in results["webPages"]["value"]:
                    if self.is_valid_source(d["url"]) and d["url"] not in exclude_urls:
                        url_to_results[d["url"]] = {
                            "url": d["url"],
                            "title": d["name"],
                            "description": d["snippet"],
                        }
            except Exception as e:
                logging.error(f"Error occurs when searching query {query}: {e}")

        valid_url_to_snippets = self.webpage_helper.urls_to_snippets(
            list(url_to_results.keys())
        )
        collected_results = []
        for url in valid_url_to_snippets:
            r = url_to_results[url]
            r["snippets"] = valid_url_to_snippets[url]["snippets"]
            collected_results.append(r)

        return collected_results

class VectorRM(dspy.Retrieve):
    def __init__(
        self,
        collection_name: str,
        qdrant_vs,            # Qdrant vector store (langchain_qdrant.Qdrant instance)
        embed_model,
        k: int = 3,
    ):
        super().__init__(k=k)
        self.usage = 0
        if not collection_name:
            raise ValueError("Please provide a collection name.")
        if not embed_model:
            raise ValueError("Please provide an embedding model.")
        if not qdrant_vs:
            raise ValueError("Please provide a Qdrant vector store instance.")

        self.collection_name = collection_name
        self.model = embed_model
        self.qdrant = qdrant_vs

    def get_usage_and_reset(self):
        usage = self.usage
        self.usage = 0
        return {"VectorRM": usage}

    def get_vector_count(self):
        return self.qdrant.client.count(collection_name=self.collection_name)

    def forward(self, query_or_queries: Union[str, List[str]], exclude_urls: List[str]):
        """
        Search in your data for self.k top passages for query or queries.

        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.
            exclude_urls (List[str]): Dummy parameter to match the interface. Does not have any effect.

        Returns:
            a list of Dicts, each dict has keys of 'description', 'snippets' (list of strings), 'title', 'url'
        """
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        self.usage += len(queries)
        collected_results = []
        for query in queries:
            related_docs = self.qdrant.similarity_search_with_score(query, k=self.k)
            for i in range(len(related_docs)):
                doc = related_docs[i][0]
                collected_results.append(
                    {
                        "description": doc.metadata.get("description", ""),
                        "snippets": [doc.page_content],
                        "title": doc.metadata.get("title", ""),
                        "url": doc.metadata.get("url", ""),
                    }
                )
        return collected_results


class StanfordOvalArxivRM(dspy.Retrieve):
    """[Alpha] This retrieval class is for internal use only, not intended for the public."""

    def __init__(self, endpoint, k=3, rerank=True):
        super().__init__(k=k)
        self.endpoint = endpoint
        self.usage = 0
        self.rerank = rerank

    def get_usage_and_reset(self):
        usage = self.usage
        self.usage = 0

        return {"StanfordOvalArxivRM": usage}

    def _retrieve(self, query: str):
        payload = {"query": query, "num_blocks": self.k, "rerank": self.rerank}

        response = requests.post(
            self.endpoint, json=payload, headers={"Content-Type": "application/json"}
        )

        # Check if the request was successful
        if response.status_code == 200:
            response_data_list = response.json()[0]["results"]
            results = []
            for response_data in response_data_list:
                result = {
                    "title": response_data["document_title"],
                    "url": response_data["url"],
                    "snippets": [response_data["content"]],
                    "description": response_data.get("description", "N/A"),
                    "meta": {
                        key: value
                        for key, value in response_data.items()
                        if key not in ["document_title", "url", "content"]
                    },
                }

                results.append(result)

            return results
        else:
            raise Exception(
                f"Error: Unable to retrieve results. Status code: {response.status_code}"
            )

    def forward(
        self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = []
    ):
        collected_results = []
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )

        for query in queries:
            try:
                results = self._retrieve(query)
                collected_results.extend(results)
            except Exception as e:
                logging.error(f"Error occurs when searching query {query}: {e}")
        return collected_results


class BraveRM(dspy.Retrieve):
    def __init__(
        self, brave_search_api_key=None, k=3, is_valid_source: Callable = None
    ):
        super().__init__(k=k)
        if not brave_search_api_key and not os.environ.get("BRAVE_API_KEY"):
            raise RuntimeError(
                "You must supply brave_search_api_key or set environment variable BRAVE_API_KEY"
            )
        elif brave_search_api_key:
            self.brave_search_api_key = brave_search_api_key
        else:
            self.brave_search_api_key = os.environ["BRAVE_API_KEY"]
        self.usage = 0

        # If not None, is_valid_source shall be a function that takes a URL and returns a boolean.
        if is_valid_source:
            self.is_valid_source = is_valid_source
        else:
            self.is_valid_source = lambda x: True

    def get_usage_and_reset(self):
        usage = self.usage
        self.usage = 0

        return {"BraveRM": usage}

    def forward(
        self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = []
    ):
        """Search with api.search.brave.com for self.k top passages for query or queries

        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.
            exclude_urls (List[str]): A list of urls to exclude from the search results.

        Returns:
            a list of Dicts, each dict has keys of 'description', 'snippets' (list of strings), 'title', 'url'
        """
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        self.usage += len(queries)
        collected_results = []
        for query in queries:
            try:
                headers = {
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": self.brave_search_api_key,
                }
                response = requests.get(
                    f"https://api.search.brave.com/res/v1/web/search?result_filter=web&q={query}",
                    headers=headers,
                ).json()
                results = response.get("web", {}).get("results", [])

                for result in results:
                    collected_results.append(
                        {
                            "snippets": result.get("extra_snippets", []),
                            "title": result.get("title"),
                            "url": result.get("url"),
                            "description": result.get("description"),
                        }
                    )
            except Exception as e:
                logging.error(f"Error occurs when searching query {query}: {e}")

        return collected_results



class DuckDuckGoSearchRM(dspy.Retrieve):
    """Retrieve information from custom queries using DuckDuckGo."""

    def __init__(
        self,
        k: int = 3,
        is_valid_source: Callable = None,
        min_char_count: int = 150,
        snippet_chunk_size: int = 1000,
        webpage_helper_max_threads=10,
        safe_search: str = "On",
        region: str = "us-en",
    ):
        """
        Params:
            min_char_count: Minimum character count for the article to be considered valid.
            snippet_chunk_size: Maximum character count for each snippet.
            webpage_helper_max_threads: Maximum number of threads to use for webpage helper.
            **kwargs: Additional parameters for the OpenAI API.
        """
        super().__init__(k=k)
        try:
            from duckduckgo_search import DDGS
        except ImportError as err:
            raise ImportError(
                "Duckduckgo requires `pip install duckduckgo_search`."
            ) from err
        self.k = k
        self.webpage_helper = WebPageHelper(
            min_char_count=min_char_count,
            snippet_chunk_size=snippet_chunk_size,
            max_thread_num=webpage_helper_max_threads,
        )
        self.usage = 0
        # All params for search can be found here:
        #   https://duckduckgo.com/duckduckgo-help-pages/settings/params/

        # Sets the backend to be api
        self.duck_duck_go_backend = "api"

        # Only gets safe search results
        self.duck_duck_go_safe_search = safe_search

        # Specifies the region that the search will use
        self.duck_duck_go_region = region

        # If not None, is_valid_source shall be a function that takes a URL and returns a boolean.
        if is_valid_source:
            self.is_valid_source = is_valid_source
        else:
            self.is_valid_source = lambda x: True

        # Import the duckduckgo search library found here: https://github.com/deedy5/duckduckgo_search
        self.ddgs = DDGS()

    def get_usage_and_reset(self):
        usage = self.usage
        self.usage = 0
        return {"DuckDuckGoRM": usage}

    @backoff.on_exception(
        backoff.expo,
        (Exception,),
        max_time=1000,
        max_tries=8,
        on_backoff=backoff_hdlr,
        giveup=giveup_hdlr,
    )
    def request(self, query: str):
        results = self.ddgs.text(
            query, max_results=self.k, backend=self.duck_duck_go_backend
        )
        return results

    def forward(
        self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = []
    ):
        """Search with DuckDuckGoSearch for self.k top passages for query or queries
        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.
            exclude_urls (List[str]): A list of urls to exclude from the search results.
        Returns:
            a list of Dicts, each dict has keys of 'description', 'snippets' (list of strings), 'title', 'url'
        """
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        self.usage += len(queries)

        collected_results = []

        for query in queries:
            #  list of dicts that will be parsed to return
            results = self.request(query)

            for d in results:
                # assert d is dict
                if not isinstance(d, dict):
                    print(f"Invalid result: {d}\n")
                    continue

                try:
                    # ensure keys are present
                    url = d.get("href", None)
                    title = d.get("title", None)
                    description = d.get("description", title)
                    snippets = [d.get("body", None)]

                    # raise exception of missing key(s)
                    if not all([url, title, description, snippets]):
                        raise ValueError(f"Missing key(s) in result: {d}")
                    if self.is_valid_source(url) and url not in exclude_urls:
                        result = {
                            "url": url,
                            "title": title,
                            "description": description,
                            "snippets": snippets,
                        }
                        collected_results.append(result)
                    else:
                        print(f"invalid source {url} or url in exclude_urls")
                except Exception as e:
                    print(f"Error occurs when processing {result=}: {e}\n")
                    print(f"Error occurs when searching query {query}: {e}")

        return collected_results

class SearXNG(dspy.Retrieve):
    def __init__(
        self,
        searxng_api_url: str,
        k: int = 3,
        is_valid_source: Callable[[str], bool] = None,
        disabled_engines: List[str] = None,
    ):
        super().__init__(k=k)
        if not searxng_api_url:
            raise RuntimeError("You must supply searxng_api_url")
        self.searxng_api_url = searxng_api_url
        self.usage = 0
        self.k = k  # cache k for use in forward
        self.disabled_engines = set(disabled_engines or [])
        self.is_valid_source = is_valid_source or (lambda _: True)

    def get_usage_and_reset(self):
        usage = self.usage
        self.usage = 0
        return {"SearXNG": usage}

    def forward(
            self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = None
    ):
        # Ensure queries is not empty
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        # Remove empty/whitespace-only queries
        queries = [q for q in queries if q and q.strip()]
        if not queries:
            logging.info("No valid queries provided to the SearXNG forward method. Search will not be executed.")
            return []  # Return early if no valid queries are provided

        exclude_urls = set(exclude_urls or [])
        self.usage += len(queries)
        collected_results = []
        params_base = {"format": "json"}

        for query in queries:
            params = params_base.copy()
            params["q"] = query
            if self.disabled_engines:
                params["disabled_engines"] = ",".join(self.disabled_engines)
            try:
                response = requests.post(self.searxng_api_url, params=params)
                response.raise_for_status()
                results = response.json()
                count = 0
                for r in results.get("results", []):
                    engine = r.get("engine")
                    url = r.get("url")
                    if (
                            count < self.k
                            and url
                            and (not engine or engine not in self.disabled_engines)
                            and self.is_valid_source(url)
                            and url not in exclude_urls
                    ):
                        content = r.get("content", "")
                        collected_results.append(
                            {
                                "description": content,
                                "snippets": [content],
                                "title": r.get("title", ""),
                                "url": url,
                                "engine": engine,
                            }
                        )
                        count += 1
                    if count >= self.k:
                        break
            except Exception as e:
                logging.error(f"SearXNG search error for query '{query}': {e}")

        return collected_results

class FirecrawlRM(dspy.Retrieve):
    def __init__(self, search_link=None, k=5, is_valid_source: Callable = None):
        """Initialize the Firecrawl search retriever.
        
        Args:
            search_link (str, optional): The URL for the Firecrawl API. If not provided,
                                         it will be read from the LINK_SEARCH environment variable.
            k (int, optional): Number of results to return. Defaults to 5.
            is_valid_source (Callable, optional): Function that takes a URL and returns a boolean
                                                 indicating if the source is valid.
        """
        super().__init__(k=k)
        if not search_link and not os.environ.get("LINK_SEARCH"):
            raise RuntimeError(
                "You must supply search_link or set environment variable LINK_SEARCH"
            )
        self.search_link = search_link or os.environ.get("LINK_SEARCH")
        self.usage = 0
        
        # If not None, is_valid_source shall be a function that takes a URL and returns a boolean
        if is_valid_source:
            self.is_valid_source = is_valid_source
        else:
            self.is_valid_source = lambda x: True

    def get_usage_and_reset(self):
        usage = self.usage
        self.usage = 0
        return {"FirecrawlRM": usage}
    
    def search_firecrawl(self, query, limit=5):
        """Search Firecrawl API with the given query."""
        url = f"{self.search_link}/v1/search"
        headers = {"Content-Type": "application/json"}
        payload = {
            "query": query,
            "limit": limit
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Request error: {e}")
            if hasattr(e, 'response') and e.response:
                logging.error(f"Status code: {e.response.status_code}")
                logging.error(f"Response text: {e.response.text}")
            return None

    def forward(self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = []):
        """Search with Firecrawl for self.k top passages for query or queries.
        
        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.
            exclude_urls (List[str]): A list of urls to exclude from the search results.
            
        Returns:
            a list of Dicts, each dict has keys of 'description', 'snippets' (list of strings), 'title', 'url'
        """
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        self.usage += len(queries)
        
        collected_results = []
        for query in queries:
            try:
                results = self.search_firecrawl(query, self.k)
                
                # Process results based on Firecrawl's API response format
                if results and "results" in results:
                    for item in results["results"]:
                        if "url" in item and self.is_valid_source(item["url"]) and item["url"] not in exclude_urls:
                            collected_results.append({
                                "url": item.get("url", ""),
                                "title": item.get("title", ""),
                                "description": item.get("description", item.get("title", "")),
                                "snippets": [item.get("content", "")]
                            })
                        
            except Exception as e:
                logging.error(f"Error occurs when searching query {query}: {e}")
                
        return collected_results