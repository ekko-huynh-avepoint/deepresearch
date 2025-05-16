import os
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union
from langchain_ollama import OllamaEmbeddings



class Encoder:
    def __init__(self):
        self._ollama_url = os.environ.get("OLLAMA_EMB", "").strip()
        self._ollama_model_name = os.environ.get("EMB_MODEL", "").strip()
        self._ollama2_url = os.environ.get("OLLAMA_EMB2", "").strip()
        self._ollama2_model_name = os.environ.get("EMB_MODEL", "").strip()
        self._ollama = None
        self._ollama2 = None

        if not (self._ollama_url and self._ollama_model_name) and not (
                self._ollama2_url and self._ollama2_model_name):
            raise RuntimeError("No embedding backend configured (Ollama or Ollama2).")

    @property
    def ollama(self):
        if self._ollama is None and self._ollama_url and self._ollama_model_name:
            self._ollama = OllamaEmbeddings(
                model=self._ollama_model_name,
                base_url=self._ollama_url
            )
        return self._ollama

    @property
    def ollama2(self):
        if self._ollama2 is None and self._ollama2_url and self._ollama2_model_name:
            self._ollama2 = OllamaEmbeddings(
                model=self._ollama2_model_name,
                base_url=self._ollama2_url
            )
        return self._ollama2

    def encode(
            self, texts: Union[str, List[str]], max_workers: int = 8, verbose: bool = False
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        n = len(texts)
        results = [None] * n

        def embed_ollama(text):
            start = time.time()
            embed = self.ollama.embed_query(text)
            count = time.time() - start
            if verbose:
                print(f"Ollama ({self._ollama_model_name} @ {self._ollama_url}) took {count:.3f}s for: {text[:40]}...")
            return embed, count

        def embed_ollama2(text):
            start = time.time()
            embed = self.ollama2.embed_query(text)
            count = time.time() - start
            if verbose:
                print(
                    f"Ollama2 ({self._ollama2_model_name} @ {self._ollama2_url}) took {count:.3f}s for: {text[:40]}...")
            return embed, count

        temp_results = {i: {} for i in range(n)}
        time_results = {i: {} for i in range(n)}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            # Only submit jobs for available backends
            if self.ollama is not None:
                for i, text in enumerate(texts):
                    futures[executor.submit(embed_ollama, text)] = (i, "ollama")
            if self.ollama2 is not None:
                for i, text in enumerate(texts):
                    futures[executor.submit(embed_ollama2, text)] = (i, "ollama2")
            for fut in as_completed(futures):
                i, backend = futures[fut]
                try:
                    emb, elapsed = fut.result()
                    temp_results[i][backend] = emb
                    time_results[i][backend] = elapsed
                except Exception as e:
                    print(f"Embedding failed for index {i} ({backend}): {e}")
                    temp_results[i][backend] = None
                    time_results[i][backend] = None

        for i in range(n):
            embeddings = []
            for backend in ("ollama", "ollama2"):
                emb = temp_results[i].get(backend)
                if emb is not None:
                    embeddings.append(np.array(emb))
            if embeddings:
                results[i] = np.concatenate(embeddings)
            else:
                results[i] = None

        if verbose:
            for i in range(n):
                print(
                    f"Input {i}: "
                    f"Ollama time = {time_results[i].get('ollama')}, "
                    f"Ollama2 time = {time_results[i].get('ollama2')}"
                )
        return np.array(results)


_ENCODER_INSTANCE = None

def get_encoder():
    global _ENCODER_INSTANCE
    if _ENCODER_INSTANCE is None:
        _ENCODER_INSTANCE = Encoder()
    return _ENCODER_INSTANCE