import os
import numpy as np
import time
import threading
import queue
from typing import List, Union, Dict, Any, Optional
import requests
import math
import random
import sys


class CleanOllamaEmbeddings:
    def __init__(self, model: str, base_url: str, name: str):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.name = name
        self.session = requests.Session()
        self.health_check()

    def health_check(self):
        """Check if the service is available"""
        try:
            resp = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def embed_query(self, text: str) -> Optional[List[float]]:
        payloads = [
            {"model": self.model, "prompt": text},
            {"model": self.model, "input": text}
        ]

        for payload in payloads:
            try:
                resp = self.session.post(
                    f"{self.base_url}/api/embed",
                    json=payload,
                    timeout=30
                )
                resp.raise_for_status()
                result = resp.json()

                # Handle different response formats
                if isinstance(result, dict):
                    if "embedding" in result and result["embedding"]:
                        return result["embedding"]
                    elif "embeddings" in result and result["embeddings"]:
                        return result["embeddings"]
                    elif "data" in result and isinstance(result["data"], list):
                        return result["data"]
                elif isinstance(result, list) and result:
                    return result
            except Exception:
                pass

        return None


class EmbeddingWorker(threading.Thread):
    """Worker thread for processing embeddings from a queue"""

    def __init__(self,
                 name: str,
                 embedding_service,
                 work_queue: queue.Queue,
                 result_dict: Dict,
                 failed_queue: queue.Queue,
                 lock: threading.Lock,
                 progress_callback=None):
        super().__init__(name=name)
        self.embedding_service = embedding_service
        self.work_queue = work_queue
        self.result_dict = result_dict
        self.failed_queue = failed_queue
        self.lock = lock
        self.daemon = True
        self.processed_count = 0
        self.progress_callback = progress_callback

    def run(self):
        """Process items from the queue until it's empty"""
        while True:
            try:
                # Get an item with a timeout to prevent blocking forever
                item = self.work_queue.get(timeout=0.1)
            except queue.Empty:
                break

            try:
                idx, text = item
                start_time = time.time()
                embedding = self.embedding_service.embed_query(text)
                elapsed_time = time.time() - start_time

                # If embedding succeeded
                if embedding is not None and (isinstance(embedding, list) and len(embedding) > 0):
                    with self.lock:
                        self.result_dict[idx] = {
                            "text": text,
                            "embedding": embedding,
                            "backend": self.embedding_service.name,
                            "time_ms": round(elapsed_time * 1000)
                        }
                    self.processed_count += 1

                    # Report progress if callback is provided
                    if self.progress_callback:
                        self.progress_callback(1, self.embedding_service.name)
                else:
                    # Put failed item in the retry queue
                    self.failed_queue.put(item)

            except Exception as e:
                # If there's an error, put the item in the retry queue
                print(f"Error in {self.name} processing '{text[:30]}...': {str(e)}")
                self.failed_queue.put(item)

            finally:
                # Mark task as done
                self.work_queue.task_done()


class Encoder:
    def __init__(self):
        model_name = os.environ.get("EMB_MODEL", "snowflake-arctic-embed2").strip()
        self._ollama_url = os.environ.get("OLLAMA_EMB2", "http://168.231.119.208:11434").strip()
        self._ollama2_url = os.environ.get("OLLAMA_EMB3", "http://69.62.78.141:11434").strip()

        self._ollama_model_name = model_name

        # Initialize embedding services
        self.embedding_services = []
        self.init_embedding_services()

        if not self.embedding_services:
            raise RuntimeError("No embedding backends available.")

    def init_embedding_services(self):
        """Initialize all available embedding services"""
        # Set up Ollama embeddings
        if self._ollama_url:
            try:
                ollama_embedding = CleanOllamaEmbeddings(
                    model=self._ollama_model_name,
                    base_url=self._ollama_url,
                    name="ollama"
                )
                self.embedding_services.append(ollama_embedding)
                print("Ollama embedding service initialized")
            except Exception as e:
                print(f"Failed to initialize Ollama embeddings: {str(e)}")

        if self._ollama2_url:
            try:
                ollama2_embedding = CleanOllamaEmbeddings(
                    model=self._ollama_model_name,
                    base_url=self._ollama2_url,
                    name="ollama2"
                )
                self.embedding_services.append(ollama2_embedding)
                print("Ollama2 embedding service initialized")
            except Exception as e:
                print(f"Failed to initialize Ollama2 embeddings: {str(e)}")


    def _dynamic_chunk_size(self, total_texts, num_services):
        """Calculate optimal chunk size based on total texts and available services"""
        # If very few texts, just distribute them
        if total_texts <= num_services * 3:
            return 1

        # Aim for a minimum of ~20 texts per worker with a ceiling
        chunk_size = min(25, max(1, total_texts // (num_services * 5)))
        return chunk_size

    def encode(self, texts: Union[str, List[str]], verbose: bool = False,
               batch_size: int = None, max_workers_per_backend: int = 3) -> Dict[int, Dict]:
        """
        Embeds texts using all available backend services in parallel.

        Args:
            texts: String or list of strings to embed
            verbose: Whether to print verbose output
            batch_size: Size of each batch (will be calculated dynamically if None)
            max_workers_per_backend: Maximum worker threads per backend

        Returns:
            Dictionary of embeddings indexed by the original list index
        """
        if isinstance(texts, str):
            texts = [texts]

        # Create queues for work distribution
        work_queue = queue.Queue()
        failed_queue = queue.Queue()

        # Dictionary to store results with thread-safe access
        results = {}
        lock = threading.Lock()

        # Total items to process
        total_items = len(texts)
        num_services = len(self.embedding_services)

        # Calculate dynamic batch size if not provided
        if batch_size is None:
            batch_size = self._dynamic_chunk_size(total_items, num_services)

        if verbose:
            print(f"Embedding {total_items} texts using {num_services} services with batch size {batch_size}")

        # Track progress
        completed = 0
        backend_counts = {s.name: 0 for s in self.embedding_services}

        def update_progress(count, backend_name):
            nonlocal completed
            nonlocal backend_counts
            with lock:
                completed += count
                backend_counts[backend_name] = backend_counts.get(backend_name, 0) + count
                if verbose and completed % 10 == 0:
                    percent = (completed / total_items) * 100
                    print(f"\rProgress: {completed}/{total_items} ({percent:.1f}%)", end="")
                    sys.stdout.flush()

        # Process in batches to avoid memory issues with large text sets
        for batch_start in range(0, total_items, batch_size * num_services):
            batch_end = min(batch_start + batch_size * num_services, total_items)
            batch_texts = texts[batch_start:batch_end]
            batch_size_actual = len(batch_texts)

            if verbose:
                print(f"\nProcessing batch {batch_start // batch_size + 1}: texts {batch_start + 1}-{batch_end}")

            # Clear the work queue
            while not work_queue.empty():
                try:
                    work_queue.get_nowait()
                    work_queue.task_done()
                except queue.Empty:
                    break

            # Populate work queue with this batch
            for i, text in enumerate(batch_texts):
                # Use original index for consistency
                orig_idx = batch_start + i
                work_queue.put((orig_idx, text))

            # Calculate texts per backend for this batch
            texts_per_backend = math.ceil(batch_size_actual / num_services)

            # First pass: Distribute work among available workers
            workers = []
            for service in self.embedding_services:
                # Create multiple workers per backend for better parallelism
                worker_count = min(max_workers_per_backend, texts_per_backend)

                for i in range(worker_count):
                    worker_name = f"{service.name}-worker-{i + 1}"
                    worker = EmbeddingWorker(
                        name=worker_name,
                        embedding_service=service,
                        work_queue=work_queue,
                        result_dict=results,
                        failed_queue=failed_queue,
                        lock=lock,
                        progress_callback=update_progress
                    )
                    workers.append(worker)
                    worker.start()

            # Wait for all workers to finish
            for worker in workers:
                worker.join()

            # Handle any failed embeddings in this batch
            failed_count = failed_queue.qsize()
            if failed_count > 0:
                if verbose:
                    print(f"\nRetrying {failed_count} failed embeddings")

                # Create a separate queue for retries
                retry_queue = queue.Queue()

                # Move failed items to retry queue
                while not failed_queue.empty():
                    retry_queue.put(failed_queue.get())

                # Create retry workers with shuffled services
                retry_workers = []
                retry_services = list(self.embedding_services)
                random.shuffle(retry_services)

                for service in retry_services:
                    worker_name = f"retry-{service.name}"
                    worker = EmbeddingWorker(
                        name=worker_name,
                        embedding_service=service,
                        work_queue=retry_queue,
                        result_dict=results,
                        failed_queue=failed_queue,
                        lock=lock,
                        progress_callback=update_progress
                    )
                    retry_workers.append(worker)
                    worker.start()

                for worker in retry_workers:
                    worker.join()

        # Final failed count
        final_failed = failed_queue.qsize()

        # Print summary if verbose
        if verbose:
            print("\n")  # Ensure we're on a new line after progress reporting
            success_count = len(results)
            print(f"\nEmbedding Summary:")
            print(f"- Total texts: {total_items}")
            print(f"- Successfully embedded: {success_count}")
            print(f"- Failed: {total_items - success_count}")

            if results:
                # Get a sample embedding to analyze its shape
                sample_idx = next(iter(results))
                sample = results[sample_idx]["embedding"]

                if isinstance(sample, list):
                    if isinstance(sample[0], list):
                        # Handle case where embedding is [[...]] (2D array)
                        dims = f"{len(sample)}x{len(sample[0])}"
                    else:
                        # Handle case where embedding is [...] (1D array)
                        dims = f"{len(sample)}"
                    print(f"- Embedding dimensions: {dims}")

                # Show which backend processed what
                print("- Backend distribution:")
                for backend, count in backend_counts.items():
                    if count > 0:
                        print(f"  • {backend}: {count} texts ({count / total_items * 100:.1f}%)")

        return results


def get_encoder():
    """Singleton pattern to get or create an encoder instance"""
    if not hasattr(get_encoder, 'instance'):
        get_encoder.instance = Encoder()
    return get_encoder.instance


def main():
    # Set up embedding endpoints
    os.environ["EMB_MODEL"] = "snowflake-arctic-embed2"
    encoder = get_encoder()

    # Generate more test texts for a bigger batch
    base_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Hello world!",
        "Ollama embeddings test input.",
        "Pack my box with five dozen liquor jugs.",
        "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
    ]

    # Create a larger dataset to test batching (111 texts total)
    texts = []
    for i in range(22):  # 22 * 5 = 110 + 1 = 111
        for text in base_texts:
            texts.append(f"{text} (Batch {i + 1})")
    texts.append("Final test text")

    print(f"Embedding {len(texts)} texts...")
    results = encoder.encode(texts, verbose=True, max_workers_per_backend=3)

    # Display a sample of results
    print("\nSample of results:")
    sample_indices = list(range(0, min(len(texts), 111), len(texts) // 10))

    for i in sample_indices:
        text = texts[i]
        print(f"\nText {i + 1}: '{text}'")
        if i in results:
            res = results[i]
            embedding = res["embedding"]
            backend = res["backend"]
            time_ms = res["time_ms"]

            if isinstance(embedding, list):
                if isinstance(embedding[0], list):
                    # Handle 2D array
                    shape = f"{len(embedding)}x{len(embedding[0])}"
                    sample = embedding[0][:5] if embedding[0] else []
                else:
                    # Handle 1D array
                    shape = f"{len(embedding)}"
                    sample = embedding[:5]
            else:
                shape = "unknown"
                sample = "N/A"

            print(f"  Backend: {backend}")
            print(f"  Time: {time_ms}ms")
            print(f"  Shape: {shape}")
            print(f"  Sample: {sample}")
        else:
            print("  ❌ Failed to embed")


if __name__ == "__main__":
    main()