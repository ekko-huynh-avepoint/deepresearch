import numpy as np
from typing import Union, List, Dict, Any, Optional


def ensure_embedding_format(embedding: Union[List[float], Dict[str, Any], Dict[int, Dict]]) -> List[float]:
    """
    Ensures that embeddings are in the correct format for cosine similarity calculation.
    Converts dictionaries or nested structures to flat float lists.

    Args:
        embedding: An embedding in any format (list, dict with embedding key, etc.)

    Returns:
        A flat list of floats suitable for cosine similarity
    """
    # Handle None case
    if embedding is None:
        return [0.0]  # Return default embedding with single dimension

    # If it's already a list of numbers, just return it
    if isinstance(embedding, (list, np.ndarray)) and embedding and isinstance(embedding[0], (int, float)):
        return embedding

    # If it's a dict with "embedding" key
    if isinstance(embedding, dict):
        if "embedding" in embedding:
            result = embedding["embedding"]
            # Make sure it's a flat list
            if isinstance(result, list) and result and isinstance(result[0], (int, float)):
                return result
            elif isinstance(result, list) and result and isinstance(result[0], list):
                return result[0]  # Take first element if it's a nested list

        # If it has a text/embedding structure
        if "text" in embedding and "embedding" in embedding:
            result = embedding["embedding"]
            if isinstance(result, list) and result:
                return result

        # Try to find any value that looks like an embedding
        for v in embedding.values():
            if isinstance(v, list) and v and isinstance(v[0], (int, float)):
                return v

    # Handle index->result structure from encoder.encode()
    if isinstance(embedding, dict) and all(isinstance(k, int) for k in embedding.keys()):
        # Take any value with an embedding field
        for v in embedding.values():
            if isinstance(v, dict) and "embedding" in v:
                result = v["embedding"]
                if isinstance(result, list) and result:
                    return result

    # Return default embedding as fallback instead of empty list
    return [0.0]  # Default embedding with single dimension


def safe_cosine_similarity(queries: List, documents: List) -> np.ndarray:
    """
    A safe version of cosine similarity that handles problematic inputs

    Args:
        queries: List of query embeddings
        documents: List of document embeddings

    Returns:
        Similarity scores
    """
    from sklearn.metrics.pairwise import cosine_similarity

    # Fix query embeddings
    fixed_queries = []
    for query in queries:
        fixed_query = ensure_embedding_format(query)
        if fixed_query:  # Only add non-empty embeddings
            fixed_queries.append(fixed_query)

    # If no valid queries, return zeros
    if not fixed_queries:
        return np.zeros((1, len(documents)))

    # Fix document embeddings
    fixed_docs = []
    for doc in documents:
        fixed_doc = ensure_embedding_format(doc)
        if fixed_doc:  # Only add non-empty embeddings
            fixed_docs.append(fixed_doc)

    # If no valid documents, return zeros
    if not fixed_docs:
        return np.zeros((len(fixed_queries), 1))

    # Make sure all embeddings have the same dimension
    max_dim = max(len(emb) for emb in fixed_queries + fixed_docs)

    # Pad shorter embeddings with zeros
    fixed_queries = [emb + [0.0] * (max_dim - len(emb)) for emb in fixed_queries]
    fixed_docs = [emb + [0.0] * (max_dim - len(emb)) for emb in fixed_docs]

    # Now it's safe to compute cosine similarity
    return cosine_similarity(fixed_queries, fixed_docs)