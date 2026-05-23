"""
Similarity computation and score normalization for VoiceMatch.

Computes cosine similarity between a user embedding and reference embeddings,
ranks the results, and normalizes scores via softmax into display percentages.
All numpy/math operations are confined to this module.
"""

import numpy as np

SOFTMAX_TEMPERATURE = 1.0


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the cosine similarity between two L2-normalised vectors.

    Because both inputs are unit vectors, the cosine similarity reduces to
    the dot product. The result is clamped to [-1.0, 1.0] to guard against
    floating-point rounding errors that could produce values like 1.0000001.

    Parameters
    ----------
    a : np.ndarray
        First L2-normalised embedding vector.
    b : np.ndarray
        Second L2-normalised embedding vector of the same shape.

    Returns
    -------
    float
        Cosine similarity in the range [-1.0, 1.0].
    """
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def softmax(scores: np.ndarray) -> np.ndarray:
    """Apply numerically stable softmax normalisation to an array of scores.

    Subtracts the maximum score before exponentiating to prevent overflow
    when any score is very large. The output sums to 1.0 and can be
    multiplied by 100 to produce display percentages.

    Parameters
    ----------
    scores : np.ndarray
        Raw similarity scores (any shape, typically 1-D).

    Returns
    -------
    np.ndarray
        Array of the same shape whose values sum to 1.0.
    """
    exp_scores = np.exp(scores - np.max(scores))
    return exp_scores / exp_scores.sum()


def rank_results(
    user_embedding: np.ndarray,
    reference_embeddings: dict,
    vocalist_metadata: list,
) -> list:
    """Rank all reference vocalists by similarity to the user embedding.

    Computes cosine similarity between the user embedding and every reference
    embedding, converts the raw scores to display percentages via softmax,
    then returns the full list sorted from most to least similar with 1-based
    ranks assigned.

    Parameters
    ----------
    user_embedding : np.ndarray
        L2-normalised embedding of shape (123,) for the user recording.
    reference_embeddings : dict[str, np.ndarray]
        Mapping of vocalist ID (e.g. "sample_1") to its L2-normalised
        embedding. Order must be consistent with vocalist_metadata.
    vocalist_metadata : list[dict]
        List of dicts with keys "id", "name", and "style" for each vocalist,
        in the same order as reference_embeddings.

    Returns
    -------
    list[dict]
        Sorted list of result dicts, each containing:
          - "rank"       : int   — 1-based rank (1 = most similar)
          - "id"         : str   — vocalist ID (e.g. "sample_1")
          - "name"       : str   — vocalist display name (e.g. "Luna")
          - "style"      : str   — vocal style (e.g. "Alto, Soul")
          - "similarity" : float — raw cosine score rounded to 4 decimal places
          - "percentage" : float — softmax percentage rounded to 1 decimal place
    """
    ids = list(reference_embeddings.keys())
    raw_scores = np.array([
        cosine_similarity(user_embedding, reference_embeddings[vid])
        for vid in ids
    ])

    percentages = softmax(raw_scores * SOFTMAX_TEMPERATURE) * 100

    meta_by_id = {m["id"]: m for m in vocalist_metadata}

    unsorted = [
        {
            "id": vid,
            "name": meta_by_id[vid]["name"],
            "style": meta_by_id[vid]["style"],
            "similarity": round(float(raw_scores[i]), 4),
            "percentage": round(float(percentages[i]), 1),
        }
        for i, vid in enumerate(ids)
    ]

    sorted_results = sorted(unsorted, key=lambda r: r["similarity"], reverse=True)

    for rank, result in enumerate(sorted_results, start=1):
        result["rank"] = rank

    return sorted_results


def get_top_match(results: list) -> str:
    """Return the name of the highest-ranked vocalist.

    Parameters
    ----------
    results : list[dict]
        Sorted output of rank_results, with rank 1 at index 0.

    Returns
    -------
    str
        Display name of the rank-1 vocalist (e.g. "Luna").
    """
    return results[0]["name"]
