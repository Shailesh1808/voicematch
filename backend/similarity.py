"""
Similarity computation and score normalization for VoiceMatch.

Computes cosine similarity between a user embedding and reference embeddings,
ranks the results, and normalizes scores via softmax into display percentages.
All numpy/math operations are confined to this module.
"""
