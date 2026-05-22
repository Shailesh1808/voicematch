import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from backend.similarity import (
    cosine_similarity,
    get_top_match,
    rank_results,
    softmax,
)


class TestCosineSimilarity:

    def test_identical_vectors_return_one(self):
        v = np.array([1.0, 0.0, 0.0])
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_return_zero(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors_return_minus_one(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([-1.0, 0.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_result_clamped_to_valid_range(self):
        v = np.array([1.0 + 1e-10, 0.0, 0.0])
        result = cosine_similarity(v, v)
        assert result <= 1.0
        assert result >= -1.0

    def test_returns_float(self):
        v = np.array([1.0, 0.0, 0.0])
        result = cosine_similarity(v, v)
        assert isinstance(result, float)


class TestSoftmax:

    def test_output_sums_to_one(self):
        scores = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
        result = softmax(scores)
        assert result.sum() == pytest.approx(1.0)

    def test_highest_score_gets_highest_probability(self):
        scores = np.array([0.9, 0.5, 0.3])
        result = softmax(scores)
        assert result[0] == max(result)

    def test_output_shape_matches_input(self):
        scores = np.array([0.9, 0.7, 0.5, 0.3, 0.1, 0.0, -0.1, -0.3, -0.5, -0.7])
        result = softmax(scores)
        assert result.shape == scores.shape

    def test_all_equal_scores_give_uniform_distribution(self):
        scores = np.array([0.5, 0.5, 0.5, 0.5])
        result = softmax(scores)
        expected = np.array([0.25, 0.25, 0.25, 0.25])
        np.testing.assert_array_almost_equal(result, expected)

    def test_numerically_stable_with_large_scores(self):
        scores = np.array([1000.0, 999.0, 998.0])
        result = softmax(scores)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        assert result.sum() == pytest.approx(1.0)


class TestRankResults:

    def setup_method(self):
        self.embeddings = {
            "sample_1": np.array([1.0, 0.0, 0.0]),
            "sample_2": np.array([0.0, 1.0, 0.0]),
            "sample_3": np.array([0.0, 0.0, 1.0]),
        }
        self.metadata = [
            {"id": "sample_1", "name": "Luna",  "style": "Alto, Soul"},
            {"id": "sample_2", "name": "Atlas", "style": "Baritone, Folk"},
            {"id": "sample_3", "name": "Echo",  "style": "Soprano, Pop"},
        ]

    def test_returns_correct_number_of_results(self):
        user_emb = np.array([1.0, 0.0, 0.0])
        results = rank_results(user_emb, self.embeddings, self.metadata)
        assert len(results) == 3

    def test_top_result_is_most_similar(self):
        user_emb = np.array([1.0, 0.0, 0.0])
        results = rank_results(user_emb, self.embeddings, self.metadata)
        assert results[0]["name"] == "Luna"
        assert results[0]["rank"] == 1

    def test_ranks_are_sequential(self):
        user_emb = np.array([1.0, 0.0, 0.0])
        results = rank_results(user_emb, self.embeddings, self.metadata)
        ranks = [r["rank"] for r in results]
        assert ranks == [1, 2, 3]

    def test_percentages_sum_to_100(self):
        user_emb = np.array([1.0, 0.0, 0.0])
        results = rank_results(user_emb, self.embeddings, self.metadata)
        total = sum(r["percentage"] for r in results)
        assert abs(total - 100.0) < 0.2

    def test_result_contains_required_fields(self):
        user_emb = np.array([1.0, 0.0, 0.0])
        results = rank_results(user_emb, self.embeddings, self.metadata)
        required = {"rank", "id", "name", "style", "similarity", "percentage"}
        for result in results:
            assert required.issubset(result.keys())

    def test_similarity_scores_are_sorted_descending(self):
        user_emb = np.array([1.0, 0.0, 0.0])
        results = rank_results(user_emb, self.embeddings, self.metadata)
        similarities = [r["similarity"] for r in results]
        assert similarities == sorted(similarities, reverse=True)

    def test_with_real_audio_files(self):
        from backend.audio_processor import extract_embedding
        user_emb = extract_embedding("reference_data/sample_1.wav")
        ref_embs = {
            "sample_1": extract_embedding("reference_data/sample_1.wav"),
            "sample_2": extract_embedding("reference_data/sample_2.wav"),
            "sample_3": extract_embedding("reference_data/sample_3.wav"),
        }
        metadata = [
            {"id": "sample_1", "name": "Luna",  "style": "Alto, Soul"},
            {"id": "sample_2", "name": "Atlas", "style": "Baritone, Folk"},
            {"id": "sample_3", "name": "Echo",  "style": "Soprano, Pop"},
        ]
        results = rank_results(user_emb, ref_embs, metadata)
        assert results[0]["name"] == "Luna"
        assert results[0]["similarity"] == pytest.approx(1.0, abs=1e-4)


class TestGetTopMatch:

    def test_returns_name_of_rank_one(self):
        results = [
            {"rank": 1, "name": "Luna",  "similarity": 0.9},
            {"rank": 2, "name": "Atlas", "similarity": 0.7},
            {"rank": 3, "name": "Echo",  "similarity": 0.5},
        ]
        assert get_top_match(results) == "Luna"

    def test_returns_string(self):
        results = [{"rank": 1, "name": "Luna", "similarity": 0.9}]
        assert isinstance(get_top_match(results), str)
