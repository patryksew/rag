import pytest
from unittest.mock import MagicMock
from src.retriever import EmbeddingRetriever, HybridRetriever
from src.models import (
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults,
    UnansweredQuestion,
)


class TestEmbeddingRetriever:
    @pytest.fixture
    def mock_collection(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def retriever(self, mock_collection: MagicMock) -> EmbeddingRetriever:
        return EmbeddingRetriever(collection=mock_collection)

    def test_retrieve_one_normal_case(
        self, retriever: EmbeddingRetriever, mock_collection: MagicMock
    ) -> None:
        mock_collection.query.return_value = {
            "metadatas": [
                [
                    {
                        "file_path": "a.py",
                        "first_character_index": 0,
                        "last_character_index": 100
                    },
                    {
                        "file_path": "b.py",
                        "first_character_index": 50,
                        "last_character_index": 150
                    }
                ]
            ]
        }

        res = retriever.retrieve_one("find class A", k=2)

        assert isinstance(res, MinimalSearchResults)
        assert res.question == "find class A"
        assert len(res.retrieved_sources) == 2
        assert res.retrieved_sources[0].file_path == "a.py"
        assert res.retrieved_sources[1].file_path == "b.py"

        mock_collection.query.assert_called_once_with(
            query_texts=["find class A"],
            n_results=2
        )

    def test_retrieve_one_empty_metadata(
        self, retriever: EmbeddingRetriever, mock_collection: MagicMock
    ) -> None:
        mock_collection.query.return_value = {
            "metadatas": [[]]
        }

        res = retriever.retrieve_one("impossible query", k=5)

        assert len(res.retrieved_sources) == 0

    def test_retrieve_multiple_normal_case(
        self, retriever: EmbeddingRetriever, mock_collection: MagicMock
    ) -> None:
        mock_collection.query.return_value = {
            "metadatas": [
                [
                    {"file_path": "a.py", "first_character_index": 0, "last_character_index": 100}
                ],
                [
                    {"file_path": "b.py", "first_character_index": 50, "last_character_index": 150}
                ]
            ]
        }
        queries: list[UnansweredQuestion] = [
            UnansweredQuestion(question_id="1", question="q1"),
            UnansweredQuestion(question_id="2", question="q2")
        ]

        res = retriever.retrieve(queries, k=1)

        assert isinstance(res, StudentSearchResults)
        assert res.k == 1
        assert len(res.search_results) == 2

        assert res.search_results[0].question == "q1"
        assert res.search_results[0].question_id == "1"
        assert len(res.search_results[0].retrieved_sources) == 1
        assert res.search_results[0].retrieved_sources[0].file_path == "a.py"

        assert res.search_results[1].question == "q2"
        assert res.search_results[1].question_id == "2"
        assert len(res.search_results[1].retrieved_sources) == 1
        assert res.search_results[1].retrieved_sources[0].file_path == "b.py"

        mock_collection.query.assert_called_once_with(
            query_texts=["q1", "q2"],
            n_results=1
        )

    def test_retrieve_empty_queries_list(
        self, retriever: EmbeddingRetriever, mock_collection: MagicMock
    ) -> None:
        queries: list[UnansweredQuestion] = []

        res = retriever.retrieve(queries, k=3)

        assert len(res.search_results) == 0
        assert res.k == 3
        # Should not call ChromaDB query with empty list
        mock_collection.query.assert_not_called()

    def test_retrieve_k_zero(
        self, retriever: EmbeddingRetriever, mock_collection: MagicMock
    ) -> None:
        res = retriever.retrieve_one("valid query", k=0)
        assert len(res.retrieved_sources) == 0
        mock_collection.query.assert_not_called()

        queries = [UnansweredQuestion(question_id="1", question="valid query")]
        res_multi = retriever.retrieve(queries, k=0)

        # Depending on implementation, it might drop the query from results or return 0 sources
        if len(res_multi.search_results) > 0:
            assert len(res_multi.search_results[0].retrieved_sources) == 0

        mock_collection.query.assert_not_called()

    def test_retrieve_empty_query_string(
        self, retriever: EmbeddingRetriever, mock_collection: MagicMock
    ) -> None:
        res = retriever.retrieve_one("", k=5)
        assert len(res.retrieved_sources) == 0
        mock_collection.query.assert_not_called()

        queries = [UnansweredQuestion(question_id="1", question="")]
        res_multi = retriever.retrieve(queries, k=5)

        if len(res_multi.search_results) > 0:
            assert len(res_multi.search_results[0].retrieved_sources) == 0

        mock_collection.query.assert_not_called()


class TestHybridRetrieverRRF:
    """Tests for the Reciprocal Rank Fusion (RRF) logic in HybridRetriever."""

    @pytest.fixture
    def sample_sources(self) -> list[MinimalSource]:
        """Provide distinct MinimalSource instances for testing."""
        return [
            MinimalSource(
                file_path="a.py", first_character_index=0, last_character_index=100
            ),
            MinimalSource(
                file_path="b.py", first_character_index=0, last_character_index=100
            ),
            MinimalSource(
                file_path="c.py", first_character_index=0, last_character_index=100
            ),
            MinimalSource(
                file_path="d.py", first_character_index=0, last_character_index=100
            ),
        ]

    def test_rrf_empty_inputs(self) -> None:
        """Verify that empty inputs produce an empty list."""
        assert HybridRetriever._rrf([], k=5) == []
        assert HybridRetriever._rrf([[], []], k=5) == []

    def test_rrf_single_retriever_order(
        self, sample_sources: list[MinimalSource]
    ) -> None:
        """Verify that a single retriever list preserves relative ranking."""
        src_a, src_b, src_c = sample_sources[:3]
        results = HybridRetriever._rrf([[src_a, src_b, src_c]], k=3)

        assert results == [src_a, src_b, src_c]

    def test_rrf_overlapping_sources_fused_score(
        self, sample_sources: list[MinimalSource]
    ) -> None:
        """Verify that sources retrieved by multiple systems rank higher."""
        src_a, src_b, src_c = sample_sources[:3]
        # Retriever 1: [src_a, src_b]
        # Retriever 2: [src_b, src_c]
        # src_b appears in both and should accumulate a higher score than src_a
        results = HybridRetriever._rrf([[src_a, src_b], [src_b, src_c]], k=3)

        assert results[0] == src_b
        assert results[1] == src_a
        assert results[2] == src_c

    def test_rrf_k_parameter(
        self, sample_sources: list[MinimalSource]
    ) -> None:
        """Verify that k limits the output size appropriately."""
        src_a, src_b, src_c = sample_sources[:3]
        res_k1 = HybridRetriever._rrf([[src_a, src_b, src_c]], k=1)
        assert len(res_k1) == 1
        assert res_k1 == [src_a]

        res_k0 = HybridRetriever._rrf([[src_a, src_b, src_c]], k=0)
        assert res_k0 == []

        res_k_large = HybridRetriever._rrf([[src_a, src_b]], k=10)
        assert len(res_k_large) == 2
        assert res_k_large == [src_a, src_b]

    def test_rrf_identical_lists(
        self, sample_sources: list[MinimalSource]
    ) -> None:
        """Verify that duplicate lists preserve order and deduplicate sources."""
        src_a, src_b = sample_sources[:2]
        results = HybridRetriever._rrf([[src_a, src_b], [src_a, src_b]], k=5)

        assert results == [src_a, src_b]

    def test_rrf_more_than_k_elements(
        self, sample_sources: list[MinimalSource]
    ) -> None:
        """Verify that when more than k unique elements are fused, only top-k are returned."""
        src_a, src_b, src_c, src_d = sample_sources
        # 4 unique candidates, strictly distinct fused scores:
        # src_b: rank 1 in L1 (1/62) + rank 0 in L2 (1/61) -> highest
        # src_a: rank 0 in L1 (1/61)                       -> second
        # src_d: rank 1 in L2 (1/62)                       -> third
        # src_c: rank 2 in L1 (1/63)                       -> fourth
        results = HybridRetriever._rrf(
            [[src_a, src_b, src_c], [src_b, src_d]], k=2
        )

        assert len(results) == 2
        assert results == [src_b, src_a]
        assert src_d not in results
        assert src_c not in results
