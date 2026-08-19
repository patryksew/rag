import pytest
from unittest.mock import MagicMock
from src.retriever import EmbeddingRetriever
from src.models import MinimalSearchResults, StudentSearchResults, UnansweredQuestion


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
