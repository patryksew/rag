"""Unit tests for RAGCore facade and its caching integration."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core import RAGCore
from src.models import (
    CachedAnswer,
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults,
    UnansweredQuestion,
)


class TestRAGCore:
    """Test suite for RAGCore facade."""

    @pytest.fixture
    def core(self, tmp_path: Path) -> RAGCore:
        """Create a RAGCore instance rooted in tmp_path."""
        core = RAGCore(project_root=tmp_path)
        (tmp_path / "data" / "raw").mkdir(parents=True, exist_ok=True)
        return core

    def test_project_root_configuration(self, tmp_path: Path) -> None:
        """Verify project_root propagates to query_cache and responds to setter."""
        custom_root = tmp_path / "custom"
        core = RAGCore(project_root=custom_root)
        assert core.project_root == custom_root
        assert core._query_cache.project_root == custom_root

        new_root = tmp_path / "another"
        core.project_root = new_root
        assert core.project_root == new_root
        assert core._query_cache.project_root == new_root

    @pytest.fixture
    def dummy_sources(self) -> list[MinimalSource]:
        """Create sample MinimalSources for testing."""
        return [
            MinimalSource(
                file_path="foo.py",
                first_character_index=0,
                last_character_index=50,
            ),
            MinimalSource(
                file_path="bar.py",
                first_character_index=10,
                last_character_index=80,
            ),
        ]

    def test_get_indexer_memoization(self, core: RAGCore) -> None:
        """Verify _get_indexer memoizes indexers for the same (method, chunk_size)."""
        mock_indexer = MagicMock()
        with patch("src.indexers.bm25.BM25Indexer", return_value=mock_indexer) as mock_cls:
            idx1 = core._get_indexer("lexical", 2000)
            idx2 = core._get_indexer("lexical", 2000)

            assert idx1 is idx2
            mock_cls.assert_called_once_with(2000)

    def test_get_indexer_invalid_method(self, core: RAGCore) -> None:
        """Verify _get_indexer raises ValueError for unsupported methods."""
        with pytest.raises(ValueError, match="Unknown method: unknown"):
            core._get_indexer("unknown", 2000)

    def test_get_indexer_failed_method_rejection(self, core: RAGCore) -> None:
        """Verify _get_indexer immediately raises RuntimeError if method previously failed."""
        core._failed_methods["embedding"] = "Connection refused"
        with pytest.raises(RuntimeError, match="currently unavailable"):
            core._get_indexer("embedding", 2000)

    def test_search_cache_hit_bypasses_retriever(
        self, core: RAGCore, dummy_sources: list[MinimalSource]
    ) -> None:
        """Verify search returns cached results without loading the retriever."""
        core._query_cache.set_search("test query", dummy_sources, "lexical", 2000)

        mock_indexer = MagicMock()
        mock_indexer.has_data_changes.return_value = False
        core._indexers[("lexical", 2000)] = mock_indexer

        with patch.object(core, "_get_retriever") as mock_get_retriever:
            result = core.search("test query", "lexical", k=2, max_chunk_size=2000)

            mock_get_retriever.assert_not_called()
            assert len(result.retrieved_sources) == 2
            assert result.retrieved_sources[0].file_path == "foo.py"

    def test_search_cache_miss_loads_retriever_and_populates_cache(
        self, core: RAGCore, dummy_sources: list[MinimalSource]
    ) -> None:
        """Verify search loads retriever on cache miss and caches results."""
        mock_indexer = MagicMock()
        mock_indexer.has_data_changes.return_value = False
        core._indexers[("lexical", 2000)] = mock_indexer

        mock_retriever = MagicMock()
        mock_retriever.retrieve_one.return_value = MinimalSearchResults(
            question_id="q1",
            question="miss query",
            retrieved_sources=dummy_sources,
        )

        with patch.object(core, "_get_retriever", return_value=mock_retriever):
            result = core.search("miss query", "lexical", k=1, max_chunk_size=2000)

            mock_retriever.retrieve_one.assert_called_once_with("miss query", k=10)
            assert len(result.retrieved_sources) == 1
            assert result.retrieved_sources[0].file_path == "foo.py"

            # Should now be cached
            cached = core._query_cache.get_search("miss query", 1, "lexical", 2000)
            assert cached is not None
            assert len(cached) == 1

    def test_validate_cache_state_purges_on_changes(
        self, core: RAGCore, dummy_sources: list[MinimalSource]
    ) -> None:
        """Verify changes in data directory purge cache and retriever."""
        core._query_cache.set_search("stale query", dummy_sources, "lexical", 2000)
        core._retrievers[("lexical", 2000)] = MagicMock()

        mock_indexer = MagicMock()
        mock_indexer.has_data_changes.return_value = True
        core._indexers[("lexical", 2000)] = mock_indexer

        with patch.object(core, "_get_retriever") as mock_get_retriever:
            core._validate_cache_state("lexical", 2000)

            # Query cache should have been purged
            assert core._query_cache.get_search("stale query", 1, "lexical", 2000) is None
            # Retriever should have been reloaded
            mock_get_retriever.assert_called_once_with("lexical", 2000)

    def test_validate_cache_state_hybrid_purges_lexical_and_embedding(
        self, core: RAGCore, dummy_sources: list[MinimalSource]
    ) -> None:
        """Verify hybrid staleness clears underlying lexical and embedding caches."""
        core._query_cache.set_search("lex q", dummy_sources, "lexical", 2000)
        core._query_cache.set_search("emb q", dummy_sources, "embedding", 2000)
        core._query_cache.set_search("hyb q", dummy_sources, "hybrid", 2000)

        mock_indexer = MagicMock()
        mock_indexer.has_data_changes.return_value = True
        core._indexers[("hybrid", 2000)] = mock_indexer

        with patch.object(core, "_get_retriever"):
            core._validate_cache_state("hybrid", 2000)

            assert core._query_cache.get_search("hyb q", 1, "hybrid", 2000) is None
            assert core._query_cache.get_search("lex q", 1, "lexical", 2000) is None
            assert core._query_cache.get_search("emb q", 1, "embedding", 2000) is None

    def test_answer_cache_hit_bypasses_llm(
        self, core: RAGCore, dummy_sources: list[MinimalSource]
    ) -> None:
        """Verify answer returns cached answer without running LLM inference."""
        core._query_cache.set_search("cached q", dummy_sources, "lexical", 2000)
        core._query_cache.set_answer(
            "cached q",
            CachedAnswer(
                k=2,
                answer="Cached answer string",
                retrieved_sources=dummy_sources,
            ),
        )

        mock_indexer = MagicMock()
        mock_indexer.has_data_changes.return_value = False
        core._indexers[("lexical", 2000)] = mock_indexer

        mock_gen = MagicMock()
        core._answer_gen = mock_gen

        with patch.object(core, "_get_retriever") as mock_get_retriever:
            ans: MinimalAnswer = core.answer("cached q", "lexical", k=2, max_chunk_size=2000)

            mock_get_retriever.assert_not_called()
            mock_gen.answer_one.assert_not_called()
            assert ans.answer == "Cached answer string"
            assert len(ans.retrieved_sources) == 2

    def test_search_dataset_batch_caching(
        self, core: RAGCore, dummy_sources: list[MinimalSource]
    ) -> None:
        """Verify search_dataset mixes cached hits and batch-retrieves only misses."""
        core._query_cache.set_search("q1", dummy_sources, "lexical", 2000)

        mock_indexer = MagicMock()
        mock_indexer.has_data_changes.return_value = False
        core._indexers[("lexical", 2000)] = mock_indexer

        q2_sources = [
            MinimalSource(
                file_path="q2.py",
                first_character_index=0,
                last_character_index=10,
            )
        ]
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = MagicMock(
            search_results=[
                MinimalSearchResults(
                    question_id="id-2",
                    question="q2",
                    retrieved_sources=q2_sources,
                )
            ]
        )

        with patch.object(core, "_get_retriever", return_value=mock_retriever):
            questions = [
                UnansweredQuestion(question_id="id-1", question="q1"),
                UnansweredQuestion(question_id="id-2", question="q2"),
            ]
            student_results = core.search_dataset(questions, "lexical", k=1, max_chunk_size=2000)

            assert len(student_results.search_results) == 2
            # q1 from cache
            assert student_results.search_results[0].question == "q1"
            assert student_results.search_results[0].retrieved_sources[0].file_path == "foo.py"
            # q2 from retriever
            assert student_results.search_results[1].question == "q2"
            assert student_results.search_results[1].retrieved_sources[0].file_path == "q2.py"

            # mock_retriever should only have been called with q2 (the miss)
            mock_retriever.retrieve.assert_called_once()
            called_queries = mock_retriever.retrieve.call_args[0][0]
            assert len(called_queries) == 1
            assert called_queries[0].question == "q2"

    def test_answer_dataset_batch_caching(
        self, core: RAGCore, dummy_sources: list[MinimalSource]
    ) -> None:
        """Verify answer_dataset reuses cached answers and generates only misses."""
        # Cache answer for q1
        core._query_cache.set_answer(
            "q1",
            CachedAnswer(
                k=2,
                answer="Cached answer for q1",
                retrieved_sources=dummy_sources,
            ),
            persist=True,
        )

        mock_gen = MagicMock()
        mock_gen.answer_one.return_value = "Generated answer for q2"
        core._answer_gen = mock_gen

        student_results = StudentSearchResults(
            k=2,
            search_results=[
                MinimalSearchResults(
                    question_id="id-1",
                    question="q1",
                    retrieved_sources=dummy_sources,
                ),
                MinimalSearchResults(
                    question_id="id-2",
                    question="q2",
                    retrieved_sources=dummy_sources,
                ),
            ],
        )

        result = core.answer_dataset(student_results)

        assert len(result.search_results) == 2
        # q1 hit from cache
        assert result.search_results[0].answer == "Cached answer for q1"
        # q2 generated
        assert result.search_results[1].answer == "Generated answer for q2"

        # answer_one called ONLY once (for q2 miss)
        mock_gen.answer_one.assert_called_once_with("q2", dummy_sources)

        # q2 should now be in the cache
        cached_q2 = core._query_cache.get_answer("q2", sources=dummy_sources)
        assert cached_q2 is not None
        assert cached_q2.answer == "Generated answer for q2"

        # Verify cache was written only to tmp_path and never to real repo
        assert (core.project_root / "data" / "processed" / "answer_cache.json").exists()
        real_root = Path(__file__).resolve().parent.parent
        assert not (real_root / "data" / "processed" / "answer_cache.json").exists()

    def test_answer_dataset_source_hash_validation_and_isolation(
        self, core: RAGCore, dummy_sources: list[MinimalSource]
    ) -> None:
        """Verify answer_dataset generates on source mismatch and preserves across datasets."""
        # Pre-cache answer for q1 with dummy_sources
        core._query_cache.set_answer(
            "q1",
            CachedAnswer(
                k=2,
                answer="Old answer for dummy sources",
                retrieved_sources=dummy_sources,
            ),
            persist=True,
        )

        different_sources = [
            MinimalSource(file_path="diff.py", first_character_index=0, last_character_index=20)
        ]

        mock_gen = MagicMock()
        mock_gen.answer_one.return_value = "Fresh answer for different sources"
        core._answer_gen = mock_gen

        # Dataset 1: same question, but DIFFERENT sources -> should generate fresh answer
        student_results_diff = StudentSearchResults(
            k=1,
            search_results=[
                MinimalSearchResults(
                    question_id="id-1",
                    question="q1",
                    retrieved_sources=different_sources,
                ),
            ],
        )

        res_diff = core.answer_dataset(student_results_diff)
        assert res_diff.search_results[0].answer == "Fresh answer for different sources"
        mock_gen.answer_one.assert_called_once_with("q1", different_sources)

        # Dataset 2: same question with ORIGINAL dummy_sources -> should HIT cache and not call LLM!
        mock_gen.reset_mock()
        student_results_orig = StudentSearchResults(
            k=2,
            search_results=[
                MinimalSearchResults(
                    question_id="id-1",
                    question="q1",
                    retrieved_sources=dummy_sources,
                ),
            ],
        )

        res_orig = core.answer_dataset(student_results_orig)
        assert res_orig.search_results[0].answer == "Old answer for dummy sources"
        mock_gen.answer_one.assert_not_called()

        real_root = Path(__file__).resolve().parent.parent
        assert not (real_root / "data" / "processed" / "answer_cache.json").exists()

    def test_answer_single_query_source_grounding(
        self, core: RAGCore, dummy_sources: list[MinimalSource]
    ) -> None:
        """Verify core.answer regenerates when different methods return different sources."""
        embedding_sources = [
            MinimalSource(file_path="emb.py", first_character_index=0, last_character_index=30)
        ]

        mock_lex_retriever = MagicMock()
        mock_lex_retriever.retrieve_one.return_value = MinimalSearchResults(
            question_id="q1",
            question="query x",
            retrieved_sources=dummy_sources,
        )

        mock_emb_retriever = MagicMock()
        mock_emb_retriever.retrieve_one.return_value = MinimalSearchResults(
            question_id="q1",
            question="query x",
            retrieved_sources=embedding_sources,
        )

        mock_indexer = MagicMock()
        mock_indexer.has_data_changes.return_value = False
        core._indexers[("lexical", 2000)] = mock_indexer
        core._indexers[("embedding", 2000)] = mock_indexer

        mock_gen = MagicMock()
        mock_gen.answer_one.side_effect = [
            "Lexical answer",
            "Embedding answer",
        ]
        core._answer_gen = mock_gen

        def fake_get_retriever(method: str, max_chunk_size: int) -> MagicMock:
            return mock_lex_retriever if method == "lexical" else mock_emb_retriever

        with patch.object(core, "_get_retriever", side_effect=fake_get_retriever):
            # 1. Lexical answer
            ans_lex = core.answer("query x", "lexical", k=2, max_chunk_size=2000)
            assert ans_lex.answer == "Lexical answer"
            assert ans_lex.retrieved_sources == dummy_sources[:2]

            # 2. Embedding answer for same query: must NOT reuse lexical answer
            ans_emb = core.answer("query x", "embedding", k=1, max_chunk_size=2000)
            assert ans_emb.answer == "Embedding answer"
            assert ans_emb.retrieved_sources == embedding_sources[:1]

            # 3. Repeat lexical: should HIT cache without calling LLM
            ans_lex_cached = core.answer("query x", "lexical", k=2, max_chunk_size=2000)
            assert ans_lex_cached.answer == "Lexical answer"

            # LLM was only called twice (once for lexical, once for embedding)
            assert mock_gen.answer_one.call_count == 2
