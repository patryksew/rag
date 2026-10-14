"""Unit tests for QueryCache manager."""
from pathlib import Path
import pytest

from src.cache import QueryCache
from src.models.cache import CachedAnswer
from src.models.rag import MinimalSource


class TestQueryCache:
    """Test suite for QueryCache."""

    @pytest.fixture
    def cache_manager(self, tmp_path: Path) -> QueryCache:
        """Create a QueryCache instance backed by a temporary directory."""
        return QueryCache(project_root=tmp_path)

    @pytest.fixture
    def dummy_sources(self) -> list[MinimalSource]:
        """Create sample MinimalSources for testing."""
        return [
            MinimalSource(file_path="a.py", first_character_index=0, last_character_index=100),
            MinimalSource(file_path="b.py", first_character_index=50, last_character_index=150),
            MinimalSource(file_path="c.py", first_character_index=0, last_character_index=200),
        ]

    def test_normalize_query(self) -> None:
        """Verify queries are stripped and lowercased."""
        assert QueryCache.normalize_query("  What is VLLM?  ") == "what is vllm?"
        assert QueryCache.normalize_query("Hello WORLD") == "hello world"

    def test_get_partition_dir(self, cache_manager: QueryCache, tmp_path: Path) -> None:
        """Verify partition directory paths for each method and colon sanitization."""
        bm25_dir = cache_manager.get_partition_dir("lexical", 2000)
        assert bm25_dir == tmp_path / "data" / "processed" / "2000" / "bm25"

        embed_dir = cache_manager.get_partition_dir("embedding", 1000, "test-model")
        assert embed_dir == tmp_path / "data" / "processed" / "1000" / "embedding" / "test-model"

        # Verify colon is sanitized to underscore for cross-platform compatibility
        embed_sanitized = cache_manager.get_partition_dir(
            "embedding", 1000, "all-minilm:l6-v2"
        )
        expected_path = (
            tmp_path / "data" / "processed" / "1000" / "embedding" / "all-minilm_l6-v2"
        )
        assert embed_sanitized == expected_path

        hybrid_dir = cache_manager.get_partition_dir("hybrid", 500, "test-model")
        assert hybrid_dir == tmp_path / "data" / "processed" / "500" / "hybrid" / "test-model"

    def test_search_cache_miss_initially(self, cache_manager: QueryCache) -> None:
        """Verify search cache returns None on empty cache."""
        result = cache_manager.get_search("unknown query", k=2, method="lexical",
                                          max_chunk_size=2000)
        assert result is None

    def test_search_cache_hit_and_prefix_slicing(
        self, cache_manager: QueryCache, dummy_sources: list[MinimalSource]
    ) -> None:
        """Verify search caching and dynamic k prefix slicing."""
        cache_manager.set_search("What is vLLM?", dummy_sources, method="lexical",
                                 max_chunk_size=2000)

        # Hit when requested k <= len(cached)
        res_k2 = cache_manager.get_search("what is vllm?", k=2, method="lexical",
                                          max_chunk_size=2000)
        assert res_k2 is not None
        assert len(res_k2) == 2
        assert res_k2[0].file_path == "a.py"
        assert res_k2[1].file_path == "b.py"

        # Hit when requested k == len(cached)
        res_k3 = cache_manager.get_search("what is vllm?", k=3, method="lexical",
                                          max_chunk_size=2000)
        assert res_k3 is not None
        assert len(res_k3) == 3

        # Miss when requested k > len(cached)
        res_k5 = cache_manager.get_search("what is vllm?", k=5, method="lexical",
                                          max_chunk_size=2000)
        assert res_k5 is None

    def test_search_cache_persistence_and_reload(
        self, cache_manager: QueryCache, dummy_sources: list[MinimalSource], tmp_path: Path
    ) -> None:
        """Verify search cache persists to disk and reloads into a new QueryCache instance."""
        cache_manager.set_search("persistent query", dummy_sources, method="lexical",
                                 max_chunk_size=2000, persist=True)

        partition_dir = cache_manager.get_partition_dir("lexical", 2000)
        assert (partition_dir / "search_cache.json").exists()

        # Create a fresh QueryCache instance reading the same directory
        new_cache = QueryCache(project_root=tmp_path)
        hit = new_cache.get_search("persistent query", k=2, method="lexical", max_chunk_size=2000)
        assert hit is not None
        assert len(hit) == 2

    def test_answer_cache_hit_and_miss(
        self, cache_manager: QueryCache, dummy_sources: list[MinimalSource]
    ) -> None:
        """Verify answer cache lookup with query and context sources."""
        entry = CachedAnswer(
            k=5,
            answer="vLLM is a high-throughput LLM serving engine.",
            retrieved_sources=dummy_sources[:2],
        )

        cache_manager.set_answer("What is vLLM?", entry)

        # Hit with exact query and matching sources
        hit_sources = cache_manager.get_answer("what is vllm?", sources=dummy_sources[:2])
        assert hit_sources is not None
        assert hit_sources.answer == entry.answer
        assert len(hit_sources.retrieved_sources) == 2

        # Miss with different sources
        miss_diff_sources = cache_manager.get_answer("what is vllm?", sources=dummy_sources[1:])
        assert miss_diff_sources is None

        # Miss when no sources provided
        assert cache_manager.get_answer("what is vllm?") is None

        # Miss with unknown query
        assert cache_manager.get_answer("unknown query", sources=dummy_sources[:2]) is None

    def test_clear_cache(
        self, cache_manager: QueryCache, dummy_sources: list[MinimalSource]
    ) -> None:
        """Verify clear_cache removes both memory and disk caches."""
        cache_manager.set_search("q1", dummy_sources, method="lexical", max_chunk_size=2000)
        cache_manager.set_answer(
            "q1",
            CachedAnswer(k=3, answer="ans", retrieved_sources=[]),
        )

        p_dir = cache_manager.get_partition_dir("lexical", 2000)
        assert (p_dir / "search_cache.json").exists()
        assert cache_manager.answer_cache_path.exists()

        cache_manager.clear_cache("lexical", 2000)

        assert not (p_dir / "search_cache.json").exists()
        assert not cache_manager.answer_cache_path.exists()
        assert cache_manager.get_search("q1", k=1, method="lexical", max_chunk_size=2000) is None
        assert cache_manager.get_answer("q1", sources=[]) is None
