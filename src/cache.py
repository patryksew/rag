"""Query caching manager for partitioned search and answer caches."""
import logging
from pathlib import Path

from src.models.cache import AnswerCache, CachedAnswer, SearchCache
from src.models.rag import MinimalSource

logger = logging.getLogger(__name__)


class QueryCache:
    """
    Manages loading, persisting, and querying partitioned search and answer caches.
    """

    def __init__(self, project_root: Path | None = None) -> None:
        """
        Initialize the QueryCache manager.

        Args:
            project_root: Root directory of the project. If None, derived from file location.
        """
        if project_root is None:
            self.project_root = Path(__file__).resolve().parent.parent
        else:
            self.project_root = project_root

        self._search_caches: dict[Path, SearchCache] = {}
        self._answer_cache: AnswerCache | None = None

    @property
    def answer_cache_path(self) -> Path:
        """
        Return the path to the answer cache file directly in data/processed/.

        Returns:
            Path: Path to data/processed/answer_cache.json.
        """
        return self.project_root / "data" / "processed" / "answer_cache.json"

    @staticmethod
    def normalize_query(query: str) -> str:
        """
        Normalize a query string for cache keys.

        Args:
            query: Raw query string.

        Returns:
            str: Trimmed, lowercased query.
        """
        return query.strip().lower()

    def get_partition_dir(
        self,
        method: str,
        max_chunk_size: int,
        embed_model: str = "all-minilm:l6-v2",
    ) -> Path:
        """
        Resolve the directory path for a specific configuration partition.

        Args:
            method: Retrieval method ('lexical', 'embedding', 'hybrid').
            max_chunk_size: Maximum chunk size in characters.
            embed_model: Embedding model slug.

        Returns:
            Path: Partition directory path.
        """
        model_slug = embed_model.replace(":", "_").replace("/", "_")
        base = self.project_root / "data" / "processed" / str(max_chunk_size)
        if method == "lexical":
            return base / "bm25"
        elif method in ("embedding", "hybrid"):
            return base / method / model_slug
        return base / method

    def _get_search_cache(self, partition_dir: Path) -> SearchCache:
        """
        Retrieve or lazily load the SearchCache for a partition.

        Args:
            partition_dir: Directory of the partition.

        Returns:
            SearchCache: Active search cache instance.
        """
        if partition_dir not in self._search_caches:
            cache_file = partition_dir / "search_cache.json"
            if cache_file.exists():
                # noinspection broad-exception
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        self._search_caches[partition_dir] = (
                            SearchCache.model_validate_json(f.read())
                        )
                except Exception as e:
                    logger.warning(
                        "Failed to load search cache from %s, initializing empty cache: %s",
                        cache_file,
                        e,
                    )
                    self._search_caches[partition_dir] = SearchCache()
            else:
                self._search_caches[partition_dir] = SearchCache()

        return self._search_caches[partition_dir]

    def _get_answer_cache(self) -> AnswerCache:
        """
        Retrieve or lazily load the global AnswerCache directly under data/processed/.

        Returns:
            AnswerCache: Active answer cache instance.
        """
        if self._answer_cache is None:
            cache_file = self.answer_cache_path
            if cache_file.exists():
                # noinspection broad-exception
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        self._answer_cache = AnswerCache.model_validate_json(f.read())
                except Exception as e:
                    logger.warning(
                        "Failed to load answer cache from %s, initializing empty cache: %s",
                        cache_file,
                        e,
                    )
                    self._answer_cache = AnswerCache()
            else:
                self._answer_cache = AnswerCache()

        return self._answer_cache

    def get_search(
        self,
        query: str,
        k: int,
        method: str,
        max_chunk_size: int,
        embed_model: str = "all-minilm:l6-v2",
    ) -> list[MinimalSource] | None:
        """
        Look up cached search sources with dynamic k prefix slicing.

        Args:
            query: Raw query string.
            k: Number of sources requested.
            method: Retrieval method.
            max_chunk_size: Maximum chunk size.
            embed_model: Embedding model slug.

        Returns:
            list[MinimalSource] | None: Sliced sources if hit, None if miss.
        """
        norm_q = self.normalize_query(query)
        p_dir = self.get_partition_dir(method, max_chunk_size, embed_model)
        cache = self._get_search_cache(p_dir)
        return cache.get(norm_q, k)

    def set_search(
        self,
        query: str,
        sources: list[MinimalSource],
        method: str,
        max_chunk_size: int,
        embed_model: str = "all-minilm:l6-v2",
        persist: bool = True,
    ) -> None:
        """
        Store retrieved sources for a query in the partition search cache.

        Args:
            query: Raw query string.
            sources: List of retrieved sources to cache.
            method: Retrieval method.
            max_chunk_size: Maximum chunk size.
            embed_model: Embedding model slug.
            persist: If True, writes to disk immediately.
        """
        norm_q = self.normalize_query(query)
        p_dir = self.get_partition_dir(method, max_chunk_size, embed_model)
        cache = self._get_search_cache(p_dir)
        cache.set(norm_q, sources)
        if persist:
            self.save_search_cache(method, max_chunk_size, embed_model)

    def get_answer(
        self,
        query: str,
        sources: list[MinimalSource] | None = None,
        sources_hash: str | None = None,
    ) -> CachedAnswer | None:
        """
        Look up a cached answer matching query and sources directly in processed/.

        Args:
            query: Raw query string.
            sources: Retrieved sources used as context.
            sources_hash: Precomputed sources hash if available.

        Returns:
            CachedAnswer | None: Cached answer entry if found.
        """
        norm_q = self.normalize_query(query)
        cache = self._get_answer_cache()
        return cache.get(norm_q, sources=sources, sources_hash=sources_hash)

    def set_answer(
        self,
        query: str,
        entry: CachedAnswer,
        persist: bool = True,
    ) -> None:
        """
        Store a generated answer in the global answer cache in data/processed/.

        Args:
            query: Raw query string.
            entry: CachedAnswer to store.
            persist: If True, writes to disk immediately.
        """
        norm_q = self.normalize_query(query)
        cache = self._get_answer_cache()
        cache.set(norm_q, entry)
        if persist:
            self.save_answer_cache()

    @staticmethod
    def _atomic_write_json(path: Path, content: str) -> None:
        """
        Write JSON content atomically using a temporary file and rename.

        Args:
            path: Destination file path.
            content: Serialized JSON content.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(content)
        tmp_path.replace(path)

    def save_search_cache(
        self,
        method: str,
        max_chunk_size: int,
        embed_model: str = "all-minilm:l6-v2",
    ) -> None:
        """
        Serialize and write search cache to disk for a partition atomically.

        Args:
            method: Retrieval method.
            max_chunk_size: Maximum chunk size.
            embed_model: Embedding model slug.
        """
        p_dir = self.get_partition_dir(method, max_chunk_size, embed_model)
        cache = self._get_search_cache(p_dir)
        cache_file = p_dir / "search_cache.json"
        self._atomic_write_json(cache_file, cache.model_dump_json(indent=2))

    def save_answer_cache(self) -> None:
        """Serialize and write global answer cache atomically."""
        cache = self._get_answer_cache()
        self._atomic_write_json(self.answer_cache_path, cache.model_dump_json(indent=2))

    def clear_answer_cache(self) -> None:
        """Purge in-memory and disk global answer cache."""
        self._answer_cache = None
        cache_file = self.answer_cache_path
        if cache_file.exists():
            cache_file.unlink()

    def clear_cache(
        self,
        method: str,
        max_chunk_size: int,
        embed_model: str = "all-minilm:l6-v2",
        clear_answer: bool = True,
    ) -> None:
        """
        Purge in-memory and disk search caches for a specific partition.

        Args:
            method: Retrieval method.
            max_chunk_size: Maximum chunk size.
            embed_model: Embedding model slug.
            clear_answer: Whether to also purge the global answer cache.
        """
        p_dir = self.get_partition_dir(method, max_chunk_size, embed_model)
        self._search_caches.pop(p_dir, None)

        search_file = p_dir / "search_cache.json"
        if search_file.exists():
            search_file.unlink()

        if clear_answer:
            self.clear_answer_cache()
