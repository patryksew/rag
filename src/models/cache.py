import hashlib
from typing import Any

from pydantic import BaseModel, Field

from src.models.rag import MinimalSource


def compute_sources_hash(sources: list[MinimalSource]) -> str:
    """
    Compute a deterministic hash for a list of MinimalSources.

    Args:
        sources: List of MinimalSource objects.

    Returns:
        str: 16-character hexadecimal hash string.
    """
    raw = ";".join(
        f"{s.file_path}:{s.first_character_index}:{s.last_character_index}"
        for s in sources
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class SearchCache(BaseModel):
    """
    Represents the search query cache for a specific partition.

    Maps a normalized query string to its retrieved sources list.

    Attributes:
        queries (dict[str, list[MinimalSource]]): Mapping from normalized query
            to list of retrieved sources.
    """
    queries: dict[str, list[MinimalSource]] = Field(default_factory=dict)

    def get(self, query: str, k: int) -> list[MinimalSource] | None:
        """
        Return the top-k sources if query is cached and contains at least k sources.

        Args:
            query (str): The normalized query string.
            k (int): Number of sources requested.

        Returns:
            list[MinimalSource] | None: Sliced sources if hit, None otherwise.
        """
        sources = self.queries.get(query)
        if sources is not None and len(sources) >= k:
            return sources[:k]
        return None

    def set(self, query: str, sources: list[MinimalSource]) -> None:
        """
        Store or update sources for query, preserving longer or equal lists.

        Args:
            query (str): The normalized query string.
            sources (list[MinimalSource]): Retrieved sources to cache.
        """
        existing = self.queries.get(query)
        if existing is None or len(sources) >= len(existing):
            self.queries[query] = sources


class CachedAnswer(BaseModel):
    """
    Represents a single cached answer for a query.

    Attributes:
        k (int): Number of context chunks used to generate the answer.
        answer (str): The generated answer text.
        retrieved_sources (list[MinimalSource]): Sources used as context.
        sources_hash (str): Hash of retrieved sources for cache validation.
    """
    k: int
    answer: str
    retrieved_sources: list[MinimalSource] = Field(default_factory=list)
    sources_hash: str = ""

    def model_post_init(self, __context: Any) -> None:
        """Compute sources_hash if not provided."""
        if not self.sources_hash:
            self.sources_hash = compute_sources_hash(self.retrieved_sources)


class AnswerCache(BaseModel):
    """
    Represents the answer query cache.

    Maps compound keys '{query}:{sources_hash}' directly to a CachedAnswer.

    Attributes:
        queries (dict[str, CachedAnswer]): Flat mapping from compound key
            to CachedAnswer object.
    """
    queries: dict[str, CachedAnswer] = Field(default_factory=dict)

    @staticmethod
    def make_key(query: str, sources_hash: str) -> str:
        """
        Construct the compound cache key.

        Args:
            query (str): The normalized query string.
            sources_hash (str): Hash of retrieved sources context.

        Returns:
            str: Formatted compound key '{query}:{sources_hash}'.
        """
        return f"{query}:{sources_hash}"

    def get(
        self,
        query: str,
        sources: list[MinimalSource] | None = None,
        sources_hash: str | None = None,
    ) -> CachedAnswer | None:
        """
        Return cached answer matching query and sources or sources_hash.

        Args:
            query (str): The normalized query string.
            sources (list[MinimalSource] | None): Sources used as context.
            sources_hash (str | None): Precomputed sources hash if available.

        Returns:
            CachedAnswer | None: Cached answer if hit, None otherwise.
        """
        if sources_hash is None and sources is not None:
            sources_hash = compute_sources_hash(sources)

        if sources_hash is not None:
            return self.queries.get(self.make_key(query, sources_hash))

        return None

    def set(self, query: str, entry: CachedAnswer) -> None:
        """
        Store or update a cached answer entry under its query and sources hash key.

        Args:
            query (str): The normalized query string.
            entry (CachedAnswer): The cached answer entry to store.
        """
        if not entry.sources_hash:
            entry.sources_hash = compute_sources_hash(entry.retrieved_sources)

        key = self.make_key(query, entry.sources_hash)
        self.queries[key] = entry
