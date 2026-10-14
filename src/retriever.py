import uuid
from abc import ABC, abstractmethod
from collections import defaultdict

from chromadb import Collection as ChromaCollection
from tqdm import tqdm
import bm25s

from src.models import MinimalSearchResults, MinimalSource, StudentSearchResults, UnansweredQuestion
from src.text_utils import preprocess_text


class Retriever(ABC):
    """
    Abstract base class for information retrieval.
    """

    @abstractmethod
    def retrieve(self, queries: list[UnansweredQuestion], k: int = 5) -> StudentSearchResults:
        """
        Retrieve relevant sources for a list of queries.

        Args:
            queries (list[UnansweredQuestion]): The list of queries.
            k (int): Number of top sources to retrieve per query.

        Returns:
            StudentSearchResults: The retrieval results for the queries.
        """
        ...

    @abstractmethod
    def retrieve_one(self, query: str, k: int = 5) -> MinimalSearchResults:
        """
        Retrieve relevant sources for a single query.

        Args:
            query (str): The query string.
            k (int): Number of top sources to retrieve.

        Returns:
            MinimalSearchResults: The retrieval results for the query.
        """
        ...


class BM25Retriever(Retriever):
    """
    Retriever implementation using the BM25 algorithm.
    """

    def __init__(self, retriever: bm25s.BM25):
        """
        Initialize the BM25Retriever.

        Args:
            retriever (bm25s.BM25): The underlying BM25 retriever instance.
        """
        self._retriever = retriever

    def retrieve(self, queries: list[UnansweredQuestion], k: int = 5) -> StudentSearchResults:
        """
        Retrieve relevant sources for a list of queries using BM25.

        Args:
            queries (list[UnansweredQuestion]): The list of queries.
            k (int): Number of top sources to retrieve per query.

        Returns:
            StudentSearchResults: The BM25 retrieval results for the queries.
        """
        if not queries:
            return StudentSearchResults(search_results=[], k=k)

        tokens = bm25s.tokenize([preprocess_text(q.question) for q in queries],
                                show_progress=len(queries) > 1)
        results, scores = self._retriever.retrieve(tokens, k=k)

        mapped_results: list[MinimalSearchResults] = []
        progress_queries = tqdm(queries, desc="Formatting search results",
                                disable=len(queries) <= 1, leave=False)
        for i, query in enumerate(progress_queries):
            sources: list[MinimalSource] = []
            for j in range(results.shape[1]):
                doc = results[i, j]
                sources.append(
                    MinimalSource.model_validate(doc)
                )
            mapped_results.append(
                MinimalSearchResults(
                    question_id=query.question_id,
                    question=query.question,
                    retrieved_sources=sources
                )
            )
        return StudentSearchResults(
            search_results=mapped_results,
            k=k
        )

    def retrieve_one(self, query: str, k: int = 5) -> MinimalSearchResults:
        """
        Retrieve relevant sources for a single query using BM25.

        Args:
            query (str): The query string.
            k (int): Number of top sources to retrieve.

        Returns:
            MinimalSearchResults: The BM25 retrieval results for the query.
        """
        results, scores = self._retriever.retrieve(bm25s.tokenize(preprocess_text(query)), k=k)

        sources: list[MinimalSource] = []
        for i in range(results.shape[1]):
            doc = results[0, i]
            sources.append(
                MinimalSource.model_validate(doc)
            )

        return MinimalSearchResults(
            question_id=str(uuid.uuid4()),
            question=query,
            retrieved_sources=sources
        )


class EmbeddingRetriever(Retriever):
    def __init__(self, collection: ChromaCollection) -> None:
        self._collection = collection

    def retrieve_one(self, query: str, k: int = 5) -> MinimalSearchResults:
        if not query or k <= 0:
            return MinimalSearchResults(
                question_id=str(uuid.uuid4()),
                question=query,
                retrieved_sources=[]
            )

        results = self._collection.query(
            query_texts=[query],
            n_results=k
        )

        metadatas = results["metadatas"]
        assert metadatas is not None
        metadata = metadatas[0]

        sources = [MinimalSource.model_validate(result) for result in metadata]

        return MinimalSearchResults(
            question_id=str(uuid.uuid4()),
            question=query,
            retrieved_sources=sources
        )

    def retrieve(self, queries: list[UnansweredQuestion], k: int = 5) -> StudentSearchResults:
        if not queries or k <= 0:
            return StudentSearchResults(search_results=[], k=k)

        valid_queries = [q for q in queries if q.question]

        if valid_queries:
            results = self._collection.query(
                query_texts=[q.question for q in valid_queries],
                n_results=k
            )
            metadatas = results["metadatas"]
            assert metadatas is not None
        else:
            metadatas = []

        mapped_results: list[MinimalSearchResults] = []
        valid_idx = 0

        for query in queries:
            if query.question:
                metadata = metadatas[valid_idx]
                sources = [MinimalSource.model_validate(result) for result in metadata]
                valid_idx += 1
            else:
                sources = []

            mapped_results.append(
                MinimalSearchResults(
                    question_id=query.question_id,
                    question=query.question,
                    retrieved_sources=sources
                )
            )

        return StudentSearchResults(
            search_results=mapped_results,
            k=k
        )


class HybridRetriever(Retriever):
    def __init__(self, bm25: BM25Retriever, embedding: EmbeddingRetriever):
        self.bm25_retriever = bm25
        self.embedding_retriever = embedding

    @staticmethod
    def _rrf(results: list[list[MinimalSource]], k: int = 5) -> list[MinimalSource]:
        """Reciprocal Rank Fusion"""

        rrf_k = 60

        reranked: dict[MinimalSource, float] = defaultdict(float)

        for result_list in results:
            for rank, result in enumerate(result_list):
                reranked[result] += 1 / (rrf_k + rank + 1)

        sorted_sources = sorted(reranked, key=lambda s: reranked[s], reverse=True)

        return sorted_sources[:k]

    def retrieve_one(self, query: str, k: int = 5) -> MinimalSearchResults:
        bm25_results = self.bm25_retriever.retrieve_one(query, k=k * 2)
        embedding_results = self.embedding_retriever.retrieve_one(query, k=k * 2)
        rrf_results = self._rrf(
            [bm25_results.retrieved_sources, embedding_results.retrieved_sources],
            k=k)

        return MinimalSearchResults(
            question_id=str(uuid.uuid4()),
            question=query,
            retrieved_sources=rrf_results
        )

    def retrieve(self, queries: list[UnansweredQuestion], k: int = 5) -> StudentSearchResults:
        bm25_results = self.bm25_retriever.retrieve(queries, k=k*2).search_results
        embedding_results = self.embedding_retriever.retrieve(queries, k=k*2).search_results

        final_results = StudentSearchResults(search_results=[], k=k)

        for a, b in zip(bm25_results, embedding_results):
            # Should never happen, but just to be sure
            if a.question != b.question:
                raise ValueError("Questions do not match")
            if a.question_id != b.question_id:
                raise ValueError("Question IDs do not match")

            rrf_results = self._rrf([a.retrieved_sources, b.retrieved_sources], k=k)

            final_results.search_results.append(
                MinimalSearchResults(
                    question_id=a.question_id,
                    question=a.question,
                    retrieved_sources=rrf_results
                )
            )

        return final_results
