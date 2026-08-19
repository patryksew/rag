import uuid
from abc import ABC, abstractmethod

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
