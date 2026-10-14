import uuid
from pathlib import Path
from typing import Sequence

from tqdm import tqdm

from src.answer_generator import AnswerGenerator
from src.cache import QueryCache
from src.indexers.base import Indexer
from src.models import (
    CachedAnswer,
    MinimalAnswer,
    MinimalSearchResults,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    UnansweredQuestion,
)
from src.retriever import Retriever


class RAGCore:
    """
    Core RAG System facade that orchestrates indexing, retrieval, and answering.
    Serves as the unified business logic layer for both the CLI and API.
    """

    def __init__(self, project_root: Path | None = None) -> None:
        """
        Initialize the RAGCore facade.

        Args:
            project_root: Root directory of the project. If None, derived from file location.
        """
        self._project_root = (
            project_root if project_root is not None
            else Path(__file__).resolve().parent.parent
        )
        self._indexers: dict[tuple[str, int], Indexer] = {}
        self._retrievers: dict[tuple[str, int], Retriever] = {}
        self._failed_methods: dict[str, str] = {}
        self._answer_gen: AnswerGenerator | None = None
        self._query_cache = QueryCache(project_root=self._project_root)

    @property
    def project_root(self) -> Path:
        """
        Return the project root directory.

        Returns:
            Path: Root directory of the project.
        """
        return self._project_root

    @project_root.setter
    def project_root(self, value: Path) -> None:
        """
        Set the project root directory and update the query cache manager.

        Args:
            value: New project root directory.
        """
        self._project_root = value
        self._query_cache = QueryCache(project_root=value)

    @property
    def failed_methods(self) -> dict[str, str]:
        """
        Return a copy of the failed methods mapping.

        Returns:
            dict[str, str]: Mapping of failed method names to their failure reasons.
        """
        return self._failed_methods.copy()

    def _get_indexer(self, method: str, max_chunk_size: int) -> Indexer:
        """
        Lazily initialize and return an indexer for the specified method and chunk size.

        Args:
            method: Retrieval method ('lexical', 'embedding', 'hybrid').
            max_chunk_size: Maximum chunk size in characters.

        Returns:
            Indexer: Initialized indexer instance.

        Raises:
            RuntimeError: If the method is unavailable (e.g. Ollama service is unreachable).
            ValueError: If the method name is unknown.
        """
        if method in self._failed_methods:
            raise RuntimeError(
                f"Method '{method}' is currently unavailable (is Ollama running?): "
                f"{self._failed_methods[method]}"
            )

        key = (method, max_chunk_size)

        if key not in self._indexers:
            indexer: Indexer

            try:
                if method == "lexical":
                    from src.indexers.bm25 import BM25Indexer

                    indexer = BM25Indexer(max_chunk_size)
                elif method == "embedding":
                    from src.indexers.embedding import EmbeddingIndexer

                    indexer = EmbeddingIndexer(max_chunk_size)
                elif method == "hybrid":
                    from src.indexers.bm25 import BM25Indexer
                    from src.indexers.embedding import EmbeddingIndexer
                    from src.indexers.hybrid import HybridIndexer

                    bm25 = self._get_indexer("lexical", max_chunk_size)
                    embedding = self._get_indexer("embedding", max_chunk_size)
                    if not isinstance(bm25, BM25Indexer) or not isinstance(
                        embedding, EmbeddingIndexer
                    ):
                        raise RuntimeError(
                            "Unexpected indexer types for hybrid retrieval"
                        )

                    indexer = HybridIndexer(
                        bm25_indexer=bm25,
                        embedding_indexer=embedding,
                        max_chunk_size=max_chunk_size,
                        chunker=bm25._chunker,
                    )
                else:
                    raise ValueError(f"Unknown method: {method}")

                self._indexers[key] = indexer
            except ValueError:
                raise
            except Exception as e:
                self._failed_methods[method] = str(e)
                raise RuntimeError(
                    f"Failed to initialize method '{method}' (is Ollama running?): {e}"
                ) from e

        return self._indexers[key]

    def _get_retriever(self, method: str, max_chunk_size: int) -> Retriever:
        """
        Lazily initialize and return a retriever for the specified method and chunk size.

        Args:
            method: Retrieval method ('lexical', 'embedding', 'hybrid').
            max_chunk_size: Maximum chunk size in characters.

        Returns:
            Retriever: Initialized retriever instance.

        Raises:
            RuntimeError: If the method is unavailable (e.g. Ollama service is unreachable).
            ValueError: If the method name is unknown.
        """
        key = (method, max_chunk_size)

        if key not in self._retrievers:
            indexer = self._get_indexer(method, max_chunk_size)
            raw_path = self.project_root / "data" / "raw"
            try:
                self._retrievers[key] = indexer.process_files_from_directory(raw_path)
            except Exception as e:
                self._failed_methods[method] = str(e)
                raise RuntimeError(
                    f"Failed to initialize retriever for method '{method}': {e}"
                ) from e

        return self._retrievers[key]

    @property
    def _answer_generator(self) -> AnswerGenerator:
        """
        Lazily initialize and return the Answer Generator.

        Returns:
            AnswerGenerator: Initialized AnswerGenerator instance.

        Raises:
            RuntimeError: If connecting to Ollama fails.
        """
        if self._answer_gen is None:
            try:
                self._answer_gen = AnswerGenerator()
            except Exception as e:
                raise RuntimeError(
                    f"Answer generator unavailable (is Ollama running?): {e}"
                ) from e

        # noinspection bad-return
        return self._answer_gen

    def _validate_cache_state(self, method: str, max_chunk_size: int) -> None:
        """
        Validate cache state against the raw data directory.

        If the raw data has changed, purges stale query caches and invalidates
        the memoized retriever so the index is rebuilt/reloaded.

        Args:
            method: Retrieval method ('lexical', 'embedding', 'hybrid').
            max_chunk_size: Maximum chunk size in characters.
        """
        indexer = self._get_indexer(method, max_chunk_size)
        raw_path = self.project_root / "data" / "raw"
        if indexer.has_data_changes(raw_path):
            self._query_cache.clear_cache(method, max_chunk_size)
            self._retrievers.pop((method, max_chunk_size), None)
            if method == "hybrid":
                self._query_cache.clear_cache("lexical", max_chunk_size)
                self._query_cache.clear_cache("embedding", max_chunk_size)
                self._retrievers.pop(("lexical", max_chunk_size), None)
                self._retrievers.pop(("embedding", max_chunk_size), None)
            else:
                self._query_cache.clear_cache("hybrid", max_chunk_size)
                self._retrievers.pop(("hybrid", max_chunk_size), None)
            self._get_retriever(method, max_chunk_size)

    def index(self, method: str, max_chunk_size: int = 2000) -> None:
        """
        Ingest data/raw/ and build or load the index for the specified method.

        Args:
            method: Retrieval method ('lexical', 'embedding', 'hybrid').
            max_chunk_size: Maximum chunk size in characters.
        """
        self._retrievers.pop((method, max_chunk_size), None)
        if method == "hybrid":
            self._retrievers.pop(("lexical", max_chunk_size), None)
            self._retrievers.pop(("embedding", max_chunk_size), None)
            self._query_cache.clear_cache("lexical", max_chunk_size)
            self._query_cache.clear_cache("embedding", max_chunk_size)
        else:
            self._retrievers.pop(("hybrid", max_chunk_size), None)
            self._query_cache.clear_cache("hybrid", max_chunk_size)

        self._get_retriever(method, max_chunk_size)
        self._query_cache.clear_cache(method, max_chunk_size)

    def search(
        self,
        query: str,
        method: str,
        k: int = 5,
        max_chunk_size: int = 2000,
    ) -> MinimalSearchResults:
        """
        Return the top-k sources for a single query.

        Args:
            query: The question to search for.
            method: Retrieval method ('lexical', 'embedding', 'hybrid').
            k: Number of top sources to return.
            max_chunk_size: Maximum chunk size used during indexing.

        Returns:
            MinimalSearchResults: The top-k retrieved sources.
        """
        self._validate_cache_state(method, max_chunk_size)

        cached_sources = self._query_cache.get_search(
            query, k, method, max_chunk_size
        )
        if cached_sources is not None:
            return MinimalSearchResults(
                question_id=str(uuid.uuid4()),
                question=query,
                retrieved_sources=cached_sources,
            )

        retriever = self._get_retriever(method, max_chunk_size)
        fetch_k = max(k, 10)
        fetch_result = retriever.retrieve_one(query, k=fetch_k)
        self._query_cache.set_search(
            query, fetch_result.retrieved_sources, method, max_chunk_size
        )

        return MinimalSearchResults(
            question_id=fetch_result.question_id,
            question=query,
            retrieved_sources=fetch_result.retrieved_sources[:k],
        )

    def search_dataset(
        self,
        dataset: RagDataset | Sequence[UnansweredQuestion],
        method: str,
        k: int = 5,
        max_chunk_size: int = 2000,
    ) -> StudentSearchResults:
        """
        Run search over a whole dataset and return StudentSearchResults.

        Args:
            dataset: RagDataset or sequence of UnansweredQuestions.
            method: Retrieval method ('lexical', 'embedding', 'hybrid').
            k: Top-k sources per question.
            max_chunk_size: Maximum chunk size used during indexing.

        Returns:
            StudentSearchResults: The retrieval results for all questions.
        """
        self._validate_cache_state(method, max_chunk_size)

        questions = (
            list(dataset.rag_questions)
            if isinstance(dataset, RagDataset)
            else list(dataset)
        )

        results: list[MinimalSearchResults | None] = [None] * len(questions)
        misses_indices: list[int] = []
        misses_questions: list[UnansweredQuestion] = []

        for idx, q in enumerate(questions):
            cached = self._query_cache.get_search(
                q.question, k, method, max_chunk_size
            )
            if cached is not None:
                results[idx] = MinimalSearchResults(
                    question_id=q.question_id,
                    question=q.question,
                    retrieved_sources=cached,
                )
            else:
                misses_indices.append(idx)
                misses_questions.append(
                    UnansweredQuestion(
                        question_id=q.question_id,
                        question=q.question,
                    )
                )

        if misses_questions:
            retriever = self._get_retriever(method, max_chunk_size)
            fetch_k = max(k, 10)
            miss_results = retriever.retrieve(misses_questions, k=fetch_k)

            for original_idx, miss_res in zip(
                misses_indices, miss_results.search_results
            ):
                self._query_cache.set_search(
                    miss_res.question,
                    miss_res.retrieved_sources,
                    method,
                    max_chunk_size,
                    persist=False,
                )
                results[original_idx] = MinimalSearchResults(
                    question_id=miss_res.question_id,
                    question=miss_res.question,
                    retrieved_sources=miss_res.retrieved_sources[:k],
                )

            self._query_cache.save_search_cache(method, max_chunk_size)

        final_results = [r for r in results if r is not None]
        return StudentSearchResults(search_results=final_results, k=k)

    def answer(
        self,
        query: str,
        method: str,
        k: int = 5,
        max_chunk_size: int = 2000,
    ) -> MinimalAnswer:
        """
        Answer a single query using retrieved context.

        Args:
            query: The question to answer.
            method: Retrieval method ('lexical', 'embedding', 'hybrid').
            k: The number of retrieved context chunks to use.
            max_chunk_size: Maximum chunk size used during indexing.

        Returns:
            MinimalAnswer: The generated answer combined with search results.
        """
        self._validate_cache_state(method, max_chunk_size)

        # 1. Fast path: check if search is already cached
        cached_sources = self._query_cache.get_search(
            query, k, method, max_chunk_size
        )
        if cached_sources is not None:
            cached_answer = self._query_cache.get_answer(
                query, sources=cached_sources
            )
            if cached_answer is not None:
                return MinimalAnswer(
                    question_id=str(uuid.uuid4()),
                    question=query,
                    retrieved_sources=cached_sources,
                    answer=cached_answer.answer,
                )

        # 2. Search miss or answer miss for cached sources: execute search
        search_result = self.search(
            query, method, k=k, max_chunk_size=max_chunk_size
        )

        # 3. Check if an answer already exists for these specific retrieved sources
        cached_ans_for_sources = self._query_cache.get_answer(
            query, sources=search_result.retrieved_sources
        )
        if cached_ans_for_sources is not None:
            return MinimalAnswer(
                question_id=search_result.question_id,
                question=search_result.question,
                retrieved_sources=search_result.retrieved_sources,
                answer=cached_ans_for_sources.answer,
            )

        # 4. Generate fresh answer
        answer_text = self._answer_generator.answer_one(
            query, search_result.retrieved_sources
        )
        self._query_cache.set_answer(
            query,
            CachedAnswer(
                k=k,
                answer=answer_text,
                retrieved_sources=search_result.retrieved_sources,
            ),
            persist=True,
        )
        return MinimalAnswer(
            question_id=search_result.question_id,
            question=search_result.question,
            retrieved_sources=search_result.retrieved_sources,
            answer=answer_text,
        )

    def answer_dataset(
        self,
        student_search_results: StudentSearchResults,
    ) -> StudentSearchResultsAndAnswer:
        """
        Generate answers for a dataset of search results with answer caching.

        Args:
            student_search_results: The search results to answer.

        Returns:
            StudentSearchResultsAndAnswer: Dataset of answers and search results.
        """
        k = student_search_results.k
        answers: list[MinimalAnswer] = []
        has_new_answers = False

        for search_result in tqdm(
            student_search_results.search_results,
            desc="Answering questions",
            leave=False,
        ):
            cached = self._query_cache.get_answer(
                search_result.question,
                sources=search_result.retrieved_sources,
            )
            if cached is not None:
                answer_text = cached.answer
            else:
                answer_text = self._answer_generator.answer_one(
                    search_result.question, search_result.retrieved_sources
                )
                self._query_cache.set_answer(
                    search_result.question,
                    CachedAnswer(
                        k=k,
                        answer=answer_text,
                        retrieved_sources=search_result.retrieved_sources,
                    ),
                    persist=False,
                )
                has_new_answers = True

            answers.append(
                MinimalAnswer(
                    question_id=search_result.question_id,
                    question=search_result.question,
                    retrieved_sources=search_result.retrieved_sources,
                    answer=answer_text,
                )
            )

        if has_new_answers:
            self._query_cache.save_answer_cache()

        return StudentSearchResultsAndAnswer(search_results=answers, k=k)
