from pathlib import Path

from src.answer_generator import AnswerGenerator
from src.indexers.base import Indexer
from src.retriever import Retriever


class RAGCore:
    """
    Core RAG System that orchestrates Indexing, Retrievers, and the Answer Generator.
    This acts as a shared backbone for both the CLI and API.
    """

    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parent.parent
        self._retrievers: dict[tuple[str, int], Retriever] = {}
        self._failed_methods: set[str] = set()
        self._answer_generator: AnswerGenerator | None = None

    def get_retriever(self, method: str, max_chunk_size: int) -> Retriever:
        """
        Lazily initialize and return a retriever for the specified method and chunk size.
        """
        if method in self._failed_methods:
            raise ImportError(f"Dependencies for method '{method}' are missing.")

        key = (method, max_chunk_size)

        if key not in self._retrievers:
            raw_path = self.project_root / "data" / "raw"
            indexer: Indexer

            try:
                if method == "lexical":
                    from src.indexers.bm25 import BM25Indexer
                    indexer = BM25Indexer(max_chunk_size)
                elif method == "embedding":
                    from src.indexers.embedding import EmbeddingIndexer
                    indexer = EmbeddingIndexer(max_chunk_size)
                else:
                    raise ValueError(f"Unknown method: {method}")
            except ImportError as e:
                self._failed_methods.add(method)
                raise e

            self._retrievers[key] = indexer.process_files_from_directory(raw_path)

        return self._retrievers[key]

    def preload_retriever(self, method: str, max_chunk_size: int) -> None:
        """
        Force preload a retriever (useful for API initialization).
        """
        try:
            self.get_retriever(method, max_chunk_size)
        except ImportError:
            print(f"{method.capitalize()} dependencies not found."
                  f" '{method}' method will be unavailable.")

    @property
    def answer_generator(self) -> AnswerGenerator:
        """
        Lazily initialize and return the Answer Generator.
        """
        if self._answer_generator is None:
            self._answer_generator = AnswerGenerator()

        # noinspection bad-return
        return self._answer_generator
