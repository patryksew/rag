import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from src.models import MinimalAnswer
from src.answer_generator import AnswerGenerator
from src.indexers.bm25 import BM25Indexer
from src.models import MinimalSearchResults
from src.retriever import Retriever


class RAGAPI:
    """RAG API Class that initializes resources once and routes requests."""

    def __init__(self, max_chunk_size: int = 2000):
        print("Initializing RAG API (Loading index and models)...")
        self.project_root = Path(__file__).resolve().parent.parent
        self.max_chunk_size = max_chunk_size

        self.retrievers: dict[str, Retriever] = {}

        # Load lexical retriever
        lexical_indexer = BM25Indexer(self.max_chunk_size)
        raw_path = self.project_root / "data" / "raw"
        self.retrievers["lexical"] = lexical_indexer.process_files_from_directory(raw_path)

        # Attempt to load embedding retriever
        try:
            from src.indexers.embedding import EmbeddingIndexer
            embedding_indexer = EmbeddingIndexer(self.max_chunk_size)
            self.retrievers["embedding"] = embedding_indexer.process_files_from_directory(raw_path)
        except ImportError:
            print("Embedding dependencies not found. 'embedding' method will be unavailable.")

        self.answer_generator = AnswerGenerator()

        self.app = FastAPI()
        self._setup_routes()

    def _setup_routes(self) -> None:
        self.app.add_api_route("/search", self.search, methods=["GET"])
        self.app.add_api_route("/answer", self.answer, methods=["GET"])

    def _get_retriever(self, method: str) -> Retriever:
        if method not in ["lexical", "embedding"]:
            raise HTTPException(status_code=400, detail=f"Invalid method: {method}")
        if method not in self.retrievers:
            raise HTTPException(
                status_code=501,
                detail=f"Method '{method}' is not available. Missing dependencies?"
            )
        return self.retrievers[method]

    def search(self, query: str, k: int = 5,
               method: Literal["lexical", "embedding"] = "lexical") -> MinimalSearchResults:
        """Search endpoint that returns top-k retrieved sources."""
        retriever = self._get_retriever(method)
        return retriever.retrieve_one(query, k=k)

    def answer(self, query: str, k: int = 5,
               method: Literal["lexical", "embedding"] = "lexical") -> MinimalAnswer:
        """
        Answer endpoint that uses the local model to generate an answer based on retrieved context.
        """

        retriever = self._get_retriever(method)
        result = retriever.retrieve_one(query, k=k)
        answer = self.answer_generator.answer_one(query, result.retrieved_sources)

        return MinimalAnswer(answer=answer,
                             question_id=str(uuid.uuid4()),
                             question=query,
                             retrieved_sources=result.retrieved_sources)


api_instance = RAGAPI()
app = api_instance.app
