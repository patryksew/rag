import uuid
from typing import Literal

from fastapi import FastAPI, HTTPException
from src.models import MinimalAnswer
from src.models import MinimalSearchResults
from src.retriever import Retriever


class RAGAPI:
    """RAG API Class that initializes resources once and routes requests."""

    def __init__(self, max_chunk_size: int = 2000):
        print("Initializing RAG API (Loading index and models)...")
        self.max_chunk_size = max_chunk_size

        from src.core import RAGCore
        self.core = RAGCore()

        # Preload retrievers
        self.core.preload_retriever("lexical", self.max_chunk_size)
        self.core.preload_retriever("embedding", self.max_chunk_size)

        # Preload model
        _ = self.core.answer_generator

        self.app = FastAPI()
        self._setup_routes()

    def _setup_routes(self) -> None:
        self.app.add_api_route("/search", self.search, methods=["GET"])
        self.app.add_api_route("/answer", self.answer, methods=["GET"])

    def _get_retriever(self, method: str) -> Retriever:
        if method not in ["lexical", "embedding"]:
            raise HTTPException(status_code=400, detail=f"Invalid method: {method}")

        try:
            return self.core.get_retriever(method, self.max_chunk_size)
        except ImportError as e:
            raise HTTPException(
                status_code=501,
                detail=f"Method '{method}' is not available. Missing dependencies? Error: {e}"
            )

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
        answer = self.core.answer_generator.answer_one(query, result.retrieved_sources)

        return MinimalAnswer(answer=answer,
                             question_id=str(uuid.uuid4()),
                             question=query,
                             retrieved_sources=result.retrieved_sources)


api_instance = RAGAPI()
app = api_instance.app
