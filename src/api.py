import logging
from typing import Literal

from fastapi import FastAPI, HTTPException
from src.core import RAGCore
from src.models import MinimalAnswer, MinimalSearchResults

logger = logging.getLogger(__name__)


class RAGAPI:
    """RAG API Class that initializes resources once and routes requests."""

    def __init__(self, max_chunk_size: int = 2000):
        print("Initializing RAG API (Loading index and models)...")
        self.max_chunk_size = max_chunk_size
        self.core = RAGCore()
        self.app = FastAPI()
        self._setup_routes()

    def _setup_routes(self) -> None:
        self.app.add_api_route("/search", self.search, methods=["GET"])
        self.app.add_api_route("/answer", self.answer, methods=["GET"])

    def search(
        self,
        query: str,
        k: int = 5,
        method: Literal["lexical", "embedding", "hybrid"] = "lexical"
    ) -> MinimalSearchResults:
        """Search endpoint that returns top-k retrieved sources."""
        try:
            return self.core.search(
                query=query,
                method=method,
                k=k,
                max_chunk_size=self.max_chunk_size
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e:
            logger.error("Method '%s' is unavailable: %s", method, e)
            raise HTTPException(
                status_code=503,
                detail=f"Method '{method}' is currently unavailable. Is Ollama running?"
            )
        except Exception as e:
            logger.exception("Unexpected error during search: %s", e)
            raise HTTPException(status_code=500, detail="Internal Server Error")

    def answer(
        self,
        query: str,
        k: int = 5,
        method: Literal["lexical", "embedding", "hybrid"] = "lexical"
    ) -> MinimalAnswer:
        """
        Answer endpoint that uses the local model to generate an answer based on retrieved context.
        """
        try:
            return self.core.answer(
                query=query,
                method=method,
                k=k,
                max_chunk_size=self.max_chunk_size
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e:
            logger.error("Answer generation unavailable for method '%s': %s", method, e)
            raise HTTPException(
                status_code=503,
                detail="Answer generation is currently unavailable. Is Ollama running?"
            )
        except Exception as e:
            logger.exception("Unexpected error during answer generation: %s", e)
            raise HTTPException(status_code=500, detail="Internal Server Error")


api_instance = RAGAPI()
app = api_instance.app
