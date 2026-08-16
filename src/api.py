import uuid
from pathlib import Path
from fastapi import FastAPI
from src.models import MinimalAnswer
from src.answer_generator import AnswerGenerator
from src.indexer import BM25Indexer
from src.models import MinimalSearchResults


class RAGAPI:
    """RAG API Class that initializes resources once and routes requests."""

    def __init__(self, max_chunk_size: int = 2000):
        print("Initializing RAG API (Loading index and models)...")
        self.project_root = Path(__file__).resolve().parent.parent
        self.max_chunk_size = max_chunk_size

        indexer = BM25Indexer(self.max_chunk_size)

        self.retriever = indexer.process_files_from_directory(self.project_root / "data" / "raw")

        self.answer_generator = AnswerGenerator()

        self.app = FastAPI()
        self._setup_routes()

    def _setup_routes(self) -> None:
        self.app.add_api_route("/search", self.search, methods=["GET"])
        self.app.add_api_route("/answer", self.answer, methods=["GET"])

    def search(self, query: str, k: int = 5) -> MinimalSearchResults:
        """Search endpoint that returns top-k retrieved sources."""
        return self.retriever.retrieve_one(query, k=k)

    def answer(self, query: str, k: int = 5) -> MinimalAnswer:
        """
        Answer endpoint that uses the local model to generate an answer based on retrieved context.
        """

        result = self.retriever.retrieve_one(query, k=k)
        answer = self.answer_generator.answer_one(query, result.retrieved_sources)

        return MinimalAnswer(answer=answer,
                             question_id=str(uuid.uuid4()),
                             question=query,
                             retrieved_sources=result.retrieved_sources)


api_instance = RAGAPI()
app = api_instance.app
