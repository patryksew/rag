import uuid
from typing import List

from pydantic import BaseModel, ConfigDict, Field


class MinimalSource(BaseModel):
    """
    Represents a single source of information retrieved from the index.

    Attributes:
        file_path (str): The path to the source file, relative to project root.
        first_character_index (int): Starting character position of the chunk.
        last_character_index (int): Ending character position of the chunk.
    """
    model_config = ConfigDict(frozen=True)

    file_path: str
    first_character_index: int = Field(ge=0)
    last_character_index: int = Field(ge=0)


class ChunkData(MinimalSource):
    """
    Represents a chunk of data with its text content.

    Attributes:
        content (str): The chunk's text content.
    """
    content: str


class UnansweredQuestion(BaseModel):
    """
    Represents an unanswered question.

    Attributes:
        question_id (str): Unique identifier for the question.
        question (str): The question text.
    """
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """
    Represents a question that has been answered.

    Attributes:
        sources (List[MinimalSource]): The retrieved snippets used to answer.
        answer (str): The generated answer.
    """
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """
    Represents a dataset of RAG questions.

    Attributes:
        rag_questions (List[AnsweredQuestion | UnansweredQuestion]): The questions.
    """
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """
    Represents the search results for a single query.

    Attributes:
        question_id (str): The ID of the question.
        question (str): The question text.
        retrieved_sources (List[MinimalSource]): The retrieved snippets.
    """
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """
    Represents a generated answer combined with its search results.

    Attributes:
        answer (str): The generated answer.
    """
    answer: str


class StudentSearchResults(BaseModel):
    """
    Represents search results for a dataset of questions.

    Attributes:
        search_results (List[MinimalSearchResults]): The results.
        k (int): Number of sources requested per question.
    """
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """
    Represents answers and search results for a dataset of questions.

    Attributes:
        search_results (List[MinimalAnswer]): The answers.
        k (int): Number of sources requested per question.
    """
    search_results: List[MinimalAnswer] = Field(default_factory=list)
    k: int
