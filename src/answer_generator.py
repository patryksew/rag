import ollama
import sys
from tqdm import tqdm

from src.models import MinimalSource
from src.ollama_utils import ensure_model_is_pulled

MODEL = "Qwen3:0.6b"


class AnswerGenerator:
    def __init__(self, host: str | None = None) -> None:
        """
        Initialize the AnswerGenerator.

        Args:
            host (str | None): The host for the Ollama client (default: None for default client).
        """
        try:
            self.client = self._get_client(host)
        except Exception as e:
            raise ValueError(f"Failed to connect to Ollama server: {e}")
        self._ensure_model_is_pulled()

    @staticmethod
    def _get_client(host: str | None = None) -> ollama.Client:
        """
        Creates a Client and checks connection.
        If it doesn't work, tries with the default host (127.0.0.1:11434).

        Args:
            host (str | None): Ollama host.

        Returns:
            ollama.Client: Connected Ollama client.
        """

        client = ollama.Client(host=host)
        try:
            client.list()
            return client
        except Exception:
            print(
                "Warning: Failed to connect to Ollama server. "
                "Trying with default host: 127.0.0.1:11434")
            client = ollama.Client(host="127.0.0.1:11434")
            client.list()

        return client

    def _ensure_model_is_pulled(self) -> None:
        """
        Ensures that the specified model is pulled.
        """
        ensure_model_is_pulled(MODEL, client=self.client)

    def answer_one(self, question: str, sources: list[MinimalSource]) -> str:
        """
        Answer a single question based on provided sources.

        Args:
            question (str): The question to answer.
            sources (list[MinimalSource]): The retrieved snippets to use as context.

        Returns:
            str: The generated answer.
        """
        if not sources:
            print("Cannot generate answer with no context")
            return "Cannot generate answer with no context"

        content = ""

        # Ollama will truncate prompt if too long,
        # reversing to have the least important sources truncated
        reversed_sources = sources.copy()
        reversed_sources.reverse()

        for source in reversed_sources:
            try:
                with open(source.file_path, "r", encoding="utf-8") as f:
                    content += "\n" + f.read()[
                        source.first_character_index:source.last_character_index + 1]
            except Exception as e:
                print(
                    f"Warning: Failed to read source file {source.file_path}: {e}",
                    file=sys.stderr
                )

        try:
            resp_gen = self.client.generate(
                stream=True,
                model=MODEL,
                prompt=f"Information: {content}"
                       f"Answer the question based on passed information."
                       f" Your answer must be short and concise:"
                       f"Question: {question}\n",
                think=False,
                keep_alive=100
            )
        except Exception as e:
            print(f"Error communicating with Ollama client: {e}", file=sys.stderr)
            return f"Error communicating with Ollama client: {e}"

        s = ""

        i = 0
        try:
            for c in tqdm(resp_gen, desc="Generating answer", leave=False):
                s += c.response if c.response is not None else ""
                i += 1
        except Exception as e:
            print(f"Error communicating with Ollama client: {e}", file=sys.stderr)
            return f"Error communicating with Ollama client: {e}"

        if not s:
            return "Unable to get answer from Ollama"

        return s
