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
        ensure_model_is_pulled(MODEL)

    def answer_one(self, question: str, sources: list[MinimalSource]) -> str:
        """
        Answer a single question based on provided sources.

        Args:
            question (str): The question to answer.
            sources (list[MinimalSource]): The retrieved snippets to use as context.

        Returns:
            str: The generated answer.
        """
        content = ""

        for source in sources:
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
                prompt=f"Answer the question based on passed information."
                       f" Your answer must be short and concise:"
                       f"Question: {question}\n"
                       f"Information: {content}",
                think=False,
                keep_alive=100
            )
        except Exception as e:
            print(f"Error communicating with Ollama client: {e}", file=sys.stderr)
            return ""

        s = ""

        i = 0

        for c in tqdm(resp_gen, desc="Generating answer", leave=False):
            s += c.response if c.response is not None else ""
            i += 1
            # if not i % 1000:
            #     print(s)
            # print(s)

        if not s:
            raise ValueError("No response from the model")

        return s
