from pathlib import Path
import sys
from typing import Any

import fire

from src.evaluator import Evaluator
from src.models import RagDataset, StudentSearchResults


class RAGCLI:
    """RAG against the machine Command-Line Interface."""

    def __init__(self, method: str = "lexical") -> None:
        from src.core import RAGCore
        self.core = RAGCore()

        if method not in ("lexical", "embedding", "hybrid"):
            print(">" * 20, f" Invalid method '{method}'. Using 'lexical' instead.", "<" * 20,
                  sep="")
            self.method = "lexical"
        else:
            self.method = method

    @staticmethod
    def _validate_max_chunk_size(max_chunk_size: int | str) -> int:
        """Validate the maximum chunk size parameter."""

        if type(max_chunk_size) is not int:
            print("Invalid max_chunk_size. Must be an integer")
            sys.exit(1)

        if int(max_chunk_size) < 1:
            print("Invalid max_chunk_size. Must be a positive integer")
            sys.exit(1)

        if max_chunk_size > 2000:
            print(">" * 20, "max_chunk_size cannot be greater than 2000. Using 2000 instead.",
                  "<" * 20, sep="")
            max_chunk_size = 2000

        return max_chunk_size

    def index(self, max_chunk_size: int = 2000) -> None:
        """
        Ingest data/raw/ and build the index under data/processed/.

        Args:
            max_chunk_size: Maximum chunk size in characters (default: 2000).
        """
        max_chunk_size = self._validate_max_chunk_size(max_chunk_size)
        self.core.index(self.method, max_chunk_size)

    def search(self, query: Any, k: int = 5, max_chunk_size: int = 2000) -> None:
        """
        Return the top-k sources for a single query.

        Args:
            query: The question to search for.
            k: The number of top sources to return.
            max_chunk_size: Maximum chunk size used during indexing (default: 2000).
        """
        query_str = str(query)
        max_chunk_size = self._validate_max_chunk_size(max_chunk_size)
        try:
            result = self.core.search(query_str, self.method, k=k, max_chunk_size=max_chunk_size)
        except (ValueError, TypeError) as e:
            print(e)
            sys.exit(1)
        print(result.model_dump_json(indent=4))

    def search_dataset(self, dataset_path: str, save_directory: str,
                       max_chunk_size: int = 2000, k: int = 5) -> None:
        """
        Run search over a whole dataset and write a StudentSearchResults JSON file.

        Args:
            dataset_path: Path to the dataset JSON file.
            k: The number of top sources to return per question.
            save_directory: Directory to save the results.
            max_chunk_size: Maximum chunk size used during indexing (default: 2000).
        """
        max_chunk_size = self._validate_max_chunk_size(max_chunk_size)
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                dataset = RagDataset.model_validate_json(f.read())
        except FileNotFoundError:
            print(f"Error: Dataset file not found at {dataset_path}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error reading or parsing dataset file: {e}", file=sys.stderr)
            sys.exit(1)

        try:
            results = self.core.search_dataset(
                dataset, self.method, k=k, max_chunk_size=max_chunk_size
            )
        except (ValueError, TypeError) as e:
            print(e)
            sys.exit(1)

        dataset_name = Path(dataset_path).name
        save_path = Path(save_directory)

        try:
            save_path.mkdir(parents=True, exist_ok=True)
            with open(save_path / dataset_name, "w", encoding="utf-8") as f:
                f.write(results.model_dump_json(indent=2))
        except Exception as e:
            print(f"Error saving results to {save_path / dataset_name}: {e}", file=sys.stderr)
            sys.exit(1)

    def answer(self, query: Any, k: int = 5, max_chunk_size: int = 2000) -> None:
        """
        Answer a single query using the retrieved context.

        Args:
            query: The question to answer.
            k: The number of retrieved context chunks to use.
            max_chunk_size: Maximum chunk size used during indexing (default: 2000).
        """
        query_str = str(query)
        max_chunk_size = self._validate_max_chunk_size(max_chunk_size)
        try:
            result = self.core.answer(query_str, self.method, k=k, max_chunk_size=max_chunk_size)
        except (ValueError, TypeError) as e:
            print(e)
            sys.exit(1)
        print(result.answer)

    def answer_dataset(
        self, student_search_results_path: str, save_directory: str
    ) -> None:
        """
        Generate answers for a dataset, producing a StudentSearchResultsAndAnswer JSON file.

        Args:
            student_search_results_path: Path to the search results JSON file.
            save_directory: Directory to save the answers.
        """

        try:
            with open(student_search_results_path, "r") as f:
                student_search_results = StudentSearchResults.model_validate_json(f.read())
        except FileNotFoundError:
            print(
                f"Error: Search results file not found at {student_search_results_path}",
                file=sys.stderr
            )
            sys.exit(1)
        except Exception as e:
            print(f"Error reading or parsing search results file: {e}", file=sys.stderr)
            sys.exit(1)

        result = self.core.answer_dataset(student_search_results)

        file_name = Path(student_search_results_path).name

        try:
            Path(save_directory).mkdir(parents=True, exist_ok=True)
            with open(f"{save_directory}/{file_name}", "w") as f:
                f.write(result.model_dump_json(indent=2))
        except Exception as e:
            print(f"Error saving answers to {save_directory}/{file_name}: {e}", file=sys.stderr)
            sys.exit(1)

    def evaluate(self, student_search_results_path: str, dataset_path: str) -> None:
        """
        Report your own recall@k against a ground-truth dataset, for your own testing.

        Args:
            student_search_results_path: Path to your search results JSON file.
            dataset_path: Path to the ground-truth dataset JSON file.
        """
        try:
            print(f"Recall score: {Evaluator.evaluate(student_search_results_path, dataset_path)}")
        except Exception:
            print("Could not evaluate")
            sys.exit(1)


def main() -> None:
    """
    Entry point for the RAG CLI.
    """
    try:
        fire.Fire(RAGCLI)
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(
            f"A critical error occurred that prevented the program from running: {e}",
            file=sys.stderr
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
