from pathlib import Path
import sys

import fire

from src.answer_generator import AnswerGenerator
from src.evaluator import Evaluator
from src.models import RagDataset, StudentSearchResults
from src.retriever import Retriever


class RAGCLI:
    """RAG against the machine Command-Line Interface."""

    def __init__(self, method: str = "lexical") -> None:
        import os
        os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

        self.project_root = Path(__file__).resolve().parent.parent

        if method not in ("lexical", "embedding"):
            print(">" * 20, f" Invalid method '{method}'. Using 'lexical' instead.", "<" * 20,
                  sep="")
            self.method = "lexical"
        else:
            self.method = method

    @staticmethod
    def _validate_max_chunk_size(max_chunk_size: int | str) -> int:
        """Validate the maximum chunk size parameter."""
        try:
            valid_size = int(max_chunk_size)
        except (ValueError, TypeError):
            print(">" * 20, "Invalid max_chunk_size. Using default value.", "<" * 20, sep="")
            valid_size = 2000

        if valid_size > 2000:
            print(">" * 20, "max_chunk_size cannot be greater than 2000. Using 2000 instead.",
                  "<" * 20, sep="")
            valid_size = 2000

        return valid_size

    def _index(self, max_chunk_size: int) -> Retriever:
        """
        Indexes files from a specified directory.

        Parameters:
        max_chunk_size: int
            The maximum size of chunks in the indexed data.

        Returns:
        Retriever
            An instance of a Retriever object after processing the files.
        """
        from src.indexers.base import Indexer
        indexer: Indexer
        if self.method == "embedding":
            from src.indexers.embedding import EmbeddingIndexer
            indexer = EmbeddingIndexer(max_chunk_size)
        else:
            from src.indexers.bm25 import BM25Indexer
            indexer = BM25Indexer(max_chunk_size)

        return indexer.process_files_from_directory(self.project_root / "data" / "raw")

    def index(self, max_chunk_size: int = 2000) -> None:
        """
        Ingest data/raw/ and build the index under data/processed/.

        Args:
            max_chunk_size: Maximum chunk size in characters (default: 2000).
        """

        # # REMOVE THIS PART !!!!!!!!!!!!!!!!!
        # processed_dir = self.project_root / "data" / "processed"
        # if processed_dir.exists():
        #     import shutil
        #     shutil.rmtree(processed_dir)

        max_chunk_size = self._validate_max_chunk_size(max_chunk_size)
        self._index(max_chunk_size)

    def search(self, query: str, k: int = 5, max_chunk_size: int = 2000) -> None:
        """
        Return the top-k sources for a single query.

        Args:
            query: The question to search for.
            k: The number of top sources to return.
            max_chunk_size: Maximum chunk size used during indexing (default: 2000).
        """
        max_chunk_size = self._validate_max_chunk_size(max_chunk_size)
        retriever = self._index(max_chunk_size)
        result = retriever.retrieve_one(query, k=k)
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

        retriever = self._index(max_chunk_size)
        results = retriever.retrieve(dataset.rag_questions, k)

        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        dataset_name = Path(dataset_path).name

        try:
            with open(save_path / dataset_name, "w", encoding="utf-8") as f:
                f.write(results.model_dump_json(indent=2))
        except Exception as e:
            print(f"Error saving results to {save_path / dataset_name}: {e}", file=sys.stderr)
            sys.exit(1)

    def answer(self, query: str, k: int, max_chunk_size: int = 2000) -> None:
        """
        Answer a single query using the retrieved context.

        Args:
            query: The question to answer.
            k: The number of retrieved context chunks to use.
            max_chunk_size: Maximum chunk size used during indexing (default: 2000).
        """
        max_chunk_size = self._validate_max_chunk_size(max_chunk_size)
        retriever = self._index(max_chunk_size)
        result = retriever.retrieve_one(query, k)

        answer_generator = AnswerGenerator()
        print(answer_generator.answer_one(query, result.retrieved_sources))

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

        result = AnswerGenerator().answer_dataset(student_search_results)

        file_name = Path(student_search_results_path).name
        Path(save_directory).mkdir(parents=True, exist_ok=True)

        try:
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
        print(f"Recall score: {Evaluator.evaluate(student_search_results_path, dataset_path)}")


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
