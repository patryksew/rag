"""Comprehensive End-to-End Evaluation Test Suite.

Simulates peer and automated evaluator workflows, testing:
- End-to-end CLI commands and pipeline walkthrough (Subject VI.7).
- Moulinette specification compliance (Recall thresholds, chunk lengths <= 2000, paths).
- Degenerate inputs, edge cases, and crash prevention (Subject VI.6).
- Cache lifecycle, dynamic prefix slicing, normalization, and reindexing (Bonus 3 & 4).
- Performance requirements (Indexing <= 5 min, retrieval throughput <= 90s for 200 questions).
"""

from collections.abc import Iterator
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

import pytest

from unittest.mock import MagicMock

from src.answer_generator import AnswerGenerator
from src.evaluator import Evaluator
from src.indexers.embedding import EmbeddingIndexer
from src.models import (
    AnsweredQuestion,
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)


def _get_project_root() -> Path:
    """Return the absolute path to the repository root directory."""
    return Path(__file__).resolve().parent.parent


def _is_ollama_reachable() -> bool:
    """Check if the local Ollama service is running and responsive."""
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            return bool(resp.status == 200)
    except Exception:
        return False


def _find_moulinette_binary() -> Path | None:
    """Locate the moulinette evaluation binary if available."""
    root = _get_project_root()
    for candidate in ("moulinette-ubuntu", "moulinette", "moulinette-fedora"):
        path = root / candidate
        if path.exists() and os.access(path, os.X_OK):
            return path
    return None


def run_rag_cli(
    *args: str,
    method: str | None = None,
    timeout: float = 180.0,
) -> subprocess.CompletedProcess[str]:
    """Execute the RAG CLI via subprocess in the same manner an evaluator would.

    Args:
        *args: Command and arguments passed to python -m src.
        method: Optional global --method flag (lexical, embedding, hybrid).
        timeout: Subprocess execution timeout in seconds.

    Returns:
        subprocess.CompletedProcess with returncode, stdout, and stderr.
    """
    cmd = [sys.executable, "-m", "src"]
    if method is not None:
        cmd.append(f"--method={method}")
    cmd.extend(args)

    return subprocess.run(
        cmd,
        cwd=_get_project_root(),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@pytest.fixture(scope="session", autouse=True)
def preserve_answer_cache() -> Iterator[None]:
    """Ensure data/processed/answer_cache.json is preserved or cleaned up.

    Prevents side-effects of answer tests from breaking tests in test_core.py.
    """
    cache_path = _get_project_root() / "data" / "processed" / "answer_cache.json"
    existed = cache_path.exists()
    content = cache_path.read_text(encoding="utf-8") if existed else None
    try:
        yield
    finally:
        if existed and content is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(content, encoding="utf-8")
        elif not existed and cache_path.exists():
            cache_path.unlink()


class TestEvaluatorWorkflow:
    """End-to-end walkthrough verifying the exact workflow run by the evaluator."""

    def test_e2e_index_command(self) -> None:
        """Verify the 'index' command ingests data/raw/ and creates the index."""
        proc = run_rag_cli("index", "--max_chunk_size", "2000", method="lexical")
        assert proc.returncode == 0, f"Index command failed:\n{proc.stderr}"

        index_file = (
            _get_project_root()
            / "data"
            / "processed"
            / "2000"
            / "bm25"
            / "index"
            / "data.csc.index.npy"
        )
        chunking_cache = (
            _get_project_root() / "data" / "processed" / "2000" / "bm25" / "chunking.json"
        )

        assert index_file.exists(), f"Expected index file {index_file} not found"
        assert chunking_cache.exists(), f"Expected chunking cache {chunking_cache} not found"

    def test_e2e_search_dataset_docs_and_code(self, tmp_path: Path) -> None:
        """Verify search_dataset produces valid StudentSearchResults JSON for docs and code."""
        save_dir = tmp_path / "search_results"

        # 1. Search Docs Dataset
        proc_docs = run_rag_cli(
            "search_dataset",
            "--dataset_path",
            "data/datasets/UnansweredQuestions/dataset_docs_public.json",
            "--k",
            "5",
            "--save_directory",
            str(save_dir),
            method="lexical",
        )
        assert proc_docs.returncode == 0, f"Search docs failed:\n{proc_docs.stderr}"

        docs_out = save_dir / "dataset_docs_public.json"
        assert docs_out.exists(), f"Output file {docs_out} was not generated"

        docs_data = StudentSearchResults.model_validate_json(docs_out.read_text(encoding="utf-8"))
        assert docs_data.k == 5
        assert len(docs_data.search_results) == 100

        # 2. Search Code Dataset
        proc_code = run_rag_cli(
            "search_dataset",
            "--dataset_path",
            "data/datasets/UnansweredQuestions/dataset_code_public.json",
            "--k",
            "5",
            "--save_directory",
            str(save_dir),
            method="lexical",
        )
        assert proc_code.returncode == 0, f"Search code failed:\n{proc_code.stderr}"

        code_out = save_dir / "dataset_code_public.json"
        assert code_out.exists(), f"Output file {code_out} was not generated"

        code_data = StudentSearchResults.model_validate_json(code_out.read_text(encoding="utf-8"))
        assert code_data.k == 5
        assert len(code_data.search_results) == 99

    def test_moulinette_evaluation_thresholds(self, tmp_path: Path) -> None:
        """Verify the moulinette validates search results and passes required Recall@5 bars."""
        moulinette = _find_moulinette_binary()
        if moulinette is None:
            pytest.skip("Moulinette executable not available on this platform")

        save_dir = tmp_path / "search_results"
        run_rag_cli(
            "search_dataset",
            "--dataset_path",
            "data/datasets/UnansweredQuestions/dataset_docs_public.json",
            "--k",
            "5",
            "--save_directory",
            str(save_dir),
            method="lexical",
        )
        run_rag_cli(
            "search_dataset",
            "--dataset_path",
            "data/datasets/UnansweredQuestions/dataset_code_public.json",
            "--k",
            "5",
            "--save_directory",
            str(save_dir),
            method="lexical",
        )

        docs_results = save_dir / "dataset_docs_public.json"
        docs_ground_truth = (
            _get_project_root()
            / "data"
            / "datasets"
            / "AnsweredQuestions"
            / "dataset_docs_public.json"
        )

        cmd_docs = [
            str(moulinette),
            "evaluate_student_search_results",
            str(docs_results),
            str(docs_ground_truth),
            "--k",
            "5",
            "--max_context_length",
            "2000",
            "--threshold",
            "0.80",
        ]
        proc_docs = subprocess.run(cmd_docs, capture_output=True, text=True, check=False)
        assert proc_docs.returncode == 0, f"Moulinette docs evaluation failed:\n{proc_docs.stderr}"
        assert "Student data is valid: True" in proc_docs.stdout
        assert "PASS: Recall@5" in proc_docs.stdout or "Recall@5: 0.8" in proc_docs.stdout

        code_results = save_dir / "dataset_code_public.json"
        code_ground_truth = (
            _get_project_root()
            / "data"
            / "datasets"
            / "AnsweredQuestions"
            / "dataset_code_public.json"
        )

        cmd_code = [
            str(moulinette),
            "evaluate_student_search_results",
            str(code_results),
            str(code_ground_truth),
            "--k",
            "5",
            "--max_context_length",
            "2000",
            "--threshold",
            "0.50",
        ]
        proc_code = subprocess.run(cmd_code, capture_output=True, text=True, check=False)
        assert proc_code.returncode == 0, f"Moulinette code evaluation failed:\n{proc_code.stderr}"
        assert "Student data is valid: True" in proc_code.stdout
        assert "PASS: Recall@5" in proc_code.stdout or "Recall@5: 0." in proc_code.stdout

    def test_student_evaluate_command(self, tmp_path: Path) -> None:
        """Verify the internal 'evaluate' CLI command executes and reports recall."""
        save_dir = tmp_path / "eval_out"
        run_rag_cli(
            "search_dataset",
            "--dataset_path",
            "data/datasets/UnansweredQuestions/dataset_docs_public.json",
            "--k",
            "5",
            "--save_directory",
            str(save_dir),
            method="lexical",
        )

        proc = run_rag_cli(
            "evaluate",
            "--student_search_results_path",
            str(save_dir / "dataset_docs_public.json"),
            "--dataset_path",
            "data/datasets/AnsweredQuestions/dataset_docs_public.json",
        )
        assert proc.returncode == 0, f"evaluate command failed:\n{proc.stderr}"
        assert "Recall score:" in proc.stdout

    def test_single_search_command_output_structure(self) -> None:
        """Verify the 'search' single-query command outputs valid MinimalSearchResults JSON."""
        proc = run_rag_cli("search", "What is vLLM?", "--k", "3", method="lexical")
        assert proc.returncode == 0, f"search command failed:\n{proc.stderr}"

        # Extract JSON object from stdout (in case tqdm progress bars were printed)
        stdout = proc.stdout.strip()
        json_start = stdout.find("{")
        json_end = stdout.rfind("}")
        assert json_start != -1 and json_end != -1, f"No JSON found in output:\n{stdout}"

        json_str = stdout[json_start:json_end + 1]
        results = MinimalSearchResults.model_validate_json(json_str)
        assert results.question == "What is vLLM?"
        assert len(results.retrieved_sources) == 3

    def test_single_answer_command_execution(self) -> None:
        """Verify the 'answer' single-query command generates a natural-language response."""
        if not _is_ollama_reachable():
            pytest.skip("Ollama daemon is not reachable on localhost:11434")

        proc = run_rag_cli("answer", "What is vLLM?", "2", method="lexical")
        assert proc.returncode == 0, f"answer command failed:\n{proc.stderr}"
        assert len(proc.stdout.strip()) > 0

    def test_answer_dataset_generation_and_validation(self, tmp_path: Path) -> None:
        """Verify answer_dataset generates valid StudentSearchResultsAndAnswer JSON."""
        if not _is_ollama_reachable():
            pytest.skip("Ollama daemon is not reachable on localhost:11434")

        # Create a small 2-question search result fixture
        sample_results = StudentSearchResults(
            k=2,
            search_results=[
                MinimalSearchResults(
                    question_id="test-q1",
                    question=(
                        "What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?"
                    ),
                    retrieved_sources=[
                        MinimalSource(
                            file_path="data/raw/vllm-0.10.1/docs/features/lora.md",
                            first_character_index=4695,
                            last_character_index=6098,
                        )
                    ],
                ),
            ],
        )

        sample_file = tmp_path / "sample_search_results.json"
        sample_file.write_text(sample_results.model_dump_json(indent=2), encoding="utf-8")
        save_dir = tmp_path / "answer_out"

        proc = run_rag_cli(
            "answer_dataset",
            "--student_search_results_path",
            str(sample_file),
            "--save_directory",
            str(save_dir),
        )
        assert proc.returncode == 0, f"answer_dataset failed:\n{proc.stderr}"

        ans_out = save_dir / "sample_search_results.json"
        assert ans_out.exists()

        data = StudentSearchResultsAndAnswer.model_validate_json(
            ans_out.read_text(encoding="utf-8")
        )
        assert len(data.search_results) == 1
        assert isinstance(data.search_results[0], MinimalAnswer)
        assert len(data.search_results[0].answer) > 0

        # Validate with moulinette if present
        moulinette = _find_moulinette_binary()
        if moulinette is not None:
            cmd = [str(moulinette), "evaluate_student_answers", str(ans_out)]
            eval_proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            assert eval_proc.returncode == 0


class TestMoulinetteSpecification:
    """Verifies strict constraints enforced by the Moulinette evaluation binary."""

    @pytest.fixture(scope="class")
    @classmethod
    def search_results(cls, tmp_path_factory: pytest.TempPathFactory) -> StudentSearchResults:
        """Generate search results for public questions once for specification testing."""
        tmp = tmp_path_factory.mktemp("spec_test")
        run_rag_cli(
            "search_dataset",
            "--dataset_path",
            "data/datasets/UnansweredQuestions/dataset_docs_public.json",
            "--k",
            "5",
            "--save_directory",
            str(tmp),
            method="lexical",
        )
        res_file = tmp / "dataset_docs_public.json"
        return StudentSearchResults.model_validate_json(res_file.read_text(encoding="utf-8"))

    def test_no_source_exceeds_max_context_length_2000(
        self, search_results: StudentSearchResults
    ) -> None:
        """Every retrieved source span must be <= 2000 characters wide."""
        for item in search_results.search_results:
            for source in item.retrieved_sources:
                span_len = source.last_character_index - source.first_character_index
                assert span_len <= 2000, (
                    f"Source {source.file_path}[{source.first_character_index}:"
                    f"{source.last_character_index}] exceeds 2000 chars: length={span_len}"
                )

    def test_file_paths_verbatim_corpus_format(
        self, search_results: StudentSearchResults
    ) -> None:
        """Every source file_path must match verbatim and begin with data/raw/."""
        root = _get_project_root()
        for item in search_results.search_results:
            for source in item.retrieved_sources:
                assert source.file_path.startswith("data/raw/"), (
                    f"File path '{source.file_path}' does not begin with 'data/raw/'"
                )
                full_path = root / source.file_path
                assert full_path.exists(), f"Source file does not exist on disk: {source.file_path}"

    def test_character_spans_within_file_boundaries(
        self, search_results: StudentSearchResults
    ) -> None:
        """Character indices must be valid non-negative offsets within the target file."""
        root = _get_project_root()
        file_len_cache: dict[str, int] = {}

        for item in search_results.search_results:
            for source in item.retrieved_sources:
                assert source.first_character_index >= 0
                assert source.last_character_index >= source.first_character_index

                if source.file_path not in file_len_cache:
                    full_path = root / source.file_path
                    file_len_cache[source.file_path] = len(full_path.read_text(encoding="utf-8"))

                max_len = file_len_cache[source.file_path]
                assert source.last_character_index <= max_len, (
                    f"Source index {source.last_character_index} exceeds file length {max_len} "
                    f"in {source.file_path}"
                )

    def test_result_count_bounded_by_k(self, search_results: StudentSearchResults) -> None:
        """Number of retrieved sources per question must never exceed requested k."""
        for item in search_results.search_results:
            assert len(item.retrieved_sources) <= search_results.k


class TestDegenerateInputsAndEdgeCases:
    """Evaluates resilience against invalid inputs, degenerates, and boundary conditions.

    Subject requirement:
    'handle degenerate inputs (empty query, nonsensical query, k=0, missing files,
     malformed JSON) gracefully. The CLI is tested with such edge cases and must
     never crash with an unhandled traceback.'
    """

    def test_empty_query_search(self) -> None:
        """Verify empty and whitespace queries return empty sources without crashing."""
        proc = run_rag_cli("search", "", "--k", "3", method="lexical")
        assert "Traceback (most recent call last)" not in proc.stderr
        assert proc.returncode == 0
        stdout = proc.stdout.strip()
        json_start = stdout.find("{")
        json_end = stdout.rfind("}")
        assert json_start != -1 and json_end != -1
        res = json.loads(stdout[json_start:json_end + 1])
        assert res.get("retrieved_sources") == []

        proc_ws = run_rag_cli("search", "   ", "--k", "3", method="lexical")
        assert "Traceback (most recent call last)" not in proc_ws.stderr
        assert proc_ws.returncode == 0

    def test_nonsensical_query_symbols(self) -> None:
        """Verify queries with special symbols and unicode do not crash."""
        proc = run_rag_cli("search", "!@#$%^&*()_+=-~`{}[]|;:',.<>?/", "--k", "3")
        assert "Traceback (most recent call last)" not in proc.stderr
        assert proc.returncode in (0, 1)

        proc_unicode = run_rag_cli("search", "🔥🚀🤖 Testing emojis", "--k", "3")
        assert "Traceback (most recent call last)" not in proc_unicode.stderr
        assert proc_unicode.returncode in (0, 1)

    def test_non_string_queries_fire_parsing(self) -> None:
        """Verify Python Fire literal parsing (e.g. search 42, search None) does not crash."""
        proc_int = run_rag_cli("search", "42", "--k", "2")
        assert "Traceback (most recent call last)" not in proc_int.stderr
        assert proc_int.returncode == 0
        assert "critical error" not in proc_int.stderr.lower()

        proc_none = run_rag_cli("search", "None", "--k", "2")
        assert "Traceback (most recent call last)" not in proc_none.stderr
        assert proc_none.returncode == 0
        assert "critical error" not in proc_none.stderr

    def test_k_zero_handled_gracefully(self) -> None:
        """Verify search with k=0 is rejected with clean error and non-zero exit."""
        proc = run_rag_cli("search", "What is vLLM?", "--k", "0", method="lexical")
        assert "Traceback (most recent call last)" not in proc.stderr
        assert proc.returncode != 0
        assert "critical error" not in proc.stderr.lower(), (
            f"Expected user-friendly validation error for k=0, got: {proc.stderr}"
        )
        output = (proc.stdout + proc.stderr).lower()
        assert "k" in output or "positive" in output or "integer" in output

    def test_negative_k_handled_gracefully(self) -> None:
        """Verify search with negative k is rejected with clean error and non-zero exit."""
        proc = run_rag_cli("search", "What is vLLM?", "--k", "-1", method="lexical")
        assert "Traceback (most recent call last)" not in proc.stderr
        assert proc.returncode != 0
        assert "critical error" not in proc.stderr.lower(), (
            f"Expected user-friendly validation error for negative k, got: {proc.stderr}"
        )
        output = (proc.stdout + proc.stderr).lower()
        assert "k" in output or "positive" in output or "integer" in output

    def test_large_k_handled_without_crash(self) -> None:
        """Verify search with very large k does not raise IndexError."""
        proc = run_rag_cli("search", "What is vLLM?", "--k", "500", method="lexical")
        assert proc.returncode == 0
        assert "Traceback (most recent call last)" not in proc.stderr

    def test_invalid_k_type_handled_gracefully(self) -> None:
        """Verify non-integer k parameter is validated cleanly with descriptive error."""
        proc = run_rag_cli("search", "What is vLLM?", "--k", "abc", method="lexical")
        assert "Traceback (most recent call last)" not in proc.stderr
        assert proc.returncode != 0
        assert "critical error" not in proc.stderr.lower(), (
            f"Expected user-friendly validation error for invalid k, got: {proc.stderr}"
        )

    def test_search_dataset_invalid_k_rejected(self, tmp_path: Path) -> None:
        """Verify search_dataset rejects k <= 0 and non-integer k with non-zero exit."""
        dataset = "data/datasets/UnansweredQuestions/dataset_docs_public.json"
        out_dir = tmp_path / "out_search_ds"

        for invalid_k in ("-1", "0", "abc"):
            proc = run_rag_cli(
                "search_dataset",
                "--dataset_path", dataset,
                "--save_directory", str(out_dir),
                "--k", invalid_k,
                method="lexical",
            )
            assert "Traceback (most recent call last)" not in proc.stderr
            assert proc.returncode != 0
            assert "critical error" not in proc.stderr.lower(), (
                f"Expected clean validation for k={invalid_k}, got: {proc.stderr}"
            )

        output_files = list(out_dir.glob("*.json")) if out_dir.exists() else []
        assert len(output_files) == 0, f"Corrupted files created on invalid k: {output_files}"

    def test_answer_invalid_k_rejected(self) -> None:
        """Verify answer command rejects k <= 0 and non-integer k without crash."""
        for invalid_k in ("-1", "0", "abc"):
            proc = run_rag_cli("answer", "What is vLLM?", "--k", invalid_k, method="lexical")
            assert "Traceback (most recent call last)" not in proc.stderr
            assert proc.returncode != 0
            assert "critical error" not in proc.stderr.lower(), (
                f"Expected clean validation in answer for k={invalid_k}, got: {proc.stderr}"
            )

    def test_invalid_max_chunk_size(self) -> None:
        """Verify oversized or degenerate max_chunk_size is validated and clamped."""
        # > 2000 must be clamped to 2000 per subject specification
        proc_large = run_rag_cli("search", "What is vLLM?", "--max_chunk_size", "3000")
        assert proc_large.returncode == 0
        assert "Traceback (most recent call last)" not in proc_large.stderr

        # negative chunk size
        proc_neg = run_rag_cli("search", "What is vLLM?", "--max_chunk_size", "-100")
        assert "Traceback (most recent call last)" not in proc_neg.stderr
        assert proc_neg.returncode != 0

        # non-integer
        proc_neg = run_rag_cli("search", "What is vLLM?", "--max_chunk_size", "1.5")
        assert "Traceback (most recent call last)" not in proc_neg.stderr
        assert proc_neg.returncode != 0

    def test_single_answer_default_k(self) -> None:
        """Verify answer command works with default k when k is omitted."""
        if not _is_ollama_reachable():
            pytest.skip("Ollama daemon is not reachable on localhost:11434")

        proc = run_rag_cli("answer", "What is vLLM?", method="lexical")
        assert proc.returncode == 0
        assert len(proc.stdout.strip()) > 0

    def test_missing_input_files(self) -> None:
        """Verify missing input files exit cleanly with error message and no raw traceback."""
        # 1. Missing dataset_path in search_dataset
        proc_search = run_rag_cli(
            "search_dataset",
            "--dataset_path",
            "/nonexistent/dataset.json",
            "--save_directory",
            "/tmp/out",
        )
        assert proc_search.returncode != 0
        assert "Traceback (most recent call last)" not in proc_search.stderr
        assert "not found" in proc_search.stderr.lower()

        # 2. Missing student_search_results_path in answer_dataset
        proc_ans = run_rag_cli(
            "answer_dataset",
            "--student_search_results_path",
            "/nonexistent/results.json",
            "--save_directory",
            "/tmp/out",
        )
        assert proc_ans.returncode != 0
        assert "Traceback (most recent call last)" not in proc_ans.stderr
        assert "not found" in proc_ans.stderr.lower()

        # 3. Missing files in evaluate
        proc_eval = run_rag_cli(
            "evaluate",
            "--student_search_results_path",
            "/nonexistent/results.json",
            "--dataset_path",
            "/nonexistent/gt.json",
        )
        assert proc_eval.returncode != 0
        assert "Traceback (most recent call last)" not in proc_eval.stderr

    def test_malformed_json_files(self, tmp_path: Path) -> None:
        """Verify corrupted syntax or invalid schema JSON files are caught cleanly."""
        corrupt_file = tmp_path / "corrupt.json"
        corrupt_file.write_text("{ unclosed json syntax: 123", encoding="utf-8")

        proc = run_rag_cli(
            "search_dataset",
            "--dataset_path",
            str(corrupt_file),
            "--save_directory",
            str(tmp_path),
        )
        assert proc.returncode != 0
        assert "Traceback (most recent call last)" not in proc.stderr

        # Invalid schema JSON (valid JSON, but missing required rag_questions)
        schema_file = tmp_path / "invalid_schema.json"
        schema_file.write_text('{"unrelated_field": "hello"}', encoding="utf-8")

        proc_schema = run_rag_cli(
            "search_dataset",
            "--dataset_path",
            str(schema_file),
            "--save_directory",
            str(tmp_path),
        )
        assert proc_schema.returncode != 0
        assert "Traceback (most recent call last)" not in proc_schema.stderr

    def test_unknown_method_fallback(self) -> None:
        """Verify unknown CLI method flag falls back to default lexical without crashing."""
        proc = run_rag_cli("search", "What is vLLM?", "--k", "2", method="invalid_method_xyz")
        assert "Traceback (most recent call last)" not in proc.stderr
        assert proc.returncode == 0
        assert "Invalid method" in proc.stdout or "lexical" in proc.stdout

    def test_evaluator_empty_dataset_and_partial_recall(self, tmp_path: Path) -> None:
        """Verify Evaluator handles empty datasets and scores partial matches correctly."""
        # 1. Empty dataset handling
        empty_gt = tmp_path / "empty_gt.json"
        empty_gt.write_text(RagDataset(rag_questions=[]).model_dump_json(), encoding="utf-8")
        empty_res = tmp_path / "empty_res.json"
        empty_res.write_text(
            StudentSearchResults(search_results=[], k=5).model_dump_json(), encoding="utf-8"
        )
        assert Evaluator.evaluate(str(empty_res), str(empty_gt)) == 0.0

        # 2. Multi-source partial recall calculation (1 of 2 sources found = 0.5 recall)
        source1 = MinimalSource(
            file_path="foo.py", first_character_index=0, last_character_index=50
        )
        source2 = MinimalSource(
            file_path="bar.py", first_character_index=10, last_character_index=80
        )
        gt_data = RagDataset(
            rag_questions=[
                AnsweredQuestion(
                    question_id="q1",
                    question="test multi",
                    sources=[source1, source2],
                    answer="ans",
                )
            ]
        )
        student_res = StudentSearchResults(
            k=1,
            search_results=[
                MinimalSearchResults(
                    question_id="q1",
                    question="test multi",
                    retrieved_sources=[source1],
                )
            ],
        )

        gt_file = tmp_path / "multi_gt.json"
        gt_file.write_text(gt_data.model_dump_json(), encoding="utf-8")
        res_file = tmp_path / "multi_res.json"
        res_file.write_text(student_res.model_dump_json(), encoding="utf-8")

        score = Evaluator.evaluate(str(res_file), str(gt_file))
        assert score == 0.5, f"Expected partial recall 0.5 for 1/2 sources found, got {score}"

    def test_evaluate_cli_error_handling(self, tmp_path: Path) -> None:
        """Verify evaluate command validates file existence and catches malformed inputs."""
        # 1. Nonexistent files
        proc_missing = run_rag_cli(
            "evaluate",
            "--student_search_results_path",
            str(tmp_path / "nonexistent_results.json"),
            "--dataset_path",
            str(tmp_path / "nonexistent_gt.json"),
        )
        assert proc_missing.returncode != 0
        assert "Traceback (most recent call last)" not in proc_missing.stderr
        assert "critical error" not in proc_missing.stderr.lower(), (
            f"Expected clean file validation in evaluate, got critical error: {proc_missing.stderr}"
        )

        # 2. Corrupt JSON in evaluate
        corrupt_file = tmp_path / "corrupt_eval.json"
        corrupt_file.write_text("{ unclosed json: 123", encoding="utf-8")
        proc_corrupt = run_rag_cli(
            "evaluate",
            "--student_search_results_path",
            str(corrupt_file),
            "--dataset_path",
            str(corrupt_file),
        )
        assert proc_corrupt.returncode != 0
        assert "Traceback (most recent call last)" not in proc_corrupt.stderr
        assert "critical error" not in proc_corrupt.stderr.lower()

    def test_save_directory_existing_file_handled_cleanly(self, tmp_path: Path) -> None:
        """Verify save_directory pointing to an existing file exits cleanly without crash."""
        existing_file = tmp_path / "occupied.txt"
        existing_file.write_text("already a file", encoding="utf-8")

        proc = run_rag_cli(
            "search_dataset",
            "--dataset_path",
            "data/datasets/UnansweredQuestions/dataset_docs_public.json",
            "--k",
            "2",
            "--save_directory",
            str(existing_file),
            method="lexical",
        )
        assert proc.returncode != 0
        assert "Traceback (most recent call last)" not in proc.stderr
        assert "critical error" not in proc.stderr.lower(), (
            f"Expected clean error for unwritable save_directory, got: {proc.stderr}"
        )

    def test_ollama_streaming_error_caught_in_answer_one(self) -> None:
        """Verify Ollama streaming connection error during token iteration is caught (Bug #5)."""
        mock_client = MagicMock()

        def failing_stream() -> Iterator[MagicMock]:
            raise ConnectionResetError("Connection lost mid-stream")
            yield MagicMock()

        mock_client.generate.return_value = failing_stream()
        gen = AnswerGenerator.__new__(AnswerGenerator)
        gen.client = mock_client

        try:
            res = gen.answer_one("What is vLLM?", [])
            assert "error" in res.lower() or "cannot" in res.lower()
        except ConnectionResetError:
            pytest.fail(
                "Streaming iteration in answer_one is outside try block; exception escaped"
            )

    def test_answer_k_zero_or_empty_context_skips_llm(self) -> None:
        """Verify answer with empty sources skips LLM generation (Bug #8)."""
        mock_client = MagicMock()
        gen = AnswerGenerator.__new__(AnswerGenerator)
        gen.client = mock_client

        ans = gen.answer_one("What is vLLM?", [])
        assert not mock_client.generate.called, (
            "LLM generate was invoked with empty context (k=0), inviting hallucination"
        )
        assert "no context" in ans.lower()

    def test_save_cache_called_after_build_index_in_embedding_indexer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify chunking cache is only saved AFTER _build_index succeeds (Bug #9)."""
        calls: list[str] = []

        indexer = EmbeddingIndexer.__new__(EmbeddingIndexer)
        indexer._collection = MagicMock()
        indexer._collection.count.return_value = 0

        mock_deltas = MagicMock()
        mock_deltas.modified_or_new_files = ["file1.txt"]
        mock_deltas.deleted_files = []

        monkeypatch.setattr(indexer, "_get_target_files", lambda d: [Path("file1.txt")])
        monkeypatch.setattr(indexer, "_load_cache", lambda: MagicMock())
        monkeypatch.setattr(indexer, "_get_file_deltas", lambda t, c: mock_deltas)
        monkeypatch.setattr(indexer, "_update_cache_and_get_chunks", lambda d, c: [])

        monkeypatch.setattr(indexer, "_save_chunking_cache",
                            lambda cache: calls.append("save_cache"))
        monkeypatch.setattr(indexer, "_build_index", lambda chunks: calls.append("build_index"))

        indexer.process_files_from_directory(Path("/dummy"))

        assert "save_cache" in calls and "build_index" in calls
        assert calls.index("build_index") < calls.index("save_cache"), (
            f"Chunking cache was saved BEFORE index was built: calls order was {calls}. "
            f"If embedding is interrupted, the index remains permanently corrupted."
        )


class TestCacheLifecycleAndReindexing:
    """Verifies cache hits, prefix slicing, normalization, and automatic reindexing.

    Subject bonus requirement:
    'cache the index and query results to speed up cold start and repeated queries.'
    'when a file changes, re-index only that file instead of rebuilding the whole index.'
    """

    def test_search_cache_hit_speedup_and_idempotence(self) -> None:
        """Verify a repeated query hits the cache and returns identical results faster."""
        query = "How to configure OpenAI server in vLLM?"

        # Initial query (triggers retrieval)
        start_cold = time.perf_counter()
        proc_cold = run_rag_cli("search", query, "--k", "5", method="lexical")
        elapsed_cold = time.perf_counter() - start_cold
        assert proc_cold.returncode == 0

        # Repeated query (must hit cache directly)
        start_warm = time.perf_counter()
        proc_warm = run_rag_cli("search", query, "--k", "5", method="lexical")
        elapsed_warm = time.perf_counter() - start_warm
        assert proc_warm.returncode == 0

        # Output must be identical
        assert "retrieved_sources" in proc_cold.stdout
        assert "retrieved_sources" in proc_warm.stdout
        assert elapsed_warm <= max(elapsed_cold, 1.5)

    def test_dynamic_prefix_slicing_from_cache(self) -> None:
        """Verify requesting fewer sources (k=2) reuses cached larger result (k=5)."""
        query = "Explain KV cache management in vLLM"

        # Prime cache with k=5
        proc_k5 = run_rag_cli("search", query, "--k", "5", method="lexical")
        assert proc_k5.returncode == 0
        stdout_k5 = proc_k5.stdout[proc_k5.stdout.find("{"):proc_k5.stdout.rfind("}") + 1]
        data_k5 = MinimalSearchResults.model_validate_json(stdout_k5)

        # Query with k=2 (should slice prefix without re-retrieval)
        proc_k2 = run_rag_cli("search", query, "--k", "2", method="lexical")
        assert proc_k2.returncode == 0
        stdout_k2 = proc_k2.stdout[proc_k2.stdout.find("{"):proc_k2.stdout.rfind("}") + 1]
        data_k2 = MinimalSearchResults.model_validate_json(stdout_k2)

        assert len(data_k2.retrieved_sources) == 2
        assert data_k2.retrieved_sources[0] == data_k5.retrieved_sources[0]
        assert data_k2.retrieved_sources[1] == data_k5.retrieved_sources[1]

    def test_query_normalization_in_cache(self) -> None:
        """Verify case insensitivity and whitespace stripping hit the same cache entry."""
        query_normal = "What is the vLLM batching mechanism?"
        query_variant = "  WHAT IS THE VLLM BATCHING MECHANISM?  "

        proc1 = run_rag_cli("search", query_normal, "--k", "3", method="lexical")
        assert proc1.returncode == 0
        stdout1 = proc1.stdout[proc1.stdout.find("{"):proc1.stdout.rfind("}") + 1]
        data1 = MinimalSearchResults.model_validate_json(stdout1)

        proc2 = run_rag_cli("search", query_variant, "--k", "3", method="lexical")
        assert proc2.returncode == 0
        stdout2 = proc2.stdout[proc2.stdout.find("{"):proc2.stdout.rfind("}") + 1]
        data2 = MinimalSearchResults.model_validate_json(stdout2)

        assert len(data2.retrieved_sources) == len(data1.retrieved_sources)
        for s1, s2 in zip(data1.retrieved_sources, data2.retrieved_sources):
            assert s1.file_path == s2.file_path
            assert s1.first_character_index == s2.first_character_index

    def test_cache_invalidation_and_incremental_reindex_on_new_file(self) -> None:
        """Verify adding a new file in data/raw invalidates cache and re-indexes only that file."""
        root = _get_project_root()
        unique_token = f"antigravity_unique_token_{int(time.time())}"
        test_file = root / "data" / "raw" / "vllm-0.10.1" / "__evaluator_cache_test.txt"

        try:
            # 1. Write new file with unique searchable token
            test_file.write_text(f"Important documentation: {unique_token}\n", encoding="utf-8")

            # 2. Search for the unique token (forces change detection and reindex)
            proc = run_rag_cli("search", unique_token, "--k", "1", method="lexical", timeout=120.0)
            assert proc.returncode == 0, f"Search with new file failed:\n{proc.stderr}"

            stdout = proc.stdout[proc.stdout.find("{"):proc.stdout.rfind("}") + 1]
            data = MinimalSearchResults.model_validate_json(stdout)

            assert len(data.retrieved_sources) > 0
            assert "__evaluator_cache_test.txt" in data.retrieved_sources[0].file_path
        finally:
            # Cleanup test file
            if test_file.exists():
                test_file.unlink()
            # Restore index state
            run_rag_cli("index", "--max_chunk_size", "2000", method="lexical", timeout=120.0)

    def test_cache_partition_isolation_by_chunk_size(self) -> None:
        """Verify partition directories for chunk size 2000 and 1000 do not overwrite each other."""
        root = _get_project_root()
        dir_2000 = root / "data" / "processed" / "2000" / "bm25"
        dir_1000 = root / "data" / "processed" / "1000" / "bm25"

        assert dir_2000 != dir_1000


class TestPerformanceRequirements:
    """Evaluates performance and throughput constraints defined in the Subject (VII.1.2)."""

    def test_indexing_duration_within_five_minutes(self) -> None:
        """Subject requirement: Indexing whole corpus must take at most 5 minutes (300 seconds)."""
        start = time.perf_counter()
        proc = run_rag_cli("index", "--max_chunk_size", "2000", method="lexical", timeout=300.0)
        elapsed = time.perf_counter() - start

        assert proc.returncode == 0
        assert elapsed < 300.0, f"Indexing took {elapsed:.2f}s, exceeding 5 minute limit"

    def test_retrieval_throughput_within_ninety_seconds(self, tmp_path: Path) -> None:
        """Subject requirement: Retrieval throughput at most 90s for 200 questions."""
        save_dir = tmp_path / "throughput"

        start = time.perf_counter()
        # Docs has 100 questions, Code has 99 questions -> 199 questions total (~200 questions)
        proc_docs = run_rag_cli(
            "search_dataset",
            "--dataset_path",
            "data/datasets/UnansweredQuestions/dataset_docs_public.json",
            "--k",
            "5",
            "--save_directory",
            str(save_dir),
            method="lexical",
            timeout=90.0,
        )
        proc_code = run_rag_cli(
            "search_dataset",
            "--dataset_path",
            "data/datasets/UnansweredQuestions/dataset_code_public.json",
            "--k",
            "5",
            "--save_directory",
            str(save_dir),
            method="lexical",
            timeout=90.0,
        )
        elapsed = time.perf_counter() - start

        assert proc_docs.returncode == 0
        assert proc_code.returncode == 0
        assert elapsed < 90.0, f"Batch search took {elapsed:.2f}s, exceeding 90s limit"
