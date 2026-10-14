.PHONY: install run api index search-dataset search-docs search-code \
	answer  answer-docs answer-code evaluate-docs evaluate-code debug lint test clean

# CLI configurable parameters
METHOD ?= lexical
MAX_CHUNK_SIZE ?= 2000
K ?= 5
QUERY ?= What is vLLM?

# Datasets
DOCS_DATASET_PATH ?= data/datasets/UnansweredQuestions/dataset_docs_public.json
CODE_DATASET_PATH ?= data/datasets/UnansweredQuestions/dataset_code_public.json
DATASET_PATH ?= $(DOCS_DATASET_PATH)

# Output directories
SAVE_DIRECTORY ?= data/output/search_results/UnansweredQuestions
ANSWER_SAVE_DIRECTORY ?= data/output/search_results_and_answer/UnansweredQuestions

# Student search results
DOCS_STUDENT_RESULTS_PATH ?= data/output/search_results/UnansweredQuestions/dataset_docs_public.json
CODE_STUDENT_RESULTS_PATH ?= data/output/search_results/UnansweredQuestions/dataset_code_public.json
STUDENT_SEARCH_RESULTS_PATH ?= $(DOCS_STUDENT_RESULTS_PATH)

# Ground truth datasets for evaluation
DOCS_GROUND_TRUTH_PATH ?= data/datasets/AnsweredQuestions/dataset_docs_public.json
CODE_GROUND_TRUTH_PATH ?= data/datasets/AnsweredQuestions/dataset_code_public.json
EVAL_DATASET_PATH ?= $(DOCS_GROUND_TRUTH_PATH)

install:
	uv sync

run:
	uv run python -m src $(ARGS)

api:
	uv run uvicorn src.api:app --reload

index:
	uv run python -m src --method=$(METHOD) index --max_chunk_size=$(MAX_CHUNK_SIZE)

search:
	uv run python -m src --method=$(METHOD) search "$(QUERY)" --k=$(K) --max_chunk_size=$(MAX_CHUNK_SIZE)

search-docs:
	uv run python -m src --method=$(METHOD) search_dataset --dataset_path=$(DOCS_DATASET_PATH) --save_directory=$(SAVE_DIRECTORY) --k=$(K) --max_chunk_size=$(MAX_CHUNK_SIZE)

search-code:
	uv run python -m src --method=$(METHOD) search_dataset --dataset_path=$(CODE_DATASET_PATH) --save_directory=$(SAVE_DIRECTORY) --k=$(K) --max_chunk_size=$(MAX_CHUNK_SIZE)

answer:
	uv run python -m src --method=$(METHOD) answer "$(QUERY)" $(K) --max_chunk_size=$(MAX_CHUNK_SIZE)

answer-docs:
	uv run python -m src answer_dataset --student_search_results_path=$(DOCS_STUDENT_RESULTS_PATH) --save_directory=$(ANSWER_SAVE_DIRECTORY)

answer-code:
	uv run python -m src answer_dataset --student_search_results_path=$(CODE_STUDENT_RESULTS_PATH) --save_directory=$(ANSWER_SAVE_DIRECTORY)

evaluate-docs:
	uv run python -m src evaluate --student_search_results_path=$(DOCS_STUDENT_RESULTS_PATH) --dataset_path=$(DOCS_GROUND_TRUTH_PATH)

evaluate-code:
	uv run python -m src evaluate --student_search_results_path=$(CODE_STUDENT_RESULTS_PATH) --dataset_path=$(CODE_GROUND_TRUTH_PATH)

debug:
	uv run python -m pdb -m src

lint:
	@status=0; \
	uv run flake8 . || status=1; \
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs || status=1; \
	exit $$status

test:
	uv run pytest

clean:
	rm -rf .venv
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
