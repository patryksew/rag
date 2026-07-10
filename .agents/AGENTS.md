# RAG against the machine - Agent Instructions

Welcome, Agent. You are pair programming with the USER on a 42 school project called "RAG against the machine".
This project involves building a Retrieval-Augmented Generation (RAG) system that indexes a codebase and uses a local model (Qwen/Qwen3-0.6B) to answer questions about it.

## Core Directives

1. **Rule Adherence**: The project must strictly follow the rules in `.agents/subject.txt`. Always verify project constraints before making logic changes.
2. **User Workflow**:
   - The USER prefers to write the bulk of the implementation code themselves, as their architectural approach often evolves during development.
   - **Write code ONLY when explicitly asked.** Do not preemptively generate implementation code.
   - **DO NOT** force or implement Test-Driven Development (TDD). 
   - **DO** assist by writing comprehensive tests (using `pytest` or `unittest`) *after* the USER completes a feature, when requested.
   - **If you save some kind of artifact, you need to create a copy at the repository root** (e.g., `chunking_plan.md`), in addition to creating the system artifacts.
3. **Architecture & Frameworks**: 
   - Use **Python 3.10+**.
   - Use **uv** as the project and package manager.
   - Implement the Command-Line Interface (CLI) using **Python Fire**.
   - Use **Pydantic** for data validation and models (structures exchanged between stages).
   - Use **tqdm** for progress bars on long-running operations.
4. **Code Quality & Linting**:
   - Adhere strictly to the `flake8` coding standard.
   - Use `mypy` for static type checking. All code must pass without errors using the strict flags required (`mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs`).
   - Include type hints for all function parameters, return types, and variables where applicable.
   - Ensure functions and classes are documented with docstrings following PEP 257 (e.g., Google or NumPy style).
   - Handle exceptions gracefully to avoid crashes. Prefer context managers for resources like files or connections to ensure automatic cleanup.
5. **Project Structure**:
   - Maintain the required layout: `src/` for the Python module (runnable via `uv run python -m src`), `pyproject.toml` and `uv.lock` at the root, and appropriate `data/` subdirectories (`data/raw/`, `data/processed/`, etc.) as defined in the subject.
   - Ensure a `Makefile` is provided with `install`, `run`, `debug`, `clean`, and `lint` rules.
