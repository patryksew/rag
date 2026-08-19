from unittest.mock import patch
from pathlib import Path
import pytest
from typing import Iterator, cast, Any, Dict

import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from chromadb.errors import NotFoundError
from src.indexer import EmbeddingIndexer
from src.retriever import EmbeddingRetriever


class DummyEmbeddingFunction(EmbeddingFunction):

    # noinspection missing-constructor,protocol
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @staticmethod
    def name() -> str:
        return "dummy_embedding_function"

    def __call__(self, input: Documents) -> Embeddings:
        return cast(Embeddings, cast(object, [[1.0] * 384 for _ in input]))

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> EmbeddingFunction:
        return DummyEmbeddingFunction()


class TestEmbeddingIndexer:

    @pytest.fixture(autouse=True)
    def setup_mocks(self) -> Iterator[None]:
        self.ephemeral_client = chromadb.EphemeralClient()
        try:
            self.ephemeral_client.delete_collection("embeddings")
        except NotFoundError:
            pass
        with (patch("src.indexer.chromadb.PersistentClient") as mock_client,
              patch.object(EmbeddingIndexer, "_get_embedding_function") as mock_embed_fn):
            mock_client.return_value = self.ephemeral_client
            mock_embed_fn.return_value = DummyEmbeddingFunction()
            self.mock_client = mock_client
            self.mock_embed_fn = mock_embed_fn
            yield

    def test_init_sets_up_chromadb(self) -> None:
        _ = EmbeddingIndexer(max_chunk_size=100)

        self.mock_client.assert_called_once()
        self.mock_embed_fn.assert_called_once()
        collection = self.ephemeral_client.get_collection("embeddings")
        assert collection is not None

    def test_process_files_from_directory_integration(self, tmp_path: Path) -> None:
        # Setup files in tmp_path
        file1 = tmp_path / "hello.py"
        file1.write_text("def hello(): pass")

        file2 = tmp_path / "world.py"
        file2.write_text("def world(): pass")

        indexer = EmbeddingIndexer(max_chunk_size=100)
        indexer._cache_path = tmp_path / "cache.json"

        retriever = indexer.process_files_from_directory(tmp_path)
        assert isinstance(retriever, EmbeddingRetriever)

        # Verify initial indexing
        res = retriever.retrieve_one("hello", k=2)
        assert len(res.retrieved_sources) == 2

        file_paths = [s.file_path for s in res.retrieved_sources]
        assert any("hello.py" in p for p in file_paths)
        assert any("world.py" in p for p in file_paths)

        # Modify filesystem
        file1.unlink()
        file2.write_text("def world_modified(): pass")

        retriever = indexer.process_files_from_directory(tmp_path)

        # Verify index updates
        res_after = retriever.retrieve_one("hello", k=2)
        assert len(res_after.retrieved_sources) == 1
        assert "world.py" in res_after.retrieved_sources[0].file_path
        assert "hello.py" not in [s.file_path for s in res_after.retrieved_sources]

    def test_process_empty_directory(self, tmp_path: Path) -> None:
        indexer = EmbeddingIndexer(max_chunk_size=100)
        indexer._cache_path = tmp_path / "cache.json"

        retriever = indexer.process_files_from_directory(tmp_path)
        assert isinstance(retriever, EmbeddingRetriever)

        res = retriever.retrieve_one("test", k=1)
        assert len(res.retrieved_sources) == 0
