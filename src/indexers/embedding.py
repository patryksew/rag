from pathlib import Path
from typing import cast

import chromadb
from chromadb import Where, EmbeddingFunction
from chromadb.utils import embedding_functions
from tqdm import tqdm

from src.chunker import Chunker
from src.retriever import Retriever, EmbeddingRetriever
from src.models import ChunkData
from src.indexers.base import Indexer


class EmbeddingIndexer(Indexer):

    @staticmethod
    def _get_embedding_function() -> EmbeddingFunction:
        try:
            import torch.cuda
            if torch.cuda.is_available():
                return embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="BAAI/bge-small-en-v1.5",
                    device="cuda"
                )
        except ImportError:
            pass
        return embedding_functions.DefaultEmbeddingFunction()

    # noinspection deprecation
    # because of embedding_fn.name(), which is incorrectly marked as deprecated
    def __init__(self, max_chunk_size: int = 2000, chunker: Chunker | None = None) -> None:

        if chunker is None:
            chunker = Chunker(max_chunk_size)

        super().__init__(max_chunk_size, chunker)

        embedding_fn = self._get_embedding_function()
        self._processed_path = self._processed_path / "embedding" / embedding_fn.name()
        self._cache_path = (
            self._cache_path.parent / "embedding" / embedding_fn.name() / self._cache_path.name)

        self._client = chromadb.PersistentClient(self._processed_path)
        self._collection = self._client.get_or_create_collection(
            name="embeddings",
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def _remove_old_files(self, deleted_files: list[str]) -> None:
        if not deleted_files:
            return

        BATCH_SIZE = 100
        for i in tqdm(range(0, len(deleted_files), BATCH_SIZE), desc="Deleting files"):
            batch = deleted_files[i:i + BATCH_SIZE]
            self._collection.delete(where=cast(Where, cast(object, {"file_path": {"$in": batch}})))

    def _build_index(self, chunked_files: list[ChunkData]) -> None:
        BATCH_SIZE = 200

        for i in tqdm(range(0, len(chunked_files), BATCH_SIZE), desc="Embedding chunks",
                      leave=False):
            chunks_batch = chunked_files[i:i + BATCH_SIZE]
            self._collection.upsert(
                ids=[f"{chunk.file_path}_{chunk.first_character_index}" for chunk in chunks_batch],
                documents=[chunk.content for chunk in chunks_batch],
                metadatas=[
                    {
                        "file_path": chunk.file_path,
                        "first_character_index": chunk.first_character_index,
                        "last_character_index": chunk.last_character_index,
                    }
                    for chunk in chunks_batch
                ]
            )

    def process_files_from_directory(self, directory: Path) -> Retriever:
        target_files = self._get_target_files(directory)
        cache = self._load_cache()
        deltas = self._get_file_deltas(target_files, cache)

        is_modified = len(deltas.modified_or_new_files) > 0 or len(deltas.deleted_files) > 0
        index_exists = self._collection.count() > 0

        if not is_modified and index_exists:
            return EmbeddingRetriever(self._collection)

        chunked_files = self._update_cache_and_get_chunks(deltas, cache)

        if index_exists:
            chunked_files = [
                chunk for chunk in chunked_files if
                chunk.file_path in deltas.modified_or_new_files
            ]

        if is_modified:
            self._save_cache(cache)

        if index_exists:
            files_to_remove = deltas.deleted_files + deltas.modified_or_new_files
            self._remove_old_files(files_to_remove)

        self._build_index(chunked_files)

        return EmbeddingRetriever(self._collection)
