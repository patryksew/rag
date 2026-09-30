from pathlib import Path

from src.chunker import Chunker
from src.indexers.base import Indexer
from src.indexers.bm25 import BM25Indexer
from src.indexers.embedding import EmbeddingIndexer
from src.retriever import Retriever, HybridRetriever


class HybridIndexer(Indexer):
    def __init__(self, bm25_indexer: BM25Indexer, embedding_indexer: EmbeddingIndexer,
                 max_chunk_size: int = 2000, chunker: Chunker | None = None):
        if chunker is None:
            chunker = Chunker(max_chunk_size)

        super().__init__(max_chunk_size, chunker)

        self.bm25_indexer = bm25_indexer
        self.embedding_indexer = embedding_indexer

    def process_files_from_directory(self, directory: Path) -> Retriever:
        bm25_retriever = self.bm25_indexer.process_files_from_directory(directory)
        embedding_retriever = self.embedding_indexer.process_files_from_directory(directory)

        return HybridRetriever(bm25_retriever, embedding_retriever)
