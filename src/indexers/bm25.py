from pathlib import Path

import bm25s
from tqdm import tqdm

from src.chunker import Chunker
from src.retriever import BM25Retriever
from src.models import ChunkData
from src.text_utils import preprocess_text
from src.indexers.base import Indexer


class BM25Indexer(Indexer):
    """
    Indexer implementation using the BM25 algorithm.
    """

    def __init__(self, max_chunk_size: int, chunker: Chunker | None = None, k1: float = 2.0,
                 b: float = 0.9, delta: float = 0.5, method: str = 'lucene'):
        """
        Initialize the BM25Indexer.

        Args:
            max_chunk_size (int): The maximum chunk size in characters.
            chunker (Chunker | None): Optional chunker instance.
            k1 (float): The k1 parameter for BM25.
            b (float): The b parameter for BM25.
            delta (float): The delta parameter for BM25.
            method (str): The scoring method for BM25.
        """
        if chunker is None:
            chunker = Chunker(max_chunk_size)
        super().__init__(max_chunk_size, chunker)
        self._retriever: bm25s.BM25 | None = None
        self._cache_path = self._processed_path / "bm25" / "chunking.json"
        self._processed_path = self._processed_path / "bm25" / "index"
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.method = method

    def _build_index(self, chunked_files: list[ChunkData]) -> BM25Retriever:
        """
        Build the BM25 index from a list of chunks.

        Args:
            chunked_files (list[ChunkData]): The chunks to index.

        Returns:
            Retriever: A Retriever instance wrapping the BM25 index.
        """
        k1 = self.k1
        b = self.b
        delta = self.delta
        method = self.method

        corpus = [c.model_dump() for c in tqdm(chunked_files, desc="Building corpus", leave=False)]
        if corpus:
            corpus_tokens = bm25s.tokenize(
                [preprocess_text(doc['content']) for doc in corpus],
                show_progress=True,
                leave=False
            )
            retriever = bm25s.BM25(corpus=corpus, k1=k1, method=method, b=b, delta=delta)
            retriever.index(corpus_tokens, show_progress=True, leave_progress=False)

            self._processed_path.mkdir(parents=True, exist_ok=True)
            retriever.save(str(self._processed_path), show_progress=True, leave_progress=False)
        else:
            retriever = bm25s.BM25()

        self._retriever = retriever
        return BM25Retriever(retriever)

    def process_files_from_directory(self, directory: Path) -> BM25Retriever:
        """Process files from a directory incrementally and build a BM25 index.

        Scans the given directory for target files, checks them against a local cache
        to avoid re-chunking unmodified files, and builds or loads the BM25
        search index.

        Args:
            directory: The root directory containing files to process.

        Returns:
            A Retriever instance ready to execute queries.
        """
        target_files = self._get_target_files(directory)
        cache = self._load_cache()
        deltas = self._get_file_deltas(target_files, cache)

        is_modified = len(deltas.modified_or_new_files) > 0 or len(deltas.deleted_files) > 0
        index_exists = (self._processed_path / "data.csc.index.npy").exists()

        if not is_modified and index_exists:
            retriever = bm25s.BM25.load(str(self._processed_path), load_corpus=True)
            self._retriever = retriever
            return BM25Retriever(retriever)

        chunked_files = self._update_cache_and_get_chunks(deltas, cache)

        if is_modified:
            self._save_cache(cache)

        return self._build_index(chunked_files)
