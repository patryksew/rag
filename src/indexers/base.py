import os
from abc import abstractmethod, ABC
from pathlib import Path
from typing import List, Dict

from tqdm import tqdm
from pydantic import BaseModel, Field

from src.chunker import Chunker
from src.retriever import Retriever
from src.models import ChunkData


class FileCacheData(BaseModel):
    """
    Represents the cached data for a single file.
    """
    mtime: float
    chunks: List[ChunkData]


class ChunkingCache(BaseModel):
    """
    Represents the entire chunking cache.
    """
    files: Dict[str, FileCacheData] = Field(default_factory=dict)


class FileDeltas(BaseModel):
    """
    Represents the file changes detected against the cache.
    """
    unmodified_files: List[str] = Field(default_factory=list)
    modified_or_new_files: List[str] = Field(default_factory=list)
    deleted_files: List[str] = Field(default_factory=list)


class Indexer(ABC):
    """
    Abstract base class for creating and managing a document index.
    """

    def __init__(self, max_chunk_size: int, chunker: Chunker):
        """
        Initialize the Indexer.

        Args:
            max_chunk_size (int): The maximum chunk size in characters.
            chunker (Chunker): The chunker to use for segmenting documents.
        """
        self._max_chunk_size = max_chunk_size
        self._chunker = chunker
        self._project_root = Path(__file__).resolve().parent.parent.parent
        self._processed_path = self._project_root / "data" / "processed" / str(max_chunk_size)
        self._cache_path: Path

    def _get_target_files(self, directory: Path) -> list[str]:
        """
        Retrieve a list of target file paths from the given directory.

        Args:
            directory (Path): The root directory to scan.

        Returns:
            list[str]: A list of relative file paths for processing.
        """
        target_files = []
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix in [".py", ".md", ".txt"]:
                try:
                    relative_path = path.relative_to(self._project_root)
                except ValueError:
                    relative_path = path
                target_files.append(str(relative_path))
        return target_files

    def _get_file_deltas(self, target_files: list[str], cache: ChunkingCache) -> FileDeltas:
        """
        Compare the target files with the cache to find modified, unmodified, and deleted files.

        Args:
            target_files (list[str]): The target file paths.
            cache (ChunkingCache): The current chunking cache.

        Returns:
            FileDeltas: The object describing file changes.
        """
        unmodified_files = []
        modified_or_new_files = []
        cache_keys_present = set()

        for file_path in target_files:
            try:
                mtime = os.path.getmtime(file_path)
            except OSError:
                continue

            if file_path in cache.files and cache.files[file_path].mtime == mtime:
                unmodified_files.append(file_path)
            else:
                modified_or_new_files.append(file_path)
            cache_keys_present.add(file_path)

        deleted_files = [f for f in cache.files.keys() if f not in cache_keys_present]

        return FileDeltas(
            unmodified_files=unmodified_files,
            modified_or_new_files=modified_or_new_files,
            deleted_files=deleted_files
        )

    def _load_cache(self) -> ChunkingCache:
        """
        Load the chunking cache from disk, if it exists.

        Returns:
            ChunkingCache: The loaded chunking cache, or a new empty one if not found.
        """
        if self._cache_path.exists():
            try:
                with open(self._cache_path, "r", encoding="utf-8") as f:
                    return ChunkingCache.model_validate_json(f.read())
            except Exception as e:
                import sys
                print(
                    f"Warning: Failed to load chunking cache, using an empty cache. Error: {e}",
                    file=sys.stderr
                )
        return ChunkingCache()

    def _update_cache_and_get_chunks(self,
                                     deltas: FileDeltas,
                                     cache: ChunkingCache) -> list[ChunkData]:
        """
        Update the cache with new chunks and return all valid chunks.

        Args:
            deltas (FileDeltas): The file changes detected.
            cache (ChunkingCache): The chunking cache to update.

        Returns:
            list[ChunkData]: All chunks for the current state of files.
        """
        chunked_files: list[ChunkData] = []

        for file_path in deltas.unmodified_files:
            chunked_files.extend(cache.files[file_path].chunks)

        for file_path in tqdm(deltas.modified_or_new_files,
                              desc="Chunking files",
                              disable=len(deltas.modified_or_new_files) == 0,
                              leave=False):
            chunks = self._chunker.chunk_file(file_path)
            chunked_files.extend(chunks)
            cache.files[file_path] = FileCacheData(
                mtime=os.path.getmtime(file_path),
                chunks=chunks
            )

        for file_path in deltas.deleted_files:
            if file_path in cache.files:
                del cache.files[file_path]

        return chunked_files

    def _save_cache(self, cache: ChunkingCache) -> None:
        """
        Save the chunking cache to disk.

        Args:
            cache (ChunkingCache): The chunking cache to save.
        """
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cache_path, "w", encoding="utf-8") as f:
                f.write(cache.model_dump_json())
        except Exception as e:
            import sys
            print(
                f"Warning: Failed to save chunking cache to {self._cache_path}. Error: {e}",
                file=sys.stderr
            )

    @abstractmethod
    def process_files_from_directory(self, directory: Path) -> Retriever:
        """Process files from a directory and build or load the retriever index.

        Args:
            directory: The root directory containing files to process.

        Returns:
            A Retriever instance ready to execute queries.
        """
        ...

    def has_data_changes(self, directory: Path) -> bool:
        """
        Check if there are data changes in the data directory since the last cache.

        Args:
            directory: The root directory containing files to check.

        Returns:
            True if there are data changes, False otherwise.
        """

        deltas = self._get_file_deltas(self._get_target_files(directory), self._load_cache())

        return len(deltas.deleted_files) > 0 or len(deltas.modified_or_new_files) > 0
