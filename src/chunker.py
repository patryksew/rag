import os

from tqdm import tqdm

from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from src.models import ChunkData


class Chunker:
    """
        Chunks Python and Markdown files using language-specific splitting strategies.
    """

    def __init__(self, max_chunk_size: int = 2000):
        """
        Initialize the Chunker.

        Args:
            max_chunk_size (int): The maximum chunk size in characters (default: 2000).
        """
        self._max_chunk_size = min(max_chunk_size, 2000)
        self._python_splitter = RecursiveCharacterTextSplitter.from_language(
            Language.PYTHON,
            chunk_size=self._max_chunk_size,
            chunk_overlap=self._max_chunk_size // 10,
            add_start_index=True
        )
        self._markdown_splitter = RecursiveCharacterTextSplitter.from_language(
            Language.MARKDOWN,
            chunk_size=self._max_chunk_size,
            chunk_overlap=self._max_chunk_size // 20,
            add_start_index=True
        )

    def chunk_file(self, file_path: str) -> list[ChunkData]:
        """
        Chunk a single file into smaller segments.

        Args:
            file_path (str): The path to the file to chunk.

        Returns:
            list[ChunkData]: A list of chunked data objects.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except FileNotFoundError:
            import sys
            print(f"Warning: File not found during chunking: {file_path}", file=sys.stderr)
            return []
        except UnicodeDecodeError:
            import sys
            print(f"Warning: Unicode decoding error in file: {file_path}", file=sys.stderr)
            return []

        _, ext = os.path.splitext(file_path)

        if ext == ".py":
            splitter = self._python_splitter
        elif ext == ".md":
            splitter = self._markdown_splitter
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        chunk_data: list[ChunkData] = []
        docs = splitter.create_documents([content])

        for doc in docs:
            original_content = doc.page_content
            file_name = os.path.basename(file_path)
            augmented_content = f"File: {file_path}\nFile: {file_name}\n\n{original_content}"

            chunk_data.append(
                ChunkData(
                    file_path=file_path,
                    first_character_index=doc.metadata.get("start_index", -1),
                    last_character_index=doc.metadata.get("start_index", -1) + len(
                        original_content),
                    content=augmented_content
                )
            )

        return chunk_data

    def chunk_files(self, file_paths: list[str]) -> list[ChunkData]:
        """
        Chunk a list of files into smaller segments.

        Args:
            file_paths (list[str]): The paths to the files to chunk.

        Returns:
            list[ChunkData]: A combined list of all chunked data objects.
        """
        result: list[ChunkData] = []

        for file_path in tqdm(file_paths,
                              desc="Chunking files",
                              disable=len(file_paths) == 0,
                              leave=False):
            result.extend(self.chunk_file(file_path))

        return result
