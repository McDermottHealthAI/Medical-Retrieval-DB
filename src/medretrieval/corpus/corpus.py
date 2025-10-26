"""Corpus management module for medical text documents.

This module provides functionality to read, save, and load corpus objects containing medical text documents.
"""

from pathlib import Path

from datasets import Dataset, concatenate_datasets
from langchain_text_splitters import RecursiveCharacterTextSplitter


class Corpus:
    """A utility class for managing medical text documents.

    The Corpus class provides static methods to load txt and parquet files into Hugging Face Datasets with
    document_id, chunk_id, and content columns. Supports automatic text chunking with configurable size and
    overlap, and streaming for large parquet files to enable memory-efficient processing.
    """

    @staticmethod
    def load_data(
        paths: str | Path | list[str] | list[Path],
        include_subdirs: bool = False,
        streaming: bool = False,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ) -> Dataset:
        """Load files from path and return as Hugging Face Dataset.

        Supports loading:
        - Individual .txt files
        - Individual .parquet files (with streaming)
        - Directories containing .txt/.parquet files

        Args:
            paths: Single file path, directory path, or list of file paths
            include_subdirs: If True, search subdirectories recursively.
            streaming: If True, load parquet files lazily for memory efficiency
            chunk_size: Maximum size of text chunks (default: 500)
            chunk_overlap: Overlap between consecutive chunks (default: 100)

        Returns:
            Hugging Face Dataset with document_id, chunk_id, and content columns.
            When streaming=True, parquet files are loaded lazily for memory efficiency.

        Examples:
            Load single text file

            >>> test_data = '''
            ... diabetes.txt: "Diabetes is a chronic condition..."
            ... '''
            >>> with yaml_disk(test_data) as temp_dir:
            ...     dataset = Corpus.load_data(temp_dir / "diabetes.txt", chunk_size=25, chunk_overlap=0)
            ...     dataset.column_names
            ['document_id', 'chunk_id', 'content']
            >>> len(dataset)
            2
            >>> dataset["content"][0].startswith("Diabetes is a chronic")
            True
            >>> dataset["content"][1].startswith("condition...")
            True

            Load Hugging Face dataset directory

            >>> output_path = temp_dir / "corpus_dataset.parquet"
            >>> Corpus.save(dataset, output_path)
            >>> loaded_dataset = Corpus.load_data(output_path)
            >>> len(loaded_dataset)
            2
            >>> loaded_dataset["content"][0].startswith("Diabetes is a chronic")
            True

            Load parquet file with streaming (lazy loading)

            >>> streamed_dataset = Corpus.load_data(output_path, streaming=True)
            >>> # Streaming datasets are iterable, list() to get the first element
            >>> list(streamed_dataset)[0]["content"].startswith("Diabetes is a chronic")
            True

            Unsupported file type

            >>> test_data = '''
            ... invalid_file.py: "print('hello')"
            ... '''
            >>> with yaml_disk(test_data) as temp_dir:
            ...     Corpus.load_data(temp_dir / "invalid_file.py")
            Traceback (most recent call last):
                ...
            ValueError: Unsupported file type: .py
        """
        file_paths = Corpus._get_file_paths(paths, include_subdirs)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        dsets = []
        documents = []
        for file_path in file_paths:
            try:
                if file_path.suffix.lower() == ".txt":
                    content = file_path.read_text()
                    text_chunks = text_splitter.split_text(content)
                    documents.extend(
                        [
                            {"document_id": str(file_path), "chunk_id": i, "content": chunk}
                            for i, chunk in enumerate(text_chunks)
                        ]
                    )
                elif file_path.suffix.lower() == ".parquet":
                    dataset = Dataset.from_parquet(str(file_path), streaming=streaming)
                    dsets.append(dataset)
                else:
                    raise ValueError(f"Unsupported file type: {file_path.suffix}")
            except ValueError as e:
                # Re-raise ValueError as-is (unsupported file type)
                raise e
            except Exception as e:
                raise OSError(f"Error reading file {file_path}: {e!s}") from e

        if documents:
            dsets.append(Dataset.from_list(documents))

        return concatenate_datasets(dsets)

    @staticmethod
    def _get_file_paths(
        paths: str | Path | list[str] | list[Path], include_subdirs: bool = False
    ) -> set[Path]:
        """Collect all file paths from the given paths.

        Args:
            paths: Single file path, directory path, or list of file paths
            include_subdirs: If True, search subdirectories recursively.

        Returns:
            Set of Path objects for all found files

        Examples:
            Directory (non-recursive)

            >>> test_data = '''
            ... data:
            ...   diabetes.txt: "Diabetes info"
            ...   hypertension.txt: "Hypertension info"
            ... '''
            >>> with yaml_disk(test_data) as temp_dir:
            ...     result = Corpus._get_file_paths(temp_dir / "data")
            ...     sorted([p.name for p in result])
            ['diabetes.txt', 'hypertension.txt']

            Recursive directory

            >>> test_data = '''
            ... data:
            ...   diabetes.txt: "Diabetes info"
            ...   subdir:
            ...     hypertension.txt: "Hypertension info"
            ... '''
            >>> with yaml_disk(test_data) as temp_dir:
            ...     result = Corpus._get_file_paths(temp_dir / "data", include_subdirs=True)
            ...     sorted([p.name for p in result])
            ['diabetes.txt', 'hypertension.txt']

            Error: nonexistent path

            >>> Corpus._get_file_paths("nonexistent.txt")
            Traceback (most recent call last):
                ...
            FileNotFoundError: Path not found: nonexistent.txt
        """
        if not isinstance(paths, list):
            paths = [paths]

        # Use set to avoid reading the same file multiple times.
        file_paths = set()
        for path in paths:
            path = Path(path)

            if not path.exists():
                raise FileNotFoundError(f"Path not found: {path}")

            if path.is_dir():
                if include_subdirs:
                    file_paths.update(path.rglob("*.txt"))
                    file_paths.update(path.rglob("*.parquet"))
                else:
                    file_paths.update(path.glob("*.txt"))
                    file_paths.update(path.glob("*.parquet"))
            elif path.is_file():
                file_paths.add(path)

        return file_paths

    @staticmethod
    def save(dataset: Dataset, output_path: str | Path) -> None:
        """Save a Dataset to disk in Parquet format.

        Args:
            dataset: Hugging Face Dataset to save
            output_path: Path where to save the dataset (should have .parquet extension)
        """
        output_path = Path(output_path)
        dataset.to_parquet(output_path)
