"""Corpus management module for medical text documents.

This module provides functionality to read, save, and load corpus objects containing medical text documents.
"""

from pathlib import Path

from datasets import Dataset, concatenate_datasets


class Corpus:
    """A utility class for managing medical text documents.

    The Corpus class provides static methods to load txt and parquet files into Hugging Face Datasets with
    document_id and content columns. Supports streaming for large parquet files to enable memory-efficient
    processing.
    """

    @staticmethod
    def load_data(
        paths: str | Path | list[str] | list[Path], include_subdirs: bool = False, streaming: bool = False
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

        Returns:
            Hugging Face Dataset with document_id and content columns.
            When streaming=True, parquet files are loaded lazily for memory efficiency.

        Examples:
            Load single text file

            >>> test_data = '''
            ... diabetes.txt: "Diabetes is a chronic condition..."
            ... '''
            >>> with yaml_disk(test_data) as temp_dir:
            ...     dataset = Corpus.load_data(temp_dir / "diabetes.txt")
            ...     dataset.column_names
            ['document_id', 'content']
            >>> len(dataset)
            1
            >>> dataset["content"][0].startswith("Diabetes is a chronic condition")
            True

            Load Hugging Face dataset directory

            >>> test_data = '''
            ... diabetes.txt: "Diabetes is a chronic condition..."
            ... '''
            >>> with yaml_disk(test_data) as temp_dir:
            ...     dataset = Corpus.load_data(temp_dir / "diabetes.txt")
            ...     output_path = temp_dir / "corpus_dataset.parquet"
            ...     Corpus.save(dataset, output_path)
            ...     loaded_dataset = Corpus.load_data(output_path)
            ...     len(loaded_dataset)
            1
            >>> loaded_dataset["content"][0].startswith("Diabetes is a chronic condition")
            True

            Load parquet file with streaming (lazy loading)

            >>> test_data = '''
            ... diabetes.txt: "Diabetes is a chronic condition..."
            ... '''
            >>> with yaml_disk(test_data) as temp_dir:
            ...     dataset = Corpus.load_data(temp_dir / "diabetes.txt")
            ...     output_path = temp_dir / "corpus_dataset.parquet"
            ...     Corpus.save(dataset, output_path)
            ...     # Load with streaming for memory efficiency
            ...     streamed_dataset = Corpus.load_data(output_path, streaming=True)
            ...     # Streaming datasets are iterable, list() to get the first element
            ...     list(streamed_dataset)[0]["content"].startswith("Diabetes is a chronic condition")
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

        dsets = []
        documents = []
        for file_path in file_paths:
            try:
                if file_path.suffix.lower() == ".txt":
                    content = file_path.read_text()
                    documents.append({"document_id": str(file_path), "content": content})
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
            Single file path

            >>> test_data = '''
            ... diabetes.txt: "Diabetes is a chronic condition..."
            ... '''
            >>> with yaml_disk(test_data) as temp_dir:
            ...     result = Corpus._get_file_paths(temp_dir / "diabetes.txt")
            ...     list(result)[0].name
            'diabetes.txt'

            Directory with txt files

            >>> test_data = '''
            ... data:
            ...   diabetes.txt: "Diabetes info"
            ...   subdir:
            ...     hypertension.txt: "Hypertension info"
            ... '''
            >>> with yaml_disk(test_data) as temp_dir:
            ...     # Non-recursive
            ...     result = Corpus._get_file_paths(temp_dir / "data")
            ...     sorted([p.name for p in result])
            ['diabetes.txt']
            >>> with yaml_disk(test_data) as temp_dir:
            ...     # Recursive
            ...     result = Corpus._get_file_paths(temp_dir / "data", include_subdirs=True)
            ...     sorted([p.name for p in result])
            ['diabetes.txt', 'hypertension.txt']

            Mixed sources (txt files + HF datasets)

            >>> test_data = '''
            ... data:
            ...   file1.txt: "Content 1"
            ...   dataset1:
            ...     dataset_info.json: '{"features": {"document_id": "string", "content": "string"}}'
            ...     state.json: '{"_data_files": []}'
            ...   subdir:
            ...     file2.txt: "Content 2"
            ...     dataset2:
            ...       dataset_info.json: '{"features": {"document_id": "string", "content": "string"}}'
            ...       state.json: '{"_data_files": []}'
            ... '''
            >>> with yaml_disk(test_data) as temp_dir:
            ...     result = Corpus._get_file_paths(temp_dir / "data", include_subdirs=True)
            ...     sorted([p.name for p in result])
            ['dataset1', 'dataset2', 'file1.txt', 'file2.txt']

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
                    # HF datasets are stored in directories with a dataset_info.json file
                    dataset_dirs = {p.parent for p in path.rglob("dataset_info.json")}
                    file_paths.update(dataset_dirs)
                else:
                    file_paths.update(path.glob("*.txt"))
                    if (path / "dataset_info.json").exists():
                        file_paths.add(path)
            elif path.is_file():
                file_paths.add(path)

        return file_paths

    @staticmethod
    def save(dataset: Dataset, output_path: str | Path) -> None:
        """Save a Dataset to disk in Parquet format.

        Args:
            dataset: Hugging Face Dataset to save
            output_path: Path where to save the dataset (should have .parquet extension)

        Examples:
            Basic save to existing directory

            >>> test_data = '''
            ... diabetes.txt: "Diabetes is a chronic condition..."
            ... '''
            >>> with yaml_disk(test_data) as temp_dir:
            ...     dataset = Corpus.load_data(temp_dir / "diabetes.txt")
            ...     output_file = temp_dir / "corpus.parquet"
            ...     Corpus.save(dataset, output_file)
            ...     output_file.exists()
            True

            Automatically create non-existing directories

            >>> with yaml_disk(test_data) as temp_dir:
            ...     output_file = temp_dir / "nested" / "subdir" / "corpus.parquet"
            ...     dataset = Corpus.load_data(temp_dir / "diabetes.txt")
            ...     Corpus.save(dataset, output_file)
            ...     output_file.exists()
            True
        """
        output_path = Path(output_path)
        dataset.to_parquet(output_path)
