"""Corpus management module for medical text documents.

This module provides functionality to read, save, and load corpus objects containing medical text documents.
"""

from pathlib import Path

import polars as pl


class Corpus:
    """A corpus object that manages medical text documents.

    The Corpus class automatically detects file types and loads txt and parquet files
    into a unified DataFrame with document_id and content columns.

    Attributes:
        data: Polars DataFrame containing document_id and content columns
    """

    def __init__(self, file_paths: str | Path | list[str] | list[Path], include_subdirs: bool = False):
        """Initialize a Corpus object and load files from path.

        Args:
            file_paths: Single file path, directory path, or list of file paths to load.
            include_subdirs: If True, search subdirectories recursively.
        """
        self.data = self._load_data(file_paths, include_subdirs)

    def _load_data(
        self, paths: str | Path | list[str] | list[Path], include_subdirs: bool = False
    ) -> pl.LazyFrame:
        """Load files from path using lazy loading.

        Args:
            paths: Single file path, directory path, or list of file paths
            include_subdirs: If True, search subdirectories recursively.

        Returns:
            Polars LazyFrame with document_id and content columns (call .collect() to materialize)

        Examples:
            >>> corpus = Corpus.__new__(Corpus)  # Create instance without __init__

            Load single text file
            >>> test_data = '''
            ... diabetes.txt: "Diabetes is a chronic condition..."
            ... '''
            >>> with yaml_disk(test_data) as temp_dir:
            ...     df = corpus._load_data(temp_dir / "diabetes.txt").collect()
            ...     df.columns
            ['document_id', 'content']
            >>> df.height
            1
            >>> df["content"][0].startswith("Diabetes is a chronic condition")
            True

            Unsupported file type
            >>> test_data = '''
            ... invalid_file.py: "print('hello')"
            ... '''
            >>> with yaml_disk(test_data) as temp_dir:
            ...     try:
            ...         corpus._load_data(temp_dir / "invalid_file.py")
            ...     except ValueError as e:
            ...         str(e).startswith("Unsupported file type:")
            True
        """
        file_paths = self._get_file_paths(paths, include_subdirs)

        dfs = []
        for file_path in file_paths:
            try:
                if file_path.suffix.lower() == ".txt":
                    content = file_path.read_text()
                    df = pl.DataFrame({"document_id": [str(file_path)], "content": [content]})
                    dfs.append(df.lazy())
                elif file_path.suffix.lower() == ".parquet":
                    df = pl.scan_parquet(file_path)
                    if "document_id" not in df.columns or "content" not in df.columns:
                        raise ValueError(
                            f"Parquet file {file_path} must contain 'document_id' and 'content' columns"
                        )
                    dfs.append(df)
                else:
                    raise ValueError(f"Unsupported file type: {file_path.suffix}")

            except ValueError as e:
                # Re-raise ValueError as-is (unsupported file type)
                raise e
            except Exception as e:
                raise OSError(f"Error reading file {file_path}: {e!s}") from e
        return pl.concat(dfs)

    def _get_file_paths(
        self, paths: str | Path | list[str] | list[Path], include_subdirs: bool = False
    ) -> set[Path]:
        """Collect all file paths from the given paths.

        Args:
            paths: Single file path, directory path, or list of file paths
            include_subdirs: If True, search subdirectories recursively.

        Returns:
            Set of Path objects for all found files

        Examples:
            >>> corpus = Corpus.__new__(Corpus)  # Create instance without __init__

            Single file path
            >>> test_data = '''
            ... diabetes.txt: "Diabetes is a chronic condition..."
            ... '''
            >>> with yaml_disk(test_data) as temp_dir:
            ...     result = corpus._get_file_paths(temp_dir / "diabetes.txt")
            ...     list(result)[0].name
            'diabetes.txt'

            Directory (non-recursive)
            >>> test_data = '''
            ... data:
            ...   diabetes.txt: "Diabetes info"
            ...   hypertension.txt: "Hypertension info"
            ... '''
            >>> with yaml_disk(test_data) as temp_dir:
            ...     result = corpus._get_file_paths(temp_dir / "data")
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
            ...     result = corpus._get_file_paths(temp_dir / "data", include_subdirs=True)
            ...     sorted([p.name for p in result])
            ['diabetes.txt', 'hypertension.txt']

            Error: nonexistent path
            >>> try:
            ...     corpus._get_file_paths("nonexistent.txt")
            ... except FileNotFoundError as e:
            ...     str(e).startswith("Path not found:")
            True
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
                    file_paths.update(path.glob("**/*.txt"))
                    file_paths.update(path.glob("**/*.parquet"))
                else:
                    file_paths.update(path.glob("*.txt"))
                    file_paths.update(path.glob("*.parquet"))
            elif path.is_file():
                file_paths.add(path)

        return file_paths

    def save(self, output_path: str) -> None:
        """Save the corpus DataFrame to disk in Parquet format.

        Args:
            output_path: Path where to save the corpus (should have .parquet extension)

        Examples:
            Basic save to existing directory
            >>> test_data = '''
            ... diabetes.txt: "Diabetes is a chronic condition..."
            ... '''
            >>> with yaml_disk(test_data) as temp_dir:
            ...     corpus = Corpus(str(temp_dir / "diabetes.txt"))
            ...     output_file = temp_dir / "corpus.parquet"
            ...     corpus.save(output_file)
            ...     output_file.exists()
            True

            Automatically create non-existing directories
            >>> with yaml_disk(test_data) as temp_dir:
            ...     output_dir = temp_dir / "nested" / "subdir"
            ...     output_file = output_dir / "corpus.parquet"
            ...     corpus = Corpus(str(temp_dir / "diabetes.txt"))
            ...     corpus.save(output_file)
            ...     output_file.exists()
            True

            Test invalid save path (should raise OSError or PermissionError)
            >>> with yaml_disk(test_data) as temp_dir:
            ...     corpus = Corpus(str(temp_dir / "diabetes.txt"))
            ...     try:
            ...         corpus.save("/invalid/path/that/does/not/exist/corpus.parquet")
            ...     except (OSError, PermissionError) as e:
            ...         error_type = type(e).__name__
            ...         error_type in ["OSError", "PermissionError"]
            True
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.data.collect().write_parquet(output_path)
