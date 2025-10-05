"""Corpus management module for medical text documents.

This module provides functionality to read, save, and load corpus objects
containing medical text documents.
"""

from pathlib import Path
from typing import Dict, List, Union

import pandas as pd


class Corpus:
    """A corpus object that manages medical text documents.

    The Corpus class provides three main functionalities:
    1. Read text files into corpus objects
    2. Save corpus objects to disk
    3. Load corpus objects from disk

    Attributes:
        documents: Dictionary mapping document IDs to their text content
    """

    def __init__(self):
        """Initialize a Corpus object."""
        self.documents: Dict[str, str] = {}
    
    def load_data_files(self, file_paths: Union[str, List[str]], encoding: str = "utf-8") -> None:
        """Read text files and add them to the corpus.

        Args:
            file_paths: Single file path or list of file paths to read
            encoding: Text encoding to use when reading files

        Examples:
            >>> corpus = Corpus()
            >>> # Read a single file
            >>> corpus.load_data_files("tests/data/diabetes.txt")
            >>> "tests/data/diabetes.txt" in corpus.documents
            True
            >>> # Read multiple files
            >>> corpus.load_data_files(["tests/data/diabetes.txt", "tests/data/hypertension.txt"])
            >>> "tests/data/diabetes.txt" in corpus.documents and "tests/data/hypertension.txt" in corpus.documents
            True
            >>> 
            >>> # Error cases:
            >>> # File not found
            >>> try:
            ...     corpus.load_data_files("nonexistent_file.txt")
            ... except FileNotFoundError as e:
            ...     print(f"Expected error: {e}")
            Expected error: File not found: nonexistent_file.txt
        """
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        for file_path in file_paths:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            try:
                with open(path, 'r', encoding=encoding) as f:
                    content = f.read()

                # Use full path as document ID
                doc_id = str(path)
                self.documents[doc_id] = content

            except Exception as e:
                raise IOError(f"Error reading file {file_path}: {str(e)}")

    def save_corpus(self, output_path: str) -> None:
        """Save the corpus to disk in Parquet format.

        Args:
            output_path: Path where to save the corpus (should have .parquet extension)

        Examples:
            >>> corpus = Corpus()
            >>> corpus.load_data_files("tests/data/diabetes.txt")
            >>> corpus.save_corpus("test_output.parquet")
            >>> import os
            >>> os.path.exists("test_output.parquet")
            True
            >>> # Clean up
            >>> os.remove("test_output.parquet")
            >>> # Save to not existing directory (this should work due to mkdir parents=True)
            >>> corpus.save_corpus("temp_invalid_path/corpus.parquet")
            >>> os.path.exists("temp_invalid_path/corpus.parquet")
            True
            >>> os.remove("temp_invalid_path/corpus.parquet")
            >>> os.rmdir("temp_invalid_path")
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert documents to DataFrame for Parquet format
        data = []
        for doc_id, content in self.documents.items():
            data.append({
                "document_id": doc_id,
                "content": content
            })

        df = pd.DataFrame(data)
        df.to_parquet(output_path, index=False)

    def load_corpus(self, input_path: str) -> None:
        """Load a corpus from disk in Parquet format.

        Args:
            input_path: Path to the saved corpus file (should have .parquet extension)

        Examples:
            >>> corpus = Corpus()
            >>> corpus.load_data_files("tests/data/diabetes.txt")
            >>> corpus.save_corpus("test_load.parquet")
            >>> loaded_corpus = Corpus()
            >>> loaded_corpus.load_corpus("test_load.parquet")
            >>> "tests/data/diabetes.txt" in loaded_corpus.documents
            True
            >>> # Clean up
            >>> import os
            >>> os.remove("test_load.parquet")
            >>> 
            >>> # Error cases:
            >>> # File not found
            >>> try:
            ...     corpus.load_corpus("nonexistent_corpus.parquet")
            ... except FileNotFoundError as e:
            ...     print(f"Expected error: {e}")
            Expected error: Corpus file not found: nonexistent_corpus.parquet
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {input_path}")

        try:
            # Load Parquet file
            df = pd.read_parquet(input_path)

            # Convert DataFrame back to documents dictionary
            self.documents = {}
            for _, row in df.iterrows():
                self.documents[row['document_id']] = row['content']

        except Exception as e:
            raise IOError(f"Error loading corpus from {input_path}: {str(e)}")

    def __len__(self) -> int:
        """Return the number of documents in the corpus."""
        return len(self.documents)

    def __repr__(self) -> str:
        """Return string representation of the corpus."""
        return f"Corpus(documents={len(self.documents)})"
