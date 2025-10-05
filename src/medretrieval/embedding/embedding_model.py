"""Embedding model for medical text processing using Hugging Face transformers."""

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

if TYPE_CHECKING:
    from ..corpus.corpus import Corpus

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """A class for loading and using embedding models from Hugging Face.

    This class provides functionality to load pre-trained embedding models
    from Hugging Face Hub and prepare them for inference on medical text.

    Attributes:
        model_name (str): The name of the Hugging Face model.
        model: The loaded transformer model.
        tokenizer: The tokenizer for the model.
        device (str): The device the model is loaded on.

    Examples:
        Basic usage with corpus:
        >>> from medretrieval import Corpus, EmbeddingModel
        >>> corpus = Corpus()
        >>> corpus.load_data_files("tests/data/diabetes.txt")
        >>> embedding_model = EmbeddingModel("thomas-sounack/BioClinical-ModernBERT-base")
        >>> embeddings = embedding_model.encode(corpus)
        >>> len(embeddings)
        1
        >>> "tests/data/diabetes.txt" in embeddings
        True
        >>> len(embeddings["tests/data/diabetes.txt"])
        768

        Using with different device:
        >>> embedding_model = EmbeddingModel("thomas-sounack/BioClinical-ModernBERT-base", device="cpu")
        >>> embeddings = embedding_model.encode(corpus)
        >>> len(embeddings)
        1

        Error cases:
        >>> # Invalid model name
        >>> try:
        ...     EmbeddingModel("invalid-model-name")
        ... except Exception as e:
        ...     print(f"Error: {type(e).__name__}")
        Error: OSError

        >>> # Empty corpus
        >>> empty_corpus = Corpus()
        >>> try:
        ...     embedding_model.encode(empty_corpus)
        ... except ValueError as e:
        ...     print(f"Error: {e}")
        Error: Cannot encode empty corpus
    """

    def __init__(self, model_name: str, device: str | None = None):
        """Initialize the embedding model.

        Args:
            model_name: The name of the Hugging Face model to load.
            device: The device to load the model on. If None, uses 'cuda' if available, otherwise 'cpu'.

        Raises:
            OSError: If the model cannot be loaded from Hugging Face Hub.
            ValueError: If the model_name is empty or invalid.
        """
        if not model_name or not isinstance(model_name, str):
            raise ValueError("model_name must be a non-empty string")

        self.model_name = model_name

        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load the model
        self._load_model()

    def _load_model(self) -> None:
        """Load the embedding model from Hugging Face."""
        logger.info(f"Loading model '{self.model_name}' on device '{self.device}'")

        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)

            # Move model to device
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode

            logger.info(f"Successfully loaded model '{self.model_name}'")

        except Exception as e:
            raise OSError(f"Failed to load model '{self.model_name}': {e!s}") from e

    def encode(self, corpus: "Corpus", batch_size: int = 32, max_length: int = 512) -> dict[str, np.ndarray]:
        """Encode all documents in a corpus into embeddings.

        Args:
            corpus: A Corpus object containing documents to encode.
            batch_size: The batch size for processing multiple texts.
            max_length: Maximum sequence length for tokenization.

        Returns:
            A dictionary mapping document IDs to their embeddings.

        Raises:
            ValueError: If corpus is empty.
            OSError: If encoding fails.

        Examples:
            >>> from medretrieval import Corpus, EmbeddingModel
            >>> corpus = Corpus()
            >>> corpus.load_data_files("tests/data/diabetes.txt")
            >>> embedding_model = EmbeddingModel("thomas-sounack/BioClinical-ModernBERT-base")
            >>> embeddings = embedding_model.encode(corpus)
            >>> len(embeddings)
            1
            >>> "tests/data/diabetes.txt" in embeddings
            True
            >>> len(embeddings["tests/data/diabetes.txt"])
            768

            With custom parameters:
            >>> embeddings = embedding_model.encode(corpus, batch_size=16, max_length=256)
            >>> len(embeddings)
            1

            Error cases:
            >>> # Empty corpus
            >>> empty_corpus = Corpus()
            >>> try:
            ...     embedding_model.encode(empty_corpus)
            ... except ValueError as e:
            ...     print(f"Error: {e}")
            Error: Cannot encode empty corpus
        """
        if not corpus.documents:
            raise ValueError("Cannot encode empty corpus")

        logger.info(f"Encoding corpus with {len(corpus)} documents")

        # Extract texts and document IDs
        document_texts = list(corpus.documents.values())
        document_ids = list(corpus.documents.keys())

        # Encode texts using private method
        embeddings = self._encode_texts(document_texts, batch_size, max_length)

        # Create dictionary mapping document IDs to embeddings
        embeddings_dict = dict(zip(document_ids, embeddings, strict=False))

        logger.info(f"Successfully encoded {len(embeddings_dict)} documents")
        return embeddings_dict

    def _encode_texts(
        self, texts: str | list[str], batch_size: int = 32, max_length: int = 512
    ) -> list[np.ndarray]:
        """Encode text(s) into embeddings (private method).

        Args:
            texts: A single text string or list of text strings to encode.
            batch_size: The batch size for processing multiple texts.
            max_length: Maximum sequence length for tokenization.

        Returns:
            A list of embeddings, where each embedding is a list of floats.
            If a single text is provided, returns a list with one embedding.

        Raises:
            ValueError: If texts is empty or contains non-string elements.
            OSError: If encoding fails.

        Examples:
            Single text:
            >>> embedding_model = EmbeddingModel("thomas-sounack/BioClinical-ModernBERT-base")
            >>> embeddings = embedding_model._encode_texts("Patient has diabetes")
            >>> len(embeddings)
            1
            >>> len(embeddings[0])
            768

            Multiple texts:
            >>> texts = ["Patient has diabetes", "Blood pressure is high"]
            >>> embeddings = embedding_model._encode_texts(texts)
            >>> len(embeddings)
            2
            >>> all(len(emb) == 768 for emb in embeddings)
            True

            With custom parameters:
            >>> embeddings = embedding_model._encode_texts(texts, batch_size=16, max_length=256)
            >>> len(embeddings)
            2

            Error cases:
            >>> # Empty input
            >>> embeddings = embedding_model._encode_texts([])
            >>> len(embeddings)
            0

            >>> # Non-string input
            >>> try:
            ...     embedding_model._encode_texts([123, "text"])
            ... except ValueError as e:
            ...     print(f"Error: {e}")
            Error: All texts must be strings
        """
        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return []

        # Validate input
        if not all(isinstance(text, str) for text in texts):
            raise ValueError("All texts must be strings")

        logger.info(f"Encoding {len(texts)} text(s) with batch size {batch_size}")

        try:
            embeddings = []

            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]

                # Tokenize
                inputs = self.tokenizer(
                    batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=max_length
                )

                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Get embeddings
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Use mean pooling of last hidden states
                    batch_embeddings = outputs.last_hidden_state.mean(dim=1)

                # Convert to CPU and to numpy array
                batch_embeddings = batch_embeddings.cpu().numpy()
                embeddings.extend(batch_embeddings)

            logger.info(f"Successfully encoded {len(embeddings)} text(s)")
            return embeddings

        except Exception as e:
            raise OSError(f"Failed to encode texts: {e!s}") from e

    def save_embeddings(self, embeddings: dict[str, np.ndarray], output_path: str) -> None:
        """Save embeddings to a Parquet file.

        Args:
            embeddings: Dictionary mapping document IDs to their embeddings.
            output_path: Path where to save the Parquet file.

        Raises:
            ValueError: If embeddings is empty.
            OSError: If saving fails.

        Examples:
            >>> from medretrieval import Corpus, EmbeddingModel
            >>> corpus = Corpus()
            >>> corpus.load_data_files("tests/data/diabetes.txt")
            >>> embedding_model = EmbeddingModel("thomas-sounack/BioClinical-ModernBERT-base")
            >>> embeddings = embedding_model.encode(corpus)
            >>> embedding_model.save_embeddings(embeddings, "embeddings.parquet")
            >>> import os
            >>> os.path.exists("embeddings.parquet")
            True
            >>> os.remove("embeddings.parquet")  # Clean up

            Error cases:
            >>> # Empty embeddings
            >>> try:
            ...     embedding_model.save_embeddings({}, "test.parquet")
            ... except ValueError as e:
            ...     print(f"Error: {e}")
            Error: Cannot save empty embeddings
        """
        if not embeddings:
            raise ValueError("Cannot save empty embeddings")

        logger.info(f"Saving {len(embeddings)} embeddings to {output_path}")

        try:
            import pandas as pd

            # Extract document IDs and embeddings from dictionary
            document_ids = list(embeddings.keys())
            embeddings_list = list(embeddings.values())

            # Create DataFrame with document_id and embedding array column
            data = {"document_id": document_ids, "embedding": embeddings_list}

            df = pd.DataFrame(data)
            df.to_parquet(output_path, index=False)

            logger.info(f"Successfully saved embeddings to {output_path}")

        except Exception as e:
            raise OSError(f"Failed to save embeddings to {output_path}: {e!s}") from e

    def load_embeddings(self, input_path: str) -> dict[str, np.ndarray]:
        """Load embeddings from a Parquet file.

        Args:
            input_path: Path to the Parquet file containing embeddings.

        Returns:
            A dictionary mapping document IDs to their embeddings.

        Raises:
            OSError: If loading fails or file doesn't exist.

        Examples:
            >>> embedding_model = EmbeddingModel("thomas-sounack/BioClinical-ModernBERT-base")
            >>> texts = ["Patient has diabetes"]
            >>> embeddings = embedding_model._encode_texts(texts)
            >>> embeddings_dict = {"doc1": embeddings[0]}
            >>> embedding_model.save_embeddings(embeddings_dict, "test_embeddings.parquet")
            >>> loaded_embeddings = embedding_model.load_embeddings("test_embeddings.parquet")
            >>> "doc1" in loaded_embeddings
            True
            >>> len(loaded_embeddings["doc1"])
            768
            >>> import os
            >>> os.remove("test_embeddings.parquet")  # Clean up

            Error cases:
            >>> # File not found
            >>> try:
            ...     embedding_model.load_embeddings("nonexistent.parquet")
            ... except OSError as e:
            ...     print(f"Error: {type(e).__name__}")
            Error: OSError
        """
        logger.info(f"Loading embeddings from {input_path}")

        try:
            # Load the Parquet file
            df = pd.read_parquet(input_path)

            # Extract document IDs and embeddings
            embeddings_dict = {}
            for _, row in df.iterrows():
                doc_id = row["document_id"]
                embedding = row["embedding"]
                embeddings_dict[doc_id] = embedding

            logger.info(f"Successfully loaded {len(embeddings_dict)} embeddings from {input_path}")
            return embeddings_dict

        except Exception as e:
            raise OSError(f"Failed to load embeddings from {input_path}: {e!s}") from e

    def __repr__(self) -> str:
        """Return a string representation of the EmbeddingModel."""
        return f"EmbeddingModel(model_name='{self.model_name}', device='{self.device}')"
