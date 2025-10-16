"""Embedding class for generating embeddings from medical text documents."""

from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer


class Embedding:
    """A class for generating embeddings from medical text documents using Hugging Face models."""

    def __init__(self, model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """Initialize the Embedding class.

        Args:
            model_name: The SentenceTransformer model identifier
            device: The device to use for embedding generation (default: "cuda" if available, otherwise "cpu")
        """
        self.model = SentenceTransformer(model_name)
        self.device = device
        self.model = self.model.to(self.device)

    def embed(self, dataset: Dataset, batch_size: int = 32) -> Dataset:
        """Generate embeddings for all documents in the dataset.

        Args:
            dataset: The Hugging Face Dataset containing documents to embed
            batch_size: Batch size for embedding generation

        Returns:
            Dataset with embeddings added as a new column

        Examples:
            Generate embeddings

            >>> from medretrieval import Corpus, Embedding
            >>> test_data = '''
            ... diabetes.txt: "Diabetes is a chronic condition that affects blood sugar levels."
            ... heart.txt: "Heart disease is a leading cause of death worldwide."
            ... '''
            >>> with yaml_disk(test_data) as temp_dir:
            ...     dataset = Corpus.load_data(temp_dir)
            ...     embedding = Embedding("thomas-sounack/BioClinical-ModernBERT-base")
            ...     dataset_with_embeddings = embedding.embed(dataset, batch_size=32)
            ...     len(dataset_with_embeddings) == 2  # Number of documents
            True
            >>> "embeddings" in dataset_with_embeddings.column_names
            True
            >>> dataset_with_embeddings["embeddings"][0].shape == (768,)  # Embedding dimension
            True
        """
        dataset_with_embeddings = dataset.map(
            self._encode_batch,
            batched=True,
            batch_size=batch_size,
        )
        dataset_with_embeddings.set_format(type="numpy")
        return dataset_with_embeddings

    def _encode_batch(self, examples):
        """Private method to encode a batch of documents.
        
        Args:
            examples: Batch of examples containing 'content' field
            
        Returns:
            Dictionary with 'embeddings' field containing numpy arrays
        """
        embeddings = self.model.encode(
            examples["content"],
            convert_to_tensor=True,
            show_progress_bar=False,
            batch_size=len(examples["content"])
        )
        return {"embeddings": embeddings}
