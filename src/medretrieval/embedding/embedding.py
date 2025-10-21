"""Embedding class for generating embeddings from medical text documents."""

from typing import Any

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

    def embed(self, dataset: Dataset, batch_size: int = 32, build_faiss_index: bool = True) -> Dataset:
        """Generate embeddings for all documents in the dataset.

        Args:
            dataset: The Hugging Face Dataset containing documents to embed
            batch_size: Batch size for embedding generation
            build_faiss_index: Whether to build a FAISS index for the embeddings

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
            lambda x: {"embeddings": self.model.encode(x["content"])},
            batched=True,
            batch_size=batch_size,
        )
        dataset_with_embeddings.set_format(type="numpy")
        if build_faiss_index:
            dataset_with_embeddings.add_faiss_index("embeddings")
        return dataset_with_embeddings

    def query(
        self, dataset: Dataset, queries: list[str], k: int = 1
    ) -> tuple[list[list[float]], list[dict[str, Any]]]:
        """Query the dataset for similar documents using efficient batch processing.

        This method uses FAISS batch search for optimal performance when querying
        multiple documents simultaneously.

        Args:
            dataset: Dataset with FAISS index built on embeddings
            queries: List of query strings to search for
            k: Number of nearest neighbors to retrieve per query

        Returns:
            Tuple containing:
            - scores: List of lists, where each inner list contains similarity scores
                    for the k nearest neighbors of the corresponding query
            - retrieved_examples: List of dictionaries, where each dict contains
                                the retrieved examples for the corresponding query

        Examples:
            >>> from medretrieval import Corpus, Embedding
            >>> test_data = '''
            ... diabetes.txt: "Diabetes is a chronic condition affecting blood sugar levels."
            ... heart.txt: "Heart disease is a leading cause of death worldwide."
            ... '''
            >>> with yaml_disk(test_data) as temp_dir:
            ...     corpus_dataset = Corpus.load_data(temp_dir)
            ...     embedding = Embedding("thomas-sounack/BioClinical-ModernBERT-base")
            ...     dataset = embedding.embed(corpus_dataset)
            ...     scores, examples = embedding.query(dataset, ["Diabetes", "Heart disease"], k=1)
            ...     len(examples[0]["content"]) == len(examples[1]["content"]) == 1  # 1 neighbor per query
            True
            >>> examples[0]["content"][0] == "Diabetes is a chronic condition affecting blood sugar levels."
            True
            >>> examples[1]["content"][0] == "Heart disease is a leading cause of death worldwide."
            True
        """
        encoded_queries = self.model.encode(queries)
        scores, examples = dataset.get_nearest_examples_batch("embeddings", encoded_queries, k)
        return scores, examples
