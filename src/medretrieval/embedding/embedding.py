"""Embedding class for generating embeddings from medical text documents."""

from typing import Any

import torch
from datasets import Dataset
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from sentence_transformers import SentenceTransformer


class Embedding:
    """A class for generating embeddings from medical text documents using Hugging Face models."""

    def __init__(
        self,
        model_name: str,
        tokens_per_chunk: int = 500,
        chunk_overlap: int = 0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize the Embedding class.

        Args:
            model_name: The SentenceTransformer model identifier from Hugging Face
            tokens_per_chunk: Maximum number of tokens per text chunk (default: 500)
            chunk_overlap: Number of overlapping tokens between consecutive chunks (default: 0)
            device: The device to use for embedding generation (default: "cuda" if available, otherwise "cpu")

        Note:
            The text splitter uses the same model's tokenizer to ensure chunks respect token boundaries.
        """
        self.model = SentenceTransformer(model_name)
        self.device = device
        self.model = self.model.to(self.device)
        self.text_splitter = SentenceTransformersTokenTextSplitter(
            model_name=model_name, tokens_per_chunk=tokens_per_chunk, chunk_overlap=chunk_overlap
        )

    def embed(self, dataset: Dataset, batch_size: int = 32, build_faiss_index: bool = True) -> Dataset:
        """Generate embeddings for all documents in the dataset.

        This method first chunks each document into smaller pieces using the configured
        token-based text splitter, then generates embeddings for each chunk. The resulting
        dataset will have one row per chunk, with each chunk preserving its document_id
        and having a sequential chunk_id.

        Args:
            dataset: The Hugging Face Dataset containing documents to embed. Must have
                    "document_id" and "content" columns.
            batch_size: Batch size for embedding generation (default: 32)
            build_faiss_index: Whether to build a FAISS index for efficient similarity search
                             (default: True). Required for using the query() method.

        Returns:
            Dataset with the following columns:
            - document_id: Original document identifier
            - chunk_id: Sequential chunk identifier within each document (0-indexed)
            - content: Text content of each chunk
            - embeddings: Embedding vectors as lists of floats

        Examples:
            Without chunking

            >>> from medretrieval import Corpus
            >>> with get_test_data() as temp_dir:
            ...     dataset = Corpus.load_data(temp_dir)
            >>> embedding = Embedding("thomas-sounack/BioClinical-ModernBERT-base")
            >>> dataset_with_embeddings = embedding.embed(dataset)
            >>> len(dataset_with_embeddings) == 2  # Number of text chunks (1 per file when short)
            True
            >>> "embeddings" in dataset_with_embeddings.column_names
            True
            >>> len(dataset_with_embeddings["embeddings"][0]) == 768  # Embedding dimension
            True

            With chunking

            >>> embedding = Embedding("thomas-sounack/BioClinical-ModernBERT-base", tokens_per_chunk=5)
            >>> dataset_with_embeddings = embedding.embed(dataset)
            >>> len(dataset_with_embeddings) == 5  # 3 chunks for the first file, 2 chunks for the second one
            True
            >>> list(dataset_with_embeddings["chunk_id"])== [0, 1, 2, 0, 1]
            True
        """
        dataset_with_embeddings = dataset.map(
            self._chunk_and_embed_batch, batched=True, batch_size=batch_size
        ).with_format(type="numpy")
        if build_faiss_index:
            dataset_with_embeddings.add_faiss_index("embeddings")
        return dataset_with_embeddings

    def _chunk_and_embed_batch(self, batch: dict[str, list[str]]) -> dict:
        """Chunk and embed a batch of documents.

        This private method processes a batch of documents by:
        1. Splitting each document into chunks using the token-based text splitter
        2. Generating embeddings for all chunks in the batch
        3. Creating metadata (document_id, chunk_id) for each chunk

        Args:
            batch: Dictionary containing "document_id" and "content" lists for a batch of documents

        Returns:
            Dictionary with keys:
            - document_id: List of document IDs (repeated for each chunk from that document)
            - chunk_id: List of sequential chunk IDs (0-indexed per document)
            - content: List of chunk text contents
            - embeddings: List of embedding vectors (as lists of floats)
        """
        chunks, embeddings, doc_ids, chunk_ids = [], [], [], []
        for document_id, text in zip(batch["document_id"], batch["content"], strict=False):
            batch_chunks = self.text_splitter.split_text(text)
            batch_embeddings = self.model.encode(batch_chunks).tolist()

            chunks.extend(batch_chunks)
            embeddings.extend(batch_embeddings)
            doc_ids.extend([document_id] * len(batch_chunks))
            chunk_ids.extend(list(range(len(batch_chunks))))

        return {
            "document_id": doc_ids,
            "chunk_id": chunk_ids,
            "content": chunks,
            "embeddings": embeddings,
        }

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
            >>> from medretrieval import Corpus
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
