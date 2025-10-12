"""Embedding class for generating embeddings from medical text documents."""

import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from medretrieval.corpus.corpus import Corpus


class Embedding:
    """A class for generating embeddings from medical text documents using Hugging Face models."""
    
    def __init__(self, corpus: Corpus, model_name: str):
        """Initialize the Embedding class.
        
        Args:
            corpus: The corpus object containing documents to embed
            model_name: The Hugging Face model identifier
        """
        self.corpus = corpus
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        
    @torch.no_grad()
    def embed(self, pooling: str = "mean", max_length: int = 512) -> torch.Tensor:
        """Generate embeddings for all documents in the corpus.
        
        Args:
            pooling: Pooling strategy - "mean" or "cls"
            max_length: Maximum sequence length for tokenization
            
        Returns:
            Tensor of shape (num_documents, embedding_dim) containing document embeddings
            
        Examples:
            Generate embeddings with mean pooling
            
            >>> from medretrieval import Corpus, Embedding
            >>> test_data = '''
            ... diabetes.txt: "Diabetes is a chronic condition that affects blood sugar levels."
            ... heart.txt: "Heart disease is a leading cause of death worldwide."
            ... '''
            >>> with yaml_disk(test_data) as temp_dir:
            ...     corpus = Corpus(temp_dir)
            ...     embedding = Embedding(corpus, "thomas-sounack/BioClinical-ModernBERT-base")
            ...     embeddings = embedding.embed(pooling="mean")
            ...     embeddings.shape[0] == 2  # Number of documents
            True
            >>> embeddings.shape[1] > 0  # Embedding dimension
            True
            
            Generate embeddings with CLS pooling
            
            >>> with yaml_disk(test_data) as temp_dir:
            ...     corpus = Corpus(temp_dir)
            ...     embedding = Embedding(corpus, "thomas-sounack/BioClinical-ModernBERT-base")
            ...     embeddings_cls = embedding.embed(pooling="cls")
            ...     embeddings_cls.shape[0] == 2  # Number of documents
            True
            
            Invalid pooling strategy
            
            >>> with yaml_disk(test_data) as temp_dir:
            ...     corpus = Corpus(temp_dir)
            ...     embedding = Embedding(corpus, "thomas-sounack/BioClinical-ModernBERT-base")
            ...     try:
            ...         embedding.embed(pooling="invalid")
            ...     except ValueError as e:
            ...         print(f"Caught expected error: {e}")
            Caught expected error: Unknown pooling strategy: invalid
        """
        corpus_data = self.corpus.data.collect()
        documents = corpus_data["content"].to_list()
        
        # Tokenize all documents first (CPU operation) then move to device
        encoder_inputs = self.tokenizer(
            documents, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(self.device)
        
        # Generate embeddings
        last_hidden_state = self.model(**encoder_inputs).last_hidden_state
        
        if pooling == "cls":
            # Use [CLS] token embedding
            embeddings = last_hidden_state[:, 0, :]
        elif pooling == "mean":
            # Mean pooling over all tokens (excluding padding)
            attention_mask = encoder_inputs["attention_mask"]
            masked_embeddings = last_hidden_state * attention_mask.unsqueeze(-1)
            embeddings = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling}")
        
        return embeddings
    
    @classmethod
    def save(cls, embeddings: torch.Tensor, output_path: str | Path) -> None:
        """Save embeddings to disk as NumPy array.
        
        Args:
            embeddings: PyTorch tensor containing the embeddings
            output_path: Path where to save the embeddings (should have .npy extension)
            
        Examples:
            Save embeddings to file
            
            >>> from medretrieval import Corpus, Embedding
            >>> test_data = '''
            ... diabetes.txt: "Diabetes is a chronic condition that affects blood sugar levels."
            ... heart.txt: "Heart disease is a leading cause of death worldwide."
            ... '''
            >>> with yaml_disk(test_data) as temp_dir:
            ...     corpus = Corpus(temp_dir)
            ...     embedding = Embedding(corpus, "thomas-sounack/BioClinical-ModernBERT-base")
            ...     embeddings = embedding.embed()
            ...     output_file = temp_dir / "embeddings.npy"
            ...     Embedding.save(embeddings, output_file)
            ...     output_file.exists()
            True
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to NumPy and save
        embeddings_numpy = embeddings.cpu().numpy()
        np.save(output_path, embeddings_numpy)
