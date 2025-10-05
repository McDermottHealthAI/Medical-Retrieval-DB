"""Medical Retrieval Database package."""

from .corpus.corpus import Corpus
from .embedding.embedding_model import EmbeddingModel

__all__ = ["Corpus", "EmbeddingModel"]
