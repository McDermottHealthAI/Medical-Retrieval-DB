"""Integration test for the corpus and embedding modules.

This test demonstrates the complete workflow:
1. Load text files into a corpus
2. Save the corpus to Parquet format
3. Load the corpus back from Parquet
4. Create embeddings for the documents
5. Print preview of embeddings
6. Clean up temporary files
"""

import logging
import os

from hydra import compose, initialize

from medretrieval import Corpus, EmbeddingModel


def test_corpus_integration() -> None:
    """Test the complete corpus and embedding workflow: load -> save -> load -> embed -> verify."""

    # Initialize Hydra configuration
    with initialize(version_base=None, config_path="../src/medretrieval/configs"):
        cfg = compose(config_name="config")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # Step 1: Load the corpus
    logger.info("Step 1: Loading test files into corpus...")
    corpus = Corpus()
    corpus.load_data_files(cfg.corpus.input_files)

    # Step 1.1: Save the corpus file
    temp_parquet = cfg.corpus.output_file
    corpus.save_corpus(temp_parquet)
    logger.info(f"Saved corpus to: {temp_parquet}")

    # Step 1.2: Read the corpus file
    logger.info("Step 3: Loading corpus back from Parquet...")
    loaded_corpus = Corpus()
    loaded_corpus.load_corpus(temp_parquet)

    # Verify it worked
    assert len(corpus) == len(loaded_corpus), "Corpus size mismatch"
    assert set(corpus.documents.keys()) == set(loaded_corpus.documents.keys()), "Document IDs don't match"
    logger.info("✓ Corpus save/load verification successful")

    # Step 2: Embed the corpus
    logger.info("Step 2: Creating embeddings for documents...")
    model_name = cfg.embedding.model_name
    device = cfg.embedding.device if cfg.embedding.device != "auto" else None

    embedding_model = EmbeddingModel(model_name, device=device)
    embeddings_dict = embedding_model.encode(
        loaded_corpus, batch_size=cfg.embedding.batch_size, max_length=cfg.embedding.max_length
    )

    # Step 2.1: Print preview of embeddings
    embedding_dim = embedding_model.model.config.hidden_size
    logger.info(f"  - Created {len(embeddings_dict)} embeddings")
    logger.info(f"  - Embedding dimension: {embedding_dim}")

    for doc_id, embedding in embeddings_dict.items():
        # Get the original text content
        text_content = loaded_corpus.documents[doc_id]
        text_preview = text_content[:100].replace("\n", " ").strip()

        logger.info(f"  - {doc_id}:")
        logger.info(f"    Text preview: {text_preview}...")
        logger.info(f"    Embedding: {len(embedding)} dimensions")
        logger.info(f"    First 5 values: {embedding[:5]}")
        logger.info(f"    Mean: {sum(embedding) / len(embedding):.4f}")

    # Verify it worked
    assert len(embeddings_dict) == len(loaded_corpus.documents), (
        "Number of embeddings doesn't match documents"
    )
    assert all(len(emb) == embedding_dim for emb in embeddings_dict.values()), (
        "Embedding dimensions inconsistent"
    )
    logger.info("✓ Embedding creation verification successful")

    # Step 3: Clean up
    logger.info("Step 6: Cleaning up...")
    os.remove(temp_parquet)
    logger.info(f"Removed: {temp_parquet}")

    logger.info("✓ Integration test completed successfully!")


if __name__ == "__main__":
    test_corpus_integration()
