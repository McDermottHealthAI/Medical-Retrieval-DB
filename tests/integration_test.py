"""Integration test for the corpus module.

This test demonstrates the complete workflow:
1. Load text files into a corpus
2. Save the corpus to Parquet format
3. Load the corpus back from Parquet
4. Verify the data integrity
5. Clean up temporary files
"""

import logging
import os

import hydra
from hydra import initialize, compose
from omegaconf import DictConfig

from medretrieval import Corpus

def test_corpus_integration() -> None:
    """Test the complete corpus workflow: load -> save -> load -> verify."""
    
    # Initialize Hydra configuration
    with initialize(version_base=None, config_path="../src/medretrieval/configs"):
        cfg = compose(config_name="config")
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Step 1: Create a corpus and load the test files
    logger.info("Step 1: Loading test files into corpus...")
    corpus = Corpus()
    corpus.load_data_files(cfg.corpus.input_files)
    
    logger.info(f"Loaded {len(corpus)} documents:")
    for doc_id in corpus.documents.keys():
        logger.info(f"  - {doc_id}")
    
    # Step 2: Save corpus to Parquet
    logger.info("Step 2: Saving corpus to Parquet...")
    temp_parquet = cfg.corpus.output_file
    corpus.save_corpus(temp_parquet)
    
    # Verify file was created
    assert os.path.exists(temp_parquet), "Parquet file was not created"
    logger.info(f"Saved corpus to: {temp_parquet}")
    
    # Step 3: Load corpus back from Parquet
    logger.info("Step 3: Loading corpus back from Parquet...")
    loaded_corpus = Corpus()
    loaded_corpus.load_corpus(temp_parquet)
    
    logger.info(f"Loaded {len(loaded_corpus)} documents:")
    for doc_id in loaded_corpus.documents.keys():
        logger.info(f"  - {doc_id}")
    
    # Show some content from each loaded document
    logger.info("Content preview from loaded documents:")
    for doc_id, content in loaded_corpus.documents.items():
        logger.info(f"{doc_id}:")
        logger.info(f"  Length: {len(content)} characters")
        preview = content[:200].replace(chr(10), ' ')
        logger.info(f"  Preview: {preview}...")
    
    # Step 4: Verify data integrity
    logger.info("Step 4: Verifying data integrity...")
    assert set(corpus.documents.keys()) == set(loaded_corpus.documents.keys()), "Document IDs don't match"
    
    # Check that all content matches
    for doc_id in corpus.documents.keys():
        original_content = corpus.documents[doc_id]
        loaded_content = loaded_corpus.documents[doc_id]
        preview = loaded_content[:100].replace('\n', ' ')
        logger.info(f"  {doc_id}: {preview}...")
        assert original_content == loaded_content, f"Content mismatch for {doc_id}"
    
    # Step 5: Clean up
    logger.info("Step 5: Cleaning up...")
    os.remove(temp_parquet)
    assert not os.path.exists(temp_parquet), "Parquet file was not removed"
    logger.info(f"Removed: {temp_parquet}")
    
    logger.info("Integration test completed successfully!")


if __name__ == "__main__":
    test_corpus_integration()
