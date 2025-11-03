# Medical-Retrieval-DB

[![Python 3.12+](https://img.shields.io/badge/-Python_3.12+-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3100/)
[![PyPI - Version](https://img.shields.io/pypi/v/package_name)](https://pypi.org/project/package_name/)
[![Documentation Status](https://readthedocs.org/projects/package_name/badge/?version=latest)](https://package_name.readthedocs.io/en/latest/?badge=latest)
[![Tests](https://github.com/McDermottHealthAI/Medical-Retrieval-DB/actions/workflows/tests.yaml/badge.svg)](https://github.com/McDermottHealthAI/Medical-Retrieval-DB/actions/workflows/tests.yaml)
[![Test Coverage](https://codecov.io/github/McDermottHealthAI/MHAL-template/graph/badge.svg?token=BV119L5JQJ)](https://codecov.io/github/McDermottHealthAI/Medical-Retrieval-DB)
[![Code Quality](https://github.com/McDermottHealthAI/Medical-Retrieval-DB/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/McDermottHealthAI/Medical-Retrieval-DB/actions/workflows/code-quality-main.yaml)
[![Contributors](https://img.shields.io/github/contributors/McDermottHealthAI/MHAL-template.svg)](https://github.com/McDermottHealthAI/Medical-Retrieval-DB/graphs/contributors)
[![Pull Requests](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/McDermottHealthAI/Medical-Retrieval-DB/pulls)
[![License](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/McDermottHealthAI/Medical-Retrieval-DB/blob/main/LICENSE)

A package for building, indexing, and querying large-scale medical literature embeddings.

## Installation

Install using `uv`:

```bash
uv sync
```

## Quick Start

The `medretrieval` package provides two main classes:

- **`Corpus`**: Loads medical text documents from `.txt` and `.parquet` files into Hugging Face Datasets
- **`Embedding`**: Generates embeddings and enables semantic search with FAISS

### Example Usage

```python
>>> from medretrieval import Corpus, Embedding
>>> from yaml_to_disk import yaml_disk
>>>
>>> test_data = '''
... diabetes.txt: "Diabetes is a chronic condition that affects blood sugar levels."
... heart.txt: "Heart disease is a leading cause of death worldwide."
... '''
>>> with yaml_disk(test_data) as temp_dir:
...     dataset = Corpus.load_data(temp_dir)
>>> embedding = Embedding("thomas-sounack/BioClinical-ModernBERT-base")
>>> dataset_with_embeddings = embedding.embed(dataset)
>>> scores, examples = embedding.query(dataset_with_embeddings, ["Diabetes treatment"], k=1)
>>> examples[0]["content"][0].startswith("Diabetes is a chronic condition")
True

```

## Features

- **Corpus Management**: Load `.txt` and `.parquet` files with support for streaming large datasets
- **Embedding Generation**: Generate embeddings using medical language models like BioClinical-ModernBERT
- **Semantic Search**: Fast similarity search using FAISS indexes
- **Batch Processing**: Efficient batch processing for both embedding generation and querying
