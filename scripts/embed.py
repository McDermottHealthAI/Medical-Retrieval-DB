import argparse
import time

import wandb
from datasets import IterableDataset, load_dataset

from medretrieval import Corpus, Embedding


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for medical text documents")
    parser.add_argument("--dataset_path", type=str, required=True, help="HuggingFace dataset path (e.g. 'namespace/dataset')")
    parser.add_argument("--dataset_name", type=str, required=True, help="HuggingFace dataset config name")
    parser.add_argument("--dataset_split", type=str, required=True, help="Dataset split (e.g. 'train')")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name for embedding")
    parser.add_argument("--chunk_size", type=int, default=500, help="Maximum tokens per chunk (default: 500)")
    parser.add_argument("--document_id_column", type=str, default="document_id", help="Column to use as document ID (default: document_id)")
    parser.add_argument("--content_column", type=str, default="content", help="Column to use as content (default: content)")
    parser.add_argument("--output_dir", type=str, default="experiments/embeddings/", help="Directory to save embeddings")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to embed (default: all)")
    parser.add_argument("--num_shards", type=int, default=None, help="Total number of shards (required when using --shard_index)")
    parser.add_argument("--shard_index", type=int, default=None, help="Shard index to process (default: None)")
    parser.add_argument("--wandb_project", type=str, default="medical-retrieval-embeddings", help="Weights & Biases project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="Weights & Biases run name")
    args = parser.parse_args()

    if args.shard_index is not None and args.num_shards is None:
        parser.error("--num_shards is required when --shard_index is set")

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config={
            "dataset_path": args.dataset_path,
            "dataset_name": args.dataset_name,
            "dataset_split": args.dataset_split,
            "model": args.model,
            "chunk_size": args.chunk_size,
            "num_samples": args.num_samples,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "document_id_column": args.document_id_column,
            "content_column": args.content_column,
        },
    )

    print(f"Loading dataset {args.dataset_path}/{args.dataset_name} ({args.dataset_split})")
    dataset = load_dataset(args.dataset_path, args.dataset_name, split=args.dataset_split, streaming=True)

    if args.shard_index is not None:
        dataset = dataset.shard(num_shards=args.num_shards, index=args.shard_index)
    if args.num_samples is not None:
        dataset = dataset.take(args.num_samples)
    if args.document_id_column != "document_id":
        dataset = dataset.rename_column(args.document_id_column, "document_id")
    if args.content_column != "content":
        dataset = dataset.rename_column(args.content_column, "content")

    if isinstance(dataset, IterableDataset):
        all_columns = next(iter(dataset)).keys()
    else:
        all_columns = dataset.column_names
    dataset = dataset.remove_columns(set(all_columns) - {"document_id", "content"})

    print("Generating embeddings...")
    start = time.time()
    embedding = Embedding(model_name=args.model, tokens_per_chunk=args.chunk_size)
    dataset_with_embeddings = embedding.embed(dataset, build_faiss_index=False)

    output_file_name = (
        f"{args.dataset_path}_{args.dataset_name}_{args.model}_{args.chunk_size}__{args.shard_index}.parquet"
        .replace("/", "_")
    )
    print(f"Saving to {args.output_dir}/{output_file_name}")
    Corpus.save(dataset_with_embeddings, f"{args.output_dir}/{output_file_name}")

    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s")
    wandb.log({"total_time_seconds": elapsed})
    wandb.finish()


if __name__ == "__main__":
    main()
