import argparse
import time

import wandb
from medretrieval import Corpus, Embedding
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for medical text documents")
    parser.add_argument("--dataset_url", type=str, required=True, help="HF dataset name")
    parser.add_argument("--dataset_name", type=str, required=True, help="HF dataset name")
    parser.add_argument("--dataset_split", type=str, required=True, help="HF dataset split")
    parser.add_argument("--model", type=str, required=True, help="HF model name")
    parser.add_argument("--chunk_size", type=int, default=500, help="Maximum number of tokens per chunk (default: 500)")
    parser.add_argument("--document_id_column", type=str, default="document_id", help="Column name to use as document ID (default: document_id)")
    parser.add_argument("--content_column", type=str, default="content", help="Column name to use as content (default: content)")
    parser.add_argument("--output_dir", type=str, default="experiments/embeddings/", help="Directory to save the dataset with embeddings")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to take from the dataset (default: 500)")
    parser.add_argument("--wandb_project", type=str, default="medical-retrieval-embeddings", help="Weights & Biases project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="Weights & Biases run name")
    args = parser.parse_args()

    # Initialize wandb for tracking GPU/CPU utilization and metrics
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config={
            "dataset_url": args.dataset_url,
            "dataset_name": args.dataset_name,
            "dataset_split": args.dataset_split,
            "model": args.model,
            "chunk_size": args.chunk_size,
            "num_samples": args.num_samples,
            "document_id_column": args.document_id_column,
            "content_column": args.content_column,
        },
    )

    # Load dataset
    print(f"Loading dataset from {args.dataset_url}/{args.dataset_name}/{args.dataset_split}")
    dataset = load_dataset(args.dataset_url, args.dataset_name, split=args.dataset_split, streaming=True)
    if args.num_samples is not None:
        dataset = dataset.take(args.num_samples)
    if args.document_id_column != "document_id":
        dataset = dataset.rename_column(args.document_id_column, "document_id")
    if args.content_column != "content":
        dataset = dataset.rename_column(args.content_column, "content")
    dataset = dataset.remove_columns(set(dataset.column_names) - set(['document_id', 'content']))

    # Generate embeddings
    start = time.time()
    print(f"Generating embeddings...")
    embedding = Embedding(model_name=args.model, tokens_per_chunk=args.chunk_size)
    dataset_with_embeddings = embedding.embed(dataset, build_faiss_index=False)

    print(f"Saving dataset with embeddings to {args.output_dir}")
    output_file_name = f"{args.dataset_url}_{args.dataset_name}_{args.model}_{args.chunk_size}.parquet".replace("/", "_")
    Corpus.save(dataset_with_embeddings, f"{args.output_dir}/{output_file_name}")
    end = time.time()
    elapsed_time = end - start
    print(f"Generated embeddings in {elapsed_time} seconds")
    
    # Log metrics to wandb
    wandb.log({
        "total_time_seconds": elapsed_time,
    })
    
    print("Done!")
    wandb.finish()


if __name__ == "__main__":
    main()
