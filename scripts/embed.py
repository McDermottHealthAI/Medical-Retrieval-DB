import argparse
import time

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
    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from {args.dataset_url}/{args.dataset_name}/{args.dataset_split}")
    dataset = load_dataset(args.dataset_url, args.dataset_name, split=args.dataset_split, streaming=True)
    if args.document_id_column != "document_id":
        dataset = dataset.rename_column(args.document_id_column, "document_id")
    if args.content_column != "content":
        dataset = dataset.rename_column(args.content_column, "content")
    # dataset = dataset.remove_columns(set(dataset.column_names) - set(['document_id', 'content']))

    # Generate embeddings
    print(f"Generating embeddings...")
    embedding = Embedding(model_name=args.model, tokens_per_chunk=args.chunk_size)
    dataset_with_embeddings = embedding.embed(dataset, build_faiss_index=False)

    print(f"Saving dataset with embeddings to {args.output_dir}")
    start = time.time()
    Corpus.save(dataset_with_embeddings, f"{args.output_dir}/{args.dataset_name}_{args.model}_{args.chunk_size}.parquet")
    end = time.time()
    print(f"Generated embeddings in {end - start} seconds")
    print("Done!")


if __name__ == "__main__":
    main()

