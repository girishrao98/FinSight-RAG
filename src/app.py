import argparse

from data_utils import load_csv_dataset, normalize_text, build_embeddings, save_embeddings
from model_utils import build_vectorstore, create_rag_chain, load_vectorstore


def index_command(args: argparse.Namespace) -> None:
    df = load_csv_dataset(args.input_csv, text_column=args.text_column)
    documents = [normalize_text(text) for text in df["text"].tolist()]
    embeddings = build_embeddings(documents)
    save_embeddings(args.embeddings_path, embeddings)
    print(f"Saved embeddings to {args.embeddings_path}")


def query_command(args: argparse.Namespace) -> None:
    vectorstore = load_vectorstore(args.index_path)
    rag = create_rag_chain(vectorstore, openai_api_key=args.openai_key)
    answer = rag.run(args.query)
    print("\n=== Answer ===")
    print(answer)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FinSight-RAG starter CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Index text documents")
    index_parser.add_argument("input_csv", help="Path to CSV file containing text data")
    index_parser.add_argument("--text-column", default="text", help="Column containing document text")
    index_parser.add_argument("--embeddings-path", default="data/embeddings.npy", help="Output embeddings path")

    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument("--index-path", default="data/faiss_index", help="Path to saved FAISS index")
    query_parser.add_argument("--query", required=True, help="Question to ask")
    query_parser.add_argument("--openai-key", help="OpenAI API key")

    return parser


def main() -> None:
    args = create_parser().parse_args()
    if args.command == "index":
        index_command(args)
    elif args.command == "query":
        query_command(args)


if __name__ == "__main__":
    main()
