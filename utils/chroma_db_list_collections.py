import chromadb
import argparse

parser = argparse.ArgumentParser(description="List ChromaDB collections")


def get_parser():
    parser.add_argument(
        "--chroma_store_path",
        type=str,
        default="./chroma_store",
        help="Path to the ChromaDB persistent client store",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    return parser


def main():
    args = get_parser().parse_args()
    choma_store_path = args.chroma_store_path
    persistent_client = chromadb.PersistentClient(path=choma_store_path)
    # vectorstore = Chroma(client=persistent_client)
    # collections = vectorstore._client.list_collections()
    collections = persistent_client.list_collections()
    print("Collections in ChromaDB:")
    for collection in collections:
        print(f"Name: {collection.name}")
        print(f"ID: {collection.id}")
        print()


if __name__ == "__main__":
    main()
