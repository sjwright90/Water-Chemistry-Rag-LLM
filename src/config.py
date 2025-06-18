import argparse


def get_ingest_parser():
    parser = argparse.ArgumentParser(description="Embed PDF documents")
    parser.add_argument(
        "--dirs",
        type=str,
        default="target_directories.txt",
        help="File containing directories to search for PDFs",
    )
    parser.add_argument(
        "--direct_add",
        type=str,
        help="File containing absolute path to PDFs to add to parse list",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        help="New line separated list of substrings to ignore in PDF names",
    )
    parser.add_argument(
        "--recurse",
        action="store_true",
        help="Recursive search of directories",
    )
    parser.add_argument(
        "--absolute_path",
        type=str,
        default="./",
        help="Absolute path to root of all directories",
    )
    parser.add_argument(
        "--large_file_manual_check",
        action="store_true",
        help="T/F flag to manually confirm large files. Size can be set in '--largeFileSizeFlag",
    )
    parser.add_argument(
        "--large_file_size_flag",
        type=int,
        default=100,
        help="Size (in MB) at which to request manual confirmation of file. Only active if calling '--largeFileManualCheck'",
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default="water_chemistry",
        help="Name of collection to store embeddings in",
    )

    parser.add_argument(
        "--embedding_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Model name for embeddings",
    )

    parser.add_argument(
        "--overrite_collection",
        action="store_true",
        help="WARNING! Overwrite collection name from '--collectionName' cannot be undone",
    )

    parser.add_argument(
        "--logs_dir",
        type=str,
        default="./logs",
        help="Directory to store logs, advised not to change",
    )

    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        help="Level of verbosity, only 0|1 at the moment. 1==Verbose",
    )

    # not working yet
    # parser.add_argument(
    #     "--key_phrases_drop",
    #     type=str,
    #     help="Readable file containing line separated phrases used to automatically exclude pages",
    # )
    parser.add_argument(
        "--chroma_store_path",
        type=str,
        default="chroma_store",
        help="Path to the ChromaDB persistent client store",
    )

    return parser


def get_app_parser():
    parser = argparse.ArgumentParser(description="Run the application")
    parser.add_argument(
        "--redis_uri",
        type=str,
        default="redis://localhost:6379",
        help="URI for Redis database",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default="llama3.2",
        help="Name of the language model used for generation (must be available in Ollama)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for model inference",
    )
    parser.add_argument(
        "--embedding_model",  # need to unify this syntax across parsers
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Model name for embeddings",
    )
    parser.add_argument(
        "--chroma_store_path",
        type=str,
        default="chroma_store",
        help="Path to the ChromaDB persistent client store",
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default="water_chemistry",
        help="Name of the collection in ChromaDB",
    )
    parser.add_argument(
        "--delete_memories",
        action="store_true",
        help="Reset all memories in the Redis store",
    )
    parser.add_argument(
        "--user_id",
        type=str,
        help="User ID for personalized memory management",
    )
    parser.add_argument(
        "--thread_id",
        type=str,
        help="Custom session ID for the user",
    )

    parser.add_argument(
        "--k_retrieval",
        type=int,
        default=5,
        help="Number of retrievals to perform from the vector store",
    )
    return parser
