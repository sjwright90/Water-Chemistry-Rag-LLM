# %%
import os
from pathlib import Path
from datetime import datetime
import re

import fitz  # PyMuPDF

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

import chromadb

from src.chroma_manager import overwrite_selected_collection, chunk_batches
from src.qaqc_checks import (
    clean_text,
    filter_page_fullness,
    filter_by_phrase,
    deduplicate_chunks,
)
from src.config import get_ingest_parser

from torch import cuda

import json

import re

# import argparse

import logging

logger = logging.getLogger(__name__)


def main():
    parser = get_ingest_parser()
    args = parser.parse_args()
    server = Path(args.absolute_path)
    # collect the args
    collection_name = args.collection_name
    overwrite_collection = args.overrite_collection
    file_dir = args.dirs
    file_exclude = args.exclude
    file_direct_add = args.direct_add
    # file_key_phrases_drop = args.key_phrases_drop
    rglob = args.recurse
    verbose = args.verbose
    large_file_check = args.large_file_manual_check
    large_file_size = args.large_file_size_flag
    embedding_model = args.embedding_model
    PERSISTDIRECTORY = args.chroma_store_path

    logs_dir = Path(args.logs_dir) / collection_name
    logs_dir.mkdir(exist_ok=True, parents=True)

    print(
        "NOTE: Unique PDFs are identified by file name only.\nIf you have multiple PDFs with the same name in different directories,\nthey will be treated as the same PDF and might not be processed."
    )

    logging.basicConfig(
        filename=str(logs_dir / "ingest_module.log"), level=logging.INFO
    )

    logged_pdfs = logs_dir / f"embedded_pdfs_log.json"
    # Load the log file if it exists
    if logged_pdfs.exists():
        with open(logged_pdfs, "r") as f:
            embedded_pdfs_log = set(json.load(f))
        logger.info(f"Pulling in embedded pdfs from {logged_pdfs.as_posix()}")
    else:
        embedded_pdfs_log = set()

    exclude_strings = []

    device = "cuda" if cuda.is_available() else "cpu"
    model_name = embedding_model
    embedding_fn = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
    )
    message_setup = f"Using {model_name} with {device}\nStoring to collection {collection_name} ChromaDB in {PERSISTDIRECTORY}"

    logger.info(message_setup)
    print(message_setup)

    persistent_client = chromadb.PersistentClient(path=PERSISTDIRECTORY)

    if overwrite_collection:
        overwritten = overwrite_selected_collection(persistent_client, collection_name)
        if overwritten:
            embedded_pdfs_log = (
                set()
            )  # we have to overwrite the logged pdfs if they exist

    vectorstore = Chroma(
        client=persistent_client,
        collection_name=collection_name,
        embedding_function=embedding_fn,
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=lambda x: len(re.sub(r"\s+", " ", x)),
    )

    try:
        with open(file_dir, "r") as fname:
            subfolders_to_target = [line.strip() for line in fname.readlines()]
    except Exception as e:
        except_msg = f"Exception {e} occured while reading folders. Exiting"
        print(except_msg)
        logger.error(except_msg)
        return

    if file_exclude:
        try:
            with open(file_exclude, "r") as fname:
                exclude_strings = [line.strip() for line in fname.readlines()]
        except Exception as e:
            except_msg = f"Exception {e} occured while strings to exclude. Exiting"
            print(except_msg)
            logger.error(except_msg)
            return

    lst_key_phrases_drop = []
    # Not working yet, need to fix the logic
    # if file_key_phrases_drop:
    #     try:
    #         with open(file_key_phrases_drop, "r") as fname:
    #             lst_key_phrases_drop = [line.strip() for line in fname.readlines()]
    #     except Exception as e:
    #         except_msg = f"Exception {e} occured while reading phrases to drop. Exiting"
    #         print(except_msg)
    #         logger.error(except_msg)
    #         return

    lst_pdfs = []
    for folder in subfolders_to_target:
        folder_path = server / folder
        if folder_path.exists():
            if rglob:
                for file in folder_path.rglob("*.pdf"):
                    if file.is_file() and not any(
                        [substr in file.name for substr in exclude_strings]
                    ):
                        lst_pdfs.append(file)
            else:
                for file in folder_path.glob("*.pdf"):
                    if file.is_file() and not any(
                        [substr in file.name for substr in exclude_strings]
                    ):
                        lst_pdfs.append(file)

    if file_direct_add:
        try:
            with open(file_direct_add, "r") as fname:
                lst_additional_files = [
                    Path(line.strip()) for line in fname.readlines()
                ]
            lst_pdfs.extend(lst_additional_files)
        except Exception as e:
            except_msg = f"Exception {e} occured while incorporating directly added pdfs. Exiting"
            print(except_msg)
            logger.error(except_msg)
            return
    # Remove duplicates based on file name
    unique_by_filename = {}
    msg_deduplicate_files = (
        f"Found {len(lst_pdfs)} PDFs before deduplication based on file name."
    )
    for pdf in lst_pdfs:
        key = pdf.name.lower()
        if key not in unique_by_filename:
            unique_by_filename[key] = pdf
    lst_pdfs = list(unique_by_filename.values())
    msg_deduplicate_files += f" After deduplication: {len(lst_pdfs)} PDFs."
    logger.info(msg_deduplicate_files)
    print(msg_deduplicate_files)
    logger.info("The following PDFs will be parsed:")
    logger.info("\n".join(pth.as_posix() for pth in lst_pdfs))  # lst_pdfs)
    if verbose > 0:
        print("The following PDFs will be parsed:")
        print(lst_pdfs)

    # show really large PDFs for manual review
    if large_file_check:
        lst_files_remove = []
        large_pdfs = [
            pdf
            for pdf in lst_pdfs
            if pdf.stat().st_size > large_file_size * 1024 * 1024
        ]  # > 100 MB
        for pdf in large_pdfs:
            while True:
                response = (
                    input(
                        f"Large PDF: {pdf} - Size: {pdf.stat().st_size / (1024 * 1024):.2f} MB. Keep [Y/n]"
                    )
                    .strip()
                    .lower()
                )
                if response.lower() == "n":
                    lst_files_remove.append(pdf)
                    break
                elif response.lower() == "y":
                    break
                else:
                    print("Y/n only")
                    continue

        for pdf_remove in lst_files_remove:
            idx_remove = lst_pdfs.index(pdf_remove)
            lst_pdfs.pop(idx_remove)
            msg_removed = f"Removed {pdf_remove}"
            print(msg_removed)
            logger.info(msg_removed)

    all_chunks = []
    dct_failed_parse = {}
    previously_embedded = embedded_pdfs_log.copy()
    for pdf in lst_pdfs:
        # if str(pdf.relative_to(server)) in embedded_pdfs_log:
        if pdf.name.lower() in embedded_pdfs_log:
            msg_skip = f"Skipping already embedded PDF: {pdf}"
            logger.info(msg_skip)
            if verbose > 0:
                print(msg_skip)
            continue
        try:
            if verbose > 0:
                print(f"Processing PDF: {pdf}")
            with fitz.open(pdf) as doc:
                for page_num in range(len(doc)):
                    text = doc[page_num].get_text()
                    # clean the text
                    text = clean_text(text)
                    # Filter out pages that are too short or have too few non-whitespace characters
                    if not filter_page_fullness(text, min_length=100, min_prop=0.1):
                        msg_skip = f"Skipping page {page_num + 1} in {pdf} due to insufficient content."
                        logger.info(msg_skip)
                        if verbose > 0:
                            print(msg_skip)
                        continue
                    # Filter out pages that do not contain key phrases
                    if not filter_by_phrase(text, lst_key_phrases_drop, min_count=1):
                        msg_skip = f"Skipping page {page_num + 1} in {pdf} due to missing key phrases."
                        logger.info(msg_skip)
                        if verbose > 0:
                            print(msg_skip)
                        continue
                    # Check if the text is not empty after cleaning
                    if text.strip():
                        chunks = splitter.create_documents(
                            texts=[text],
                            metadatas=[
                                {
                                    "source": pdf.resolve().as_posix(),
                                    "page": page_num + 1,
                                    "filename": pdf.name.lower(),
                                    "timestamp": datetime.now().isoformat(),
                                }
                            ],
                        )
                        all_chunks.extend(chunks)
            # Lower case the filename for consistency
            filename = pdf.name.lower()
            embedded_pdfs_log.add(filename)
        except Exception as e:
            msg_error_proc = f"Error processing {pdf}: {e}"
            logger.error(msg_error_proc)
            print(msg_error_proc)
            dct_failed_parse[str(pdf)] = str(e)

    if all_chunks:
        # Deduplicate chunks
        msg_deduplicate = f"Total chunks before deduplication: {len(all_chunks)}"
        all_chunks = deduplicate_chunks(all_chunks)
        msg_deduplicate += f", after deduplication: {len(all_chunks)}"
        logger.info(msg_deduplicate)
        print(msg_deduplicate)
        batch_size = 5000
        n_batches = len(all_chunks) // batch_size + (len(all_chunks) % batch_size > 0)
        for idx, batch in enumerate(chunk_batches(all_chunks, batch_size=batch_size)):
            vectorstore.add_documents(batch)
            print(
                f"Batch {idx + 1} of {n_batches} processed, {len(batch)} chunks added."
            )
    # Log the successfully embedded PDF
    with open(logged_pdfs, "w") as fname:
        json.dump(sorted(list(embedded_pdfs_log)), fname, indent=2)
    # # Print the number of PDFs processed
    newely_embedded = embedded_pdfs_log - previously_embedded
    msg_process_new = f"{len(embedded_pdfs_log)} submitted PDFs processed, {len(newely_embedded)} new PDFs embedded."
    logger.info(msg_process_new)
    print(msg_process_new)

    msg_newly_embedded = "\n".join(f"- {pdf}" for pdf in sorted(newely_embedded))
    msg_newly_embedded = "Newly embedded PDFs:\n" + msg_newly_embedded
    logger.info(msg_newly_embedded)
    if verbose > 0:
        print(msg_newly_embedded)
    with open(logs_dir / "failed_parse_log.json", "w") as f:
        json.dump(dct_failed_parse, f, indent=2)
    msg_failed_embed = f"Failed to parse {len(dct_failed_parse)} PDFs. Check '{logs_dir.as_posix()}/failed_parse_log.json' for details."
    logger.info(msg_failed_embed)
    print(msg_failed_embed)


if __name__ == "__main__":
    main()
