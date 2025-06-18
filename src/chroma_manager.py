def overwrite_selected_collection(client, collection_name: str, confirm: bool = True):
    """
     Check if a collection exists and optionally delete it if it does.
     Args:
        client: The ChromaDB client instance.
        collection_name (str): The name of the collection to check.
        confirm (bool): Whether to prompt for confirmation before deleting.
    Returns:
        bool: True if the collection was deleted or did not exist, False otherwise.

    """
    existing = collection_name in [col.name for col in client.list_collections()]
    if existing:
        if confirm:
            response = (
                input(
                    f"Collection '{collection_name}' exists. Delete and overwrite? [y/N]: "
                )
                .strip()
                .lower()
            )
            if response != "y":
                print("Aborting overwrite.")
                return False
        try:
            client.delete_collection(name=collection_name)
            print(f"Deleted collection '{collection_name}'")
            return True
        except Exception as e:
            print(f"Error deleting collection: {e}")
            return False
    return True  # Proceed if collection doesn't exist


# batch the chunks in to the vectorstore
def chunk_batches(lst, batch_size=5000):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]
