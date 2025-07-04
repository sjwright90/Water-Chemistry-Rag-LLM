# Custom RAG System for PDF-Based Knowledge Retrieval

This project provides a tool set for building retrieval-augmented generation (RAG) applications on top of domain-specific documents. The framework allows you to ingest selected files, store their embeddings in different ChromaDB collections, and spin up a chatbot that answers questions using only the retrieved content. RAG logic and graph-based control flow are built entirely using the [LangChain](https://www.langchain.com/) and [LangGraph](https://github.com/langchain-ai/langgraph) frameworks. The chatbot is powered by [Ollama](https://ollama.com/) and runs entirely on your local machine. Short- and long-term memory persistence is handled via a Redis database hosted in Docker. The system is currently tuned for working with water chemistry data, but any domain can be supported, just edit the "system_message_content" in the `rag_chatbot.py` script to change the context for the chatbot.

---

## Key Features

- Ingest any set of PDFs with custom filters and QA checks (support for other document types can be added)
- Use separate ChromaDB vectorstores per topic, project, or extend existing collections with new information
- Query documents via a conversational terminal interface
- Add or delete user-specific memory with Redis
- Powered by local LLMs via [Ollama](https://ollama.com/) — no API keys needed
- Easily swap in different embedding models (e.g., `all-MiniLM-L6-v2`)

---

## Use Case: Why Multiple Vectorstores?

This setup is ideal when:
- You're working with distinct projects, facilities, or domains
- Each set of PDFs should be kept separate (e.g., software tutorials, hydrogeochemistry reports)
- You want to test retrieval performance per domain without cross-contamination
- You want to iterate rapidly across different document sets


## Project Structure
├── chroma_store/ # Persistent Chroma vector DB, not on GitHub</br>
├── ingest_inputs/ # Text files listing folders, exclusions, etc. (optional)</br>
├── logs/ # Log files for ingestion and chat operations, not on GitHub</br>
├── src/</br>
│ ├── config.py # CLI argument parsers</br>
│ ├── qaqc_checks.py # Text cleaning, filtering, deduplication logic</br>
│ ├── chroma_manager.py # Vectorstore collection utilities</br>
│ └── output_manager.py # Output formatting utilities</br>
├── utils/</br>
│ ├── chroma_db_list_collections.py # List Chroma collections</br>
├── ingest_module.py # Ingest PDF documents into Chroma</br>
├── volumes/ # Docker volumes for Redis and ChromaDB, not on GitHub</br>
├── docker-compose.yml # Docker Compose file to run Redis</br>
├── rag_chatbot.py # Conversational interface to query documents via Ollama</br>
├── README.md # This file </br>
├── requirements.txt</br>

## Getting Started
### Prerequisites
- Python 3.13.4 (has not been tested with other versions)
- Required Python packages installed (`pip install -r requirements.txt`)
- Ollama installed and running locally (see [Ollama installation instructions](https://ollama.com/))
- Docker installed and running (see [Docker installation instructions](https://www.docker.com/get-started/))
    - Windows users: ensure Docker Desktop is set to use [WSL2 backend](https://docs.docker.com/desktop/windows/wsl/)
- Redis installed and running (highly recommended to use [Docker for Redis](https://redis.io/learn/howtos/quick-start))

### Ingesting Documents
Documents are ingested into ChromaDB collections using the `ingest_module.py` script. This script reads directories from a text file, processes all PDFs found within them, and stores their embeddings in a specified collection.


To run the ingestion script, you need to specify the directories containing your PDFs, the collection name, the root directory for all the directories to search, and optionally an embedding model. The script will read the directories from a text file and process all PDFs found within them. 

To ingest documents, run the following command:

```bash
python ingest_module.py \
  --dirs ingest_components/folders.txt \
  --collection_name MY_DOCUMENT_COLLECTION \
  --absolute_path path/to/root/folder \
```
To see all available options, run:
```bash
python ingest_module.py --help
```
The script provides some very simple QA/QC checks and cleaning of document text, but for best perfomance it is recommended to be intentional with document curation.

### Running the Chatbot
Once your documents are ingested, you can run the chatbot to query the documents using a conversational interface. The chatbot uses Ollama to generate responses based on the retrieved content from the specified ChromaDB collection. The chatbot runs in the terminal.

```bash
python rag_chatbot.py \
  --collection_name MY_DOCUMENT_COLLECTION \ # Existing collection name
```

To see all available options, run:
```bash
python rag_chatbot.py --help
```
Default settings use a user id taken from login name and thread id generated by uuid4, but can be overridden with `--user_id` and `--thread_id` arguments. This allows you to have multiple users and threads in the same Redis database.
### Listing ChromaDB Collections
You can list all existing ChromaDB collections using the `chroma_db_list_collections.py` script. This is useful to see what collections you have available for querying.
To list collections, run the following command:

```bash
python utils/chroma_db_list_collections.py
```

## Tips
### Ingesting Documents
- Default settings work well for most use cases, you usually just need to specify:
  - `--dirs`: Path to a text file listing directories to search for PDFs
  - `--collection_name`: Name of the ChromaDB collection to store embeddings
  - `--absolute_path`: Root directory where the directories are located
- Other useful options include:
  - `--direct_add`: Add PDFs directly using absolute path to the file (requires creation of text file with newline separated paths)
  - `--exclude`: Exclude PDFs by matching substring in the file name
  - `--recurse`: Recurse search of directories listed in `--dirs`
### Chatbot Redis Memory
- To have the chatbot remember specific information include the keyword `remember` in your query, followed by the information you want to remember.
- To have the chatbot forget specific information, include the keyword `delete memories` in your query, this will delete all memories stored in Redis for that user.
- Memories can also be deleted when you launch the chatbot by passing the `--delete_memories` flag. This will clear all memories stored in Redis before starting the chatbot session.
### Ollama Model Configuration
- Make sure the model you specify in the `rag_chatbot.py` script is available in your Ollama installation.
- You can check with with models are available by running:
```bash
ollama list
```
### Swapping LLM Providers
- Swap out ChatOllama for any LangChain-compatible LLM (e.g. ChatOpenAI, ChatAnthropic) to use cloud-hosted models via API key.

## Future Work
- Add support for more document types (e.g., Word, Excel)
- Implement more advanced QA/QC checks during ingestion
- Wrap into a web application for easier access
- Full Docker Compose setup to run everything in containers
#### References
- [LangChain Documentation](https://www.langchain.com/docs/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Ollama Documentation](https://ollama.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Redis Documentation](https://redis.io/documentation)
- [Tutorial: Build a Retrieval Augmented Generation (RAG) App: Part 1](https://python.langchain.com/docs/tutorials/rag/)
- [Tutorial: Build a Retrieval Augmented Generation (RAG) App: Part 2](https://python.langchain.com/docs/tutorials/qa_chat_history/)