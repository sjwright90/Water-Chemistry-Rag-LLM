# %%

from langchain_ollama import ChatOllama

from langchain_core.messages import SystemMessage

# Vector store and embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb
from torch import cuda

from langgraph.graph import END, MessagesState, StateGraph
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition

from langgraph.checkpoint.redis import RedisSaver
from langgraph.store.redis import RedisStore
from langgraph.store.base import BaseStore
from langchain_core.runnables import RunnableConfig

import os
import uuid

from src.output_manager import clean_response_output
from src.config import get_app_parser

user = os.getlogin()


device = "cuda" if cuda.is_available() else "cpu"


# %%
def main(user_id=user, device=device):
    # Parse command line arguments
    parser = get_app_parser()
    args = parser.parse_args()
    embedding_model = args.embedding_model
    chroma_store_path = args.chroma_store_path
    collection_name = args.collection_name
    model_name = args.llm_model
    temperature = args.temperature
    DB_URI = args.redis_uri
    reset_memories = args.delete_memories

    k_retrieval = args.k_retrieval
    if not k_retrieval:
        k_retrieval = 5

    if args.user_id:
        user_id = args.user_id
    print(f"Using user ID: {user_id}")

    thread_id = uuid.uuid4().hex  # Generate a new UUID if not provided
    if args.thread_id:
        thread_id = args.thread_id
    print(f"Using thread ID: {thread_id}")

    embedding_fn = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": device},
    )

    persistent_client = chromadb.PersistentClient(path=chroma_store_path)
    vectorstore = Chroma(
        client=persistent_client,
        collection_name=collection_name,
        embedding_function=embedding_fn,
    )
    # print chunks loaded from the vector store
    print(f"{vectorstore._collection.count()} chunks loaded from disk")

    llm = ChatOllama(
        model=model_name,
        temperature=temperature,
    )

    @tool(response_format="content_and_artifact")
    def retrieve(query: str) -> tuple[str, list]:
        """Retrieve information related to a query."""
        retrieved_docs = vectorstore.similarity_search(query, k=k_retrieval)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    # Step 1: Generate an AIMessage that may include a tool-call to be sent.
    def query_or_respond(
        state: MessagesState,
    ):
        """Generate tool call for retrieval or respond."""
        llm_with_tools = llm.bind_tools([retrieve])
        response = llm_with_tools.invoke(state["messages"])

        # MessagesState appends messages to state instead of overwriting
        return {"messages": [response]}

    # Step 2: Execute the retrieval.
    tools = ToolNode([retrieve])

    # Step 3: Generate a response using the retrieved content.
    def generate(
        state: MessagesState,
        config: RunnableConfig,
        *,
        store: BaseStore,
    ):
        """Generate answer."""
        # Get generated ToolMessages
        user_id = config["configurable"]["user_id"]
        namespace = ("memories", user_id)

        if not config["configurable"].get("memory_loaded", False):
            # Retrieve relevant memories for this user
            memories = store.search(namespace, query=str(state["messages"][-1].content))
            user_info = "\n".join([d.value["data"] for d in memories])
            config["configurable"]["memory_loaded"] = True
            config["configurable"]["user_info"] = user_info
        else:
            # use .get just in case
            user_info = config["configurable"].get("user_info", "")

        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]

        # Format into prompt
        # docs_content = "\n\n".join(doc.content for doc in tool_messages)
        docs_content = "\n\n".join(doc.content for doc in tool_messages)
        system_message_content = (
            f"User info: {user_info}\n\n"
            "You are a helpful assistant specialized in hydrogeochemistry. "
            "You are an expert in hydrogeochemistry. Use ONLY the provided context to answer the question. "
            "If the context does not contain the answer, say: 'I don’t have enough information from the documents to answer.' "
            "Do not make up facts. Be concise, but thorough. Take as much space as you need to answer the question accurately."
            f"Documents:\n{docs_content}"
        )
        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]

        human_messages = [
            message for message in state["messages"] if message.type == "human"
        ]
        if "remember" in human_messages[-1].content.lower():
            # Store new memories if the user asks to remember something
            memory = human_messages[-1].content
            store.put(
                namespace, uuid.uuid4().hex, {"data": memory}
            )  # Use hex for UUID key for REDIS compatibility
        if "delete memories" in human_messages[-1].content.lower():
            try:
                memories = store.search(("memories", user_id))
                for memory in memories:
                    store.delete(("memories", user_id), memory.key)
            except Exception as e:
                print(f"Error deleting memories: {e}")
        # store.delete(namespace, "*")
        # print(f"Deleted all memories in namespace {namespace} for user {user_id}")
        prompt = [SystemMessage(system_message_content)] + conversation_messages
        # Run
        response = llm.invoke(prompt)
        # Return the response as a list of messages
        return {"messages": [response]}

    # Specify an ID for the thread

    config = {
        "configurable": {
            "thread_id": thread_id,  # Unique ID for the thread
            "user_id": user_id,
        },
    }
    # Run the main application logic
    with (
        RedisStore.from_conn_string(DB_URI) as redis_store,
        RedisSaver.from_conn_string(DB_URI) as checkpointer,
    ):
        try:
            redis_store.list_namespaces()
            if reset_memories:
                memories = redis_store.search(("memories", user_id))
                for memory in memories:
                    print(f"Deleting memory: {memory.value.get('data', 'No data')}")
                    redis_store.delete(("memories", user_id), memory.key)
                print(f"Deleted all memories for user {user_id}")
        except Exception as e:
            print(f"Error listing namespaces: {e}")
            # set up the Redis store if it doesn't exist
            redis_store.setup()

        try:
            checkpointer.setup()
        except Exception as e:
            print(f"Error setting up checkpointer: {e}")

        graph_builder = StateGraph(MessagesState)

        graph_builder.add_node(query_or_respond)
        graph_builder.add_node(tools)
        graph_builder.add_node(generate)

        graph_builder.set_entry_point("query_or_respond")
        graph_builder.add_conditional_edges(
            "query_or_respond",
            tools_condition,
            {END: END, "tools": "tools"},
        )
        graph_builder.add_edge("tools", "generate")
        graph_builder.add_edge("generate", END)

        graph = graph_builder.compile(
            checkpointer=checkpointer,
            store=redis_store,
        )

        # print(f"Using thread ID: {config['configurable']['thread_id']}")

        while True:
            input_message = input("Enter your message (or 'exit' to quit): ")
            if input_message.lower() == "exit":
                print("Exiting the chat.")
                break
            if input_message.strip() == "":
                print("Empty input, please enter a valid message.")
                continue
            for step in graph.stream(
                {"messages": [{"role": "user", "content": input_message}]},
                config=config,
                stream_mode="values",
            ):
                clean_response_output(step["messages"][-1])


if __name__ == "__main__":
    main()

# %%
