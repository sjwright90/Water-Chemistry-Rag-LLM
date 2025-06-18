from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


def split_tokens_content(content, n_words=12):
    """Split content into chunks of n_words."""
    words = content.split()
    words = [" ".join(words[i : i + n_words]) for i in range(0, len(words), n_words)]
    return "\n".join(words)


def format_returned_artifact(artifact):
    """Format the retrieved documents for better readability."""
    formatted_content = []
    for doc in artifact:
        source_info = doc.metadata.get("source", "Unknown source")
        page_info = f"Page: {doc.metadata.get('page', 'N/A')}"
        content = doc.page_content.strip()
        content = split_tokens_content(content, n_words=12)
        formatted_content.append(
            f"Source: {source_info}\n{page_info}\nContent:\n{content}"
        )
    return "\n\n".join(formatted_content)


def clean_response_output(step_message):
    if isinstance(step_message, HumanMessage):
        return
    elif isinstance(step_message, AIMessage):
        if step_message.content:
            step_message.pretty_print()
    elif isinstance(step_message, ToolMessage):
        if step_message.artifact:
            formatted_sources = format_returned_artifact(step_message.artifact)
            print("Retrieved documents:")
            print(formatted_sources)
    else:
        return
