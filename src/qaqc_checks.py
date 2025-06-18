import re
import unicodedata
from hashlib import md5


def clean_text(text: str) -> str:
    # Normalize unicode (helps remove diacritics and strange encodings)
    text = unicodedata.normalize("NFKD", text)
    # Remove control characters (non-printable)
    text = re.sub(r"[\x00-\x1F\x7F-\x9F]", "", text)
    # Remove excessive ? or boxed characters
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # optional: replace non-ASCII with space
    return text.strip()


def filter_page_fullness(
    text: str, min_length: int = 100, min_prop: float = None
) -> bool:
    """
    Filter pages based on text length and proportion of non-whitespace characters.

    Args:
        text (str): The text content of the page.
        min_length (int): Minimum length of text to consider the page valid.
        min_prop (int): Minimum proportion of non-whitespace characters to consider the page valid.

    Returns:
        bool: True if the page is valid, False otherwise.
    """
    if len(text) < min_length:
        return False
    if min_prop is not None:
        non_whitespace_count = len(re.sub(r"\s+", "", text))
        if non_whitespace_count / len(text) < min_prop:
            return False
    return True


def filter_by_phrase(text: str, phrases: list[str], min_count: int = 1):
    """
    Filter text based on the presence of specific phrases.
    Args:
        text (str): The text content to filter.
        phrases (list[str]): List of phrases to check for in the text.
        min_count (int): Minimum number of phrases that must be present in the text.
    Returns:
        bool: True if the text contains at least min_count of the specified phrases, False otherwise.
    """
    if not phrases:
        return True  # If no phrases are provided, consider the text valid
    count = sum(1 for phrase in phrases if phrase.lower() in text.lower())
    return count >= min_count


def deduplicate_chunks(chunks: list[str]) -> list[str]:
    """
    Deduplicate a list of text chunks.

    Args:
        chunks (list[str]): List of text chunks to deduplicate.

    Returns:
        list[str]: List of unique text chunks.
    """
    seen = set()
    unique_chunks = []
    for doc in chunks:
        hash_value = md5(doc.page_content.encode("utf-8")).hexdigest()
        if hash_value not in seen:
            seen.add(hash_value)
            unique_chunks.append(doc)
    return unique_chunks
