import re


def preprocess_text(text: str) -> str:
    """
    Preprocess text for indexing or querying.

    Args:
        text (str): The input text to preprocess.

    Returns:
        str: The preprocessed text.
    """
    if not text:
        return text
    # Replace underscores with spaces
    text = text.replace('_', ' ')
    # Split camelCase
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([A-Z])([A-Z][a-z])', r'\1 \2', text)
    return text
