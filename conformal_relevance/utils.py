"""Utility functions for conformal relevancy."""

# Re-export the NLTK-based split_sentences from data_loading for backward
# compatibility.  The previous regex-based implementation has been removed
# in favour of the more robust NLTK tokenizer.
from conformal_relevance.data_loading._common import split_sentences

__all__ = ["split_sentences"]
