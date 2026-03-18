"""
CountVectorizer with token lemmatization.
Lemmatization reduces words to their dictionary form (lemma).
"""

import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

from utils.nltk_utils import get_wordnet_pos
from utils.vectorizer_utils import calculate_common_vectorizer_metrics


class LemmaTokenizer:
    """
    Custom tokenizer for CountVectorizer with lemmatization.
    Reduces words to their base form considering part-of-speech tags.
    """

    def __init__(self):
        """Initializes the WordNet lemmatizer."""
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        """
        Tokenizes and lemmatizes a document.

        Args:
            doc (str): Input text

        Returns:
            list: List of lemmatized tokens
        """
        # Tokenize the document
        tokens = word_tokenize(doc)

        # Determine part-of-speech tags for each token
        words_and_tags = nltk.pos_tag(tokens)

        # Lemmatize each token considering its POS tag
        lemmas = [
            self.wnl.lemmatize(word, pos=get_wordnet_pos(tag))
            for word, tag in words_and_tags
        ]

        return lemmas


def create_lemmatization_vectorizer():
    """
    Creates a CountVectorizer with token lemmatization.

    Returns:
        CountVectorizer: Vectorizer with lemmatization
    """
    vectorizer = CountVectorizer(
        # Use custom tokenizer
        tokenizer=LemmaTokenizer(),
        lowercase=True,
        max_df=0.95,
        min_df=2,
        max_features=None,
        # Disable standard pattern
        # when using custom tokenizer
        token_pattern=None
    )

    return vectorizer


def get_vectorizer_info(vectorizer, X_train):
    """
    Returns information about the lemmatization vectorizer.

    Args:
        vectorizer: Fitted vectorizer
        X_train: Transformed training data

    Returns:
        dict: Dictionary containing vectorizer information
    """
    # Prepare specific info for this method
    extra_info = {
        'name': 'CountVectorizer with lemmatization',
        'tokenizer_type': 'LemmaTokenizer',
        # example_features is handled by the common function
    }

    # Calculate common metrics using the shared function
    return calculate_common_vectorizer_metrics(vectorizer, X_train, extra_info)


if __name__ == "__main__":
    # Example usage
    print("CountVectorizer with Lemmatization Module")
    tokenizer = LemmaTokenizer()
    sample_text = "The cats are running and jumping around."
    print(f"Example lemmatization: '{sample_text}'")
    print(f"Result: {tokenizer(sample_text)}")
