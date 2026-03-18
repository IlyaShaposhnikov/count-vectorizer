"""
CountVectorizer with token stemming.
Stemming truncates word endings, leaving the root (stem).
"""

from nltk import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

from utils.vectorizer_utils import calculate_common_vectorizer_metrics


class StemTokenizer:
    """
    Custom tokenizer for CountVectorizer with stemming.
    Uses the Porter Stemmer algorithm to find the word root.
    """

    def __init__(self):
        """Initializes the Porter stemmer."""
        self.porter = PorterStemmer()

    def __call__(self, doc):
        """
        Tokenizes and stems a document.

        Args:
            doc (str): Input text

        Returns:
            list: List of stemmed tokens
        """
        # Tokenize the document
        tokens = word_tokenize(doc)

        # Apply stemming to each token
        stems = [self.porter.stem(t) for t in tokens]

        return stems


def create_stemming_vectorizer():
    """
    Creates a CountVectorizer with token stemming.

    Returns:
        CountVectorizer: Vectorizer with stemming
    """
    vectorizer = CountVectorizer(
        # Use custom tokenizer
        tokenizer=StemTokenizer(),
        lowercase=True,
        max_df=0.95,
        min_df=2,
        max_features=None,
        token_pattern=None
    )

    return vectorizer


def get_vectorizer_info(vectorizer, X_train):
    """
    Returns information about the stemming vectorizer.

    Args:
        vectorizer: Fitted vectorizer
        X_train: Transformed training data

    Returns:
        dict: Dictionary containing vectorizer information
    """
    extra_info = {
        'name': 'CountVectorizer with stemming',
        'tokenizer_type': 'StemTokenizer',
    }

    # Calculate common metrics using the shared function
    return calculate_common_vectorizer_metrics(vectorizer, X_train, extra_info)


if __name__ == "__main__":
    # Example usage
    print("CountVectorizer with Stemming Module")
    tokenizer = StemTokenizer()
    sample_text = "running runners ran"
    print(f"Example stemming: '{sample_text}'")
    print(f"Result: {tokenizer(sample_text)}")
