"""
CountVectorizer with removal of stop words (common words).
Removing stop words helps reduce dimensionality and focus
on meaningful words.
"""

from sklearn.feature_extraction.text import CountVectorizer

from utils.vectorizer_utils import calculate_common_vectorizer_metrics


def create_stopwords_vectorizer():
    """
    Creates a CountVectorizer with removal of English stop words.

    Returns:
        CountVectorizer: Vectorizer with stop word removal
    """
    # Use built-in list of English stop words from sklearn
    vectorizer = CountVectorizer(
        # Remove standard English stop words
        stop_words='english',
        lowercase=True,
        token_pattern=r'(?u)\b\w\w+\b',
        # Ignore words appearing in >95% of documents
        max_df=0.95,
        # Ignore words appearing in less than 2 documents
        min_df=2,
        max_features=None
    )

    return vectorizer


def get_vectorizer_info(vectorizer, X_train):
    """
    Returns information about the stop word removal vectorizer.

    Args:
        vectorizer: Fitted vectorizer
        X_train: Transformed training data

    Returns:
        dict: Dictionary containing vectorizer information
    """
    # Get the list of removed stop words
    stop_words = vectorizer.get_stop_words()

    # Prepare specific info for this method
    extra_info = {
        'name': 'CountVectorizer with stop word removal',
        'removed_stopwords_count': len(stop_words) if stop_words else 0,
    }

    # Calculate common metrics using the shared function
    return calculate_common_vectorizer_metrics(vectorizer, X_train, extra_info)


if __name__ == "__main__":
    print("CountVectorizer with Stop Word Removal Module")
    vectorizer = create_stopwords_vectorizer()
    sample_texts = ["This is a sample.", "Another example text."]
    X = vectorizer.fit_transform(sample_texts)
    info = get_vectorizer_info(vectorizer, X)
    print(info)
