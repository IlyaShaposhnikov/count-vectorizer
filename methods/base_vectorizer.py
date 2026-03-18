"""
Base CountVectorizer method without additional processing.
Uses standard sklearn CountVectorizer settings.
"""
from sklearn.feature_extraction.text import CountVectorizer

from utils.vectorizer_utils import calculate_common_vectorizer_metrics


def create_base_vectorizer():
    """
    Creates and returns a basic CountVectorizer.

    Returns:
        CountVectorizer: Basic vectorizer with default settings
    """
    # Use standard CountVectorizer parameters
    vectorizer = CountVectorizer(
        # Convert text to lowercase
        lowercase=True,
        # Standard token pattern
        token_pattern=r'(?u)\b\w\w+\b',
        # Maximum word frequency across documents
        max_df=1.0,
        # Minimum word frequency across documents
        min_df=1,
        # No limit on number of features
        max_features=None
    )

    return vectorizer


def get_vectorizer_info(vectorizer, X_train):
    """
    Returns information about the vectorizer and transformed data.

    Args:
        vectorizer: Fitted vectorizer
        X_train: Transformed training data

    Returns:
        dict: Dictionary containing vectorizer information
    """
    # Prepare specific info for this method
    extra_info = {
        'name': 'Basic CountVectorizer',
    }
    # Calculate common metrics using the shared function
    return calculate_common_vectorizer_metrics(vectorizer, X_train, extra_info)


if __name__ == "__main__":
    # Example usage
    print("Basic CountVectorizer module")
    print("Usage: import create_base_vectorizer() in main.py")
