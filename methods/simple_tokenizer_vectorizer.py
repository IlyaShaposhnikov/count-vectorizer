"""
CountVectorizer with a simple tokenizer based on split().
This is the fastest but least accurate tokenization method.
"""

from sklearn.feature_extraction.text import CountVectorizer

from utils.vectorizer_utils import calculate_common_vectorizer_metrics


def simple_tokenizer(s):
    """
    Simple tokenizer based on split().
    Splits the string by whitespace without considering punctuation.

    Args:
        s (str): Input string

    Returns:
        list: List of tokens
    """
    return s.split()


def create_simple_tokenizer_vectorizer():
    """
    Creates a CountVectorizer with a simple tokenizer.

    Returns:
        CountVectorizer: Vectorizer with simple tokenizer
    """
    vectorizer = CountVectorizer(
        # Use simple tokenizer
        tokenizer=simple_tokenizer,
        lowercase=True,
        max_df=0.95,
        min_df=2,
        max_features=None,
        token_pattern=None
    )

    return vectorizer


def get_vectorizer_info(vectorizer, X_train):
    """
    Returns information about the simple tokenizer vectorizer.

    Args:
        vectorizer: Fitted vectorizer
        X_train: Transformed training data

    Returns:
        dict: Dictionary containing vectorizer information
    """
    # Prepare specific info for this method
    extra_info = {
        'name': 'CountVectorizer with simple tokenizer',
        'tokenizer_type': 'simple_tokenizer (split)',
        'note': (
            'Note: simple split() does not handle punctuation correctly'
        ),
        # example_features is handled by the common function
    }

    # Calculate common metrics using the shared function
    return calculate_common_vectorizer_metrics(vectorizer, X_train, extra_info)


if __name__ == "__main__":
    # Example usage
    print("CountVectorizer with Simple Tokenizer Module")
    sample_text = "I'm going to the park. It's beautiful!"
    print(f"Example tokenization: '{sample_text}'")
    print(f"Result: {simple_tokenizer(sample_text)}")
    print("Note: 'I'm' is not separated into 'I' and 'am'")
