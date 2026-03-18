"""
Utility functions for vectorizer operations.
Contains common logic for calculating metrics
across different vectorization methods.
"""


def calculate_common_vectorizer_metrics(vectorizer, X_train, extra_info=None):
    """
    Calculates common metrics for a fitted vectorizer and its transformed data.

    This function handles the shared logic for vocabulary size, density,
    shape, and example features calculation, reducing duplication across
    different vectorization methods in the 'methods' module.

    Args:
        vectorizer: A fitted scikit-learn vectorizer object
        (e.g., CountVectorizer).
        X_train: The transformed training data (usually a sparse matrix).
        extra_info (dict, optional): A dictionary containing any additional
                                     specific information relevant to the
                                     particular vectorization method.
                                     Defaults to None.

    Returns:
        dict: A dictionary containing the calculated common metrics and any
              provided extra_info merged together. Keys typically include:
              - 'vocabulary_size' (int): Number of unique features
              (words/tokens).
              - 'density_percent' (float): Percentage of
                non-zero elements in X_train.
              - 'shape' (tuple): Shape of the transformed data matrix
                (n_samples, n_features).
              - 'example_features' (list): A sample of
                feature names from the vocabulary.
              Any keys from 'extra_info' will also be present.

    Raises:
        ValueError: If extra_info contains keys that conflict
        with common metrics.
    """
    vocabulary = vectorizer.vocabulary_

    # Calculate density
    total_elements = X_train.shape[0] * X_train.shape[1]
    if total_elements == 0:
        # Handle edge case of empty matrix to avoid division by zero
        density_percent = 0.0
    else:
        # Use .nnz attribute directly (efficient for sparse matrices)
        density_percent = (X_train.nnz / total_elements) * 100

    # Prepare the base dictionary with common metrics
    metrics_dict = {
        'vocabulary_size': len(vocabulary),
        'density_percent': density_percent,
        'shape': X_train.shape,
        # Example of first 10 features
        'example_features': sorted(list(vocabulary.keys()))[:10]
    }

    # Merge with any extra_info if provided
    if extra_info:
        overlapping_keys = set(metrics_dict.keys()) & set(extra_info.keys())
        if overlapping_keys:
            raise ValueError(
                "Keys in extra_info conflict "
                f"with common keys: {overlapping_keys}"
            )
        metrics_dict.update(extra_info)

    return metrics_dict
