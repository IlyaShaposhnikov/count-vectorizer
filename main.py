"""
Main script for comparing various CountVectorizer methods.
Compares 5 text preprocessing approaches for BBC news classification.
"""

import time

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from methods.base_vectorizer import (
    create_base_vectorizer,
    get_vectorizer_info as get_base_info,
)
from methods.lemmatization_vectorizer import (
    create_lemmatization_vectorizer,
    get_vectorizer_info as get_lemmatization_info,
)
from methods.simple_tokenizer_vectorizer import (
    create_simple_tokenizer_vectorizer,
    get_vectorizer_info as get_simple_info,
)
from methods.stemming_vectorizer import (
    create_stemming_vectorizer,
    get_vectorizer_info as get_stemming_info,
)
from methods.stopwords_vectorizer import (
    create_stopwords_vectorizer,
    get_vectorizer_info as get_stopwords_info,
)
from utils.data_loader import load_data
from utils.nltk_utils import download_nltk_resources
from utils.reporting import print_detailed_comparison
from utils.visualization import visualize_results

TEST_SIZE = 0.25
RANDOM_STATE = 123


def evaluate_method(
        vectorizer_creator, info_getter, inputs_train,
        inputs_test, Ytrain, Ytest, method_name
):
    """
    Evaluates a vectorization method on data.

    Args:
        vectorizer_creator: Function to create the vectorizer
        info_getter: Function to get vectorizer information
        inputs_train: Training texts
        inputs_test: Test texts
        Ytrain: Training labels
        Ytest: Test labels
        method_name: Name of the method

    Returns:
        dict: Results of the evaluation
    """
    print(f"\n{'='*60}")
    print(f"Evaluating method: {method_name}")
    print(f"{'='*60}")

    start_time = time.time()

    # Create and fit the vectorizer
    vectorizer = vectorizer_creator()
    print("Vectorizing training data...")
    X_train = vectorizer.fit_transform(inputs_train)

    # Transform test data
    print("Vectorizing test data...")
    X_test = vectorizer.transform(inputs_test)

    # Get vectorizer info
    vectorizer_info = info_getter(vectorizer, X_train)

    # Train the Naive Bayes model
    print("Training MultinomialNB model...")
    model = MultinomialNB()
    model.fit(X_train, Ytrain)

    # Evaluate the model
    train_score = model.score(X_train, Ytrain)
    test_score = model.score(X_test, Ytest)

    end_time = time.time()
    execution_time = end_time - start_time

    # Collect results
    results = {
        'method_name': method_name,
        'train_accuracy': train_score * 100,
        'test_accuracy': test_score * 100,
        'vocabulary_size': vectorizer_info['vocabulary_size'],
        'density_percent': vectorizer_info.get('density_percent', 0),
        'execution_time': execution_time,
        'vectorizer_info': vectorizer_info
    }

    print(f"  Training accuracy: {train_score:.3%}")
    print(f"  Test accuracy: {test_score:.3%}")
    print(f"  Vocabulary size: {vectorizer_info['vocabulary_size']} words")
    print(
        "  Matrix density: "
        f"{vectorizer_info.get('density_percent', 0):.2f}%"
    )
    print(f"  Execution time: {execution_time:.2f} seconds")

    return results


def main():
    """
    Main function to compare CountVectorizer methods.
    """
    print("="*80)
    print("COUNTVECTORIZER TEXT PREPROCESSING METHODS COMPARISON")
    print("="*80)

    # Load NLTK resources
    print("\nDownloading NLTK resources...")
    download_nltk_resources()

    # Load data
    inputs, labels = load_data()

    # Split data into train and test sets
    print("\nSplitting data into train and test sets...")
    inputs_train, inputs_test, Ytrain, Ytest = train_test_split(
        inputs, labels, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=labels
    )

    print(f"  Training set: {len(inputs_train)} documents")
    print(f"  Test set: {len(inputs_test)} documents")

    # Define methods for comparison
    methods = [
        {
            'name': 'Basic',
            'vectorizer_creator': create_base_vectorizer,
            'info_getter': get_base_info
        },
        {
            'name': 'With Stop Words Removal',
            'vectorizer_creator': create_stopwords_vectorizer,
            'info_getter': get_stopwords_info
        },
        {
            'name': 'With Lemmatization',
            'vectorizer_creator': create_lemmatization_vectorizer,
            'info_getter': get_lemmatization_info
        },
        {
            'name': 'With Stemming',
            'vectorizer_creator': create_stemming_vectorizer,
            'info_getter': get_stemming_info
        },
        {
            'name': 'With Simple Tokenizer',
            'vectorizer_creator': create_simple_tokenizer_vectorizer,
            'info_getter': get_simple_info
        }
    ]

    # Evaluate each method
    all_results = []

    for method in methods:
        results = evaluate_method(
            method['vectorizer_creator'],
            method['info_getter'],
            inputs_train,
            inputs_test,
            Ytrain,
            Ytest,
            method['name']
        )
        all_results.append(results)

    # Print detailed comparison
    print_detailed_comparison(all_results)

    # Visualize results
    try:
        visualize_results(pd.DataFrame(all_results))
    except Exception as e:
        print(f"\nVisualization error: {e}")
        print("Continuing without visualization...")

    print("\n" + "="*80)
    print("COMPARISON FINISHED!")
    print("="*80)


if __name__ == "__main__":
    main()
