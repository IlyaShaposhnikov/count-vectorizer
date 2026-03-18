"""
Utilities for working with NLTK: downloading resources and helper functions.
"""
import nltk
from nltk.corpus import wordnet


def download_nltk_resources():
    """
    Downloads necessary NLTK resources.
    Called once during project initialization.
    """
    resources = [
        'wordnet',
        'punkt',
        'punkt_tab',
        'averaged_perceptron_tagger',
        'averaged_perceptron_tagger_eng',
        'stopwords'
    ]

    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
            print(f"NLTK resource '{resource}' downloaded")
        except Exception as e:
            print(f"Error downloading resource '{resource}': {e}")


def get_wordnet_pos(treebank_tag):
    """
    Converts a part-of-speech tag from the Penn Treebank format
    to the WordNet format.

    Args:
        treebank_tag (str): Part-of-speech tag in Penn Treebank format

    Returns:
        str: Part-of-speech tag in WordNet format
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        # Default to noun
        return wordnet.NOUN


if __name__ == "__main__":
    # Testing utilities
    download_nltk_resources()
    print("Tag 'NN' converts to:", get_wordnet_pos('NN'))
    print("Tag 'VBG' converts to:", get_wordnet_pos('VBG'))
