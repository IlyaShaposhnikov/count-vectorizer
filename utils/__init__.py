"""
Пакет утилит для проекта CountVectorizer.
"""

from .nltk_utils import download_nltk_resources, get_wordnet_pos

__all__ = [
    'download_nltk_resources',
    'get_wordnet_pos'
]
