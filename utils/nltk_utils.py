"""
Утилиты для работы с NLTK: загрузка ресурсов и вспомогательные функции.
"""

import nltk
from nltk.corpus import wordnet


def download_nltk_resources():
    """
    Загружает необходимые ресурсы NLTK.
    Вызывается один раз при инициализации проекта.
    """
    resources = [
        'wordnet',
        'punkt',
        'punkt_tab',
        'averaged_perceptron_tagger',
        'stopwords'
    ]

    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
            print(f"Ресурс NLTK '{resource}' загружен")
        except Exception as e:
            print(f"Ошибка загрузки ресурса '{resource}': {e}")


def get_wordnet_pos(treebank_tag):
    """
    Преобразует тег части речи из формата Penn Treebank в формат WordNet.

    Args:
        treebank_tag (str): Тег части речи в формате Penn Treebank

    Returns:
        str: Тег части речи в формате WordNet
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
        # По умолчанию считаем существительным
        return wordnet.NOUN


if __name__ == "__main__":
    # Тестирование утилит
    download_nltk_resources()
    print("Тег 'NN' преобразуется в:", get_wordnet_pos('NN'))
    print("Тег 'VBG' преобразуется в:", get_wordnet_pos('VBG'))
