"""
CountVectorizer с лемматизацией токенов.
Лемматизация приводит слова к их нормальной форме (лемме).
"""

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from utils.nltk_utils import get_wordnet_pos


class LemmaTokenizer:
    """
    Кастомный токенизатор для CountVectorizer с лемматизацией.
    Приводит слова к их базовой форме с учетом части речи.
    """

    def __init__(self):
        """Инициализирует лемматизатор WordNet."""
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        """
        Токенизирует и лемматизирует документ.

        Args:
            doc (str): Входной текст

        Returns:
            list: Список лемматизированных токенов
        """
        # Токенизируем документ
        tokens = word_tokenize(doc)

        # Определяем части речи для каждого токена
        words_and_tags = nltk.pos_tag(tokens)

        # Лемматизируем каждый токен с учетом части речи
        lemmas = [
            self.wnl.lemmatize(word, pos=get_wordnet_pos(tag))
            for word, tag in words_and_tags
        ]

        return lemmas


def create_lemmatization_vectorizer():
    """
    Создает CountVectorizer с лемматизацией токенов.

    Returns:
        CountVectorizer: Векторизатор с лемматизацией
    """
    vectorizer = CountVectorizer(
        tokenizer=LemmaTokenizer(),  # Используем кастомный токенизатор
        lowercase=True,
        max_df=0.95,
        min_df=2,
        max_features=None,
        # Отключаем стандартный паттерн
        # при использовании кастомного токенизатора
        token_pattern=None
    )

    return vectorizer


def get_vectorizer_info(vectorizer, X_train):
    """
    Возвращает информацию о векторизаторе с лемматизацией.

    Args:
        vectorizer: Обученный векторизатор
        X_train: Преобразованные тренировочные данные

    Returns:
        dict: Словарь с информацией о векторизаторе
    """
    vocabulary = vectorizer.vocabulary_

    # Рассчитываем плотность матрицы
    if hasattr(X_train, 'toarray'):
        density = (X_train != 0).sum() / np.prod(X_train.shape)
    else:
        density = np.count_nonzero(X_train) / X_train.size

    return {
        'name': 'CountVectorizer с лемматизацией',
        'vocabulary_size': len(vocabulary),
        'density_percent': density * 100,
        'shape': X_train.shape,
        'tokenizer_type': 'LemmaTokenizer',
        'example_features': list(vocabulary.keys())[:10]
    }


if __name__ == "__main__":
    # Пример использования
    print("Модуль CountVectorizer с лемматизацией")
    tokenizer = LemmaTokenizer()
    sample_text = "The cats are running and jumping around."
    print(f"Пример лемматизации: '{sample_text}'")
    print(f"Результат: {tokenizer(sample_text)}")
