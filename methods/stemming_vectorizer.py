"""
CountVectorizer со стеммингом токенов.
Стемминг обрезает окончания слов, оставляя основу (стем).
"""

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from nltk import word_tokenize
from nltk.stem import PorterStemmer


class StemTokenizer:
    """
    Кастомный токенизатор для CountVectorizer со стеммингом.
    Использует алгоритм Porter Stemmer для нахождения основы слова.
    """

    def __init__(self):
        """Инициализирует стеммер Porter."""
        self.porter = PorterStemmer()

    def __call__(self, doc):
        """
        Токенизирует и стеммирует документ.

        Args:
            doc (str): Входной текст

        Returns:
            list: Список стеммированных токенов
        """
        # Токенизируем документ
        tokens = word_tokenize(doc)

        # Применяем стемминг к каждому токену
        stems = [self.porter.stem(t) for t in tokens]

        return stems


def create_stemming_vectorizer():
    """
    Создает CountVectorizer со стеммингом токенов.

    Returns:
        CountVectorizer: Векторизатор со стеммингом
    """
    vectorizer = CountVectorizer(
        tokenizer=StemTokenizer(),  # Используем кастомный токенизатор
        lowercase=True,
        max_df=0.95,
        min_df=2,
        max_features=None,
        token_pattern=None
    )

    return vectorizer


def get_vectorizer_info(vectorizer, X_train):
    """
    Возвращает информацию о векторизаторе со стеммингом.

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
        'name': 'CountVectorizer со стеммингом',
        'vocabulary_size': len(vocabulary),
        'density_percent': density * 100,
        'shape': X_train.shape,
        'tokenizer_type': 'StemTokenizer',
        'example_features': list(vocabulary.keys())[:10]
    }


if __name__ == "__main__":
    # Пример использования
    print("Модуль CountVectorizer со стеммингом")
    tokenizer = StemTokenizer()
    sample_text = "running runners ran"
    print(f"Пример стемминга: '{sample_text}'")
    print(f"Результат: {tokenizer(sample_text)}")
