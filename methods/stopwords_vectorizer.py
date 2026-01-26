"""
CountVectorizer с удалением стоп-слов (общеупотребительных слов).
Удаление стоп-слов помогает уменьшить размерность и сфокусироваться
на значимых словах.
"""

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


def create_stopwords_vectorizer():
    """
    Создает CountVectorizer с удалением английских стоп-слов.

    Returns:
        CountVectorizer: Векторизатор с удалением стоп-слов
    """
    # Используем встроенный список английских стоп-слов из sklearn
    vectorizer = CountVectorizer(
        stop_words='english',  # Удаляем стандартные английские стоп-слов
        lowercase=True,
        token_pattern=r'(?u)\b\w\w+\b',
        max_df=0.95,  # Игнорируем слова, которые встречаются в >95% документов
        min_df=2,     # Игнорируем слова, которые встречаются менее 2 раз
        max_features=None
    )

    return vectorizer


def get_vectorizer_info(vectorizer, X_train):
    """
    Возвращает информацию о векторизаторе с удалением стоп-слов.

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

    # Получаем список удаленных стоп-слов
    stop_words = vectorizer.get_stop_words()

    return {
        'name': 'CountVectorizer с удалением стоп-слов',
        'vocabulary_size': len(vocabulary),
        'density_percent': density * 100,
        'shape': X_train.shape,
        'removed_stopwords_count': len(stop_words) if stop_words else 0,
        'example_features': list(vocabulary.keys())[:10]
    }


if __name__ == "__main__":
    print("Модуль CountVectorizer с удалением стоп-слов")
