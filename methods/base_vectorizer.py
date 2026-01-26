"""
Базовый метод CountVectorizer без дополнительной обработки.
Использует стандартные настройки sklearn CountVectorizer.
"""

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


def create_base_vectorizer():
    """
    Создает и возвращает базовый CountVectorizer.

    Returns:
        CountVectorizer: Базовый векторизатор с настройками по умолчанию
    """
    # Используем стандартные параметры CountVectorizer
    vectorizer = CountVectorizer(
        lowercase=True,      # Приводим текст к нижнему регистру
        token_pattern=r'(?u)\b\w\w+\b',  # Стандартный паттерн для токенов
        max_df=1.0,          # Максимальная частота слова в документах
        min_df=1,           # Минимальная частота слова в документах
        max_features=None   # Без ограничения количества фич
    )

    return vectorizer


def get_vectorizer_info(vectorizer, X_train):
    """
    Возвращает информацию о векторизаторе и преобразованных данных.

    Args:
        vectorizer: Обученный векторизатор
        X_train: Преобразованные тренировочные данные

    Returns:
        dict: Словарь с информацией о векторизаторе
    """
    # Получаем словарь (слово -> индекс)
    vocabulary = vectorizer.vocabulary_

    # Рассчитываем плотность матрицы (процент ненулевых элементов)
    if hasattr(X_train, 'toarray'):
        # Если это разреженная матрица
        density = (X_train != 0).sum() / np.prod(X_train.shape)
    else:
        # Если это плотная матрица
        density = np.count_nonzero(X_train) / X_train.size

    return {
        'name': 'Базовый CountVectorizer',
        'vocabulary_size': len(vocabulary),
        'density_percent': density * 100,
        'shape': X_train.shape,
        # Пример первых 10 фич
        'example_features': list(vocabulary.keys())[:10]
    }


if __name__ == "__main__":
    # Пример использования
    print("Базовый модуль CountVectorizer")
    print("Использование: импортируйте create_base_vectorizer() в main.py")
