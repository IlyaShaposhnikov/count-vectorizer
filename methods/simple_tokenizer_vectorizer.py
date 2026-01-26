"""
CountVectorizer с простым токенизатором на основе split().
Это самый быстрый, но наименее точный метод токенизации.
"""

from sklearn.feature_extraction.text import CountVectorizer


def simple_tokenizer(s):
    """
    Простой токенизатор на основе split().
    Разделяет строку по пробелам без учета пунктуации.

    Args:
        s (str): Входная строка

    Returns:
        list: Список токенов
    """
    return s.split()


def create_simple_tokenizer_vectorizer():
    """
    Создает CountVectorizer с простым токенизатором.

    Returns:
        CountVectorizer: Векторизатор с простым токенизатором
    """
    vectorizer = CountVectorizer(
        tokenizer=simple_tokenizer,  # Используем простой токенизатор
        lowercase=True,
        max_df=0.95,
        min_df=2,
        max_features=None,
        token_pattern=None
    )

    return vectorizer


def get_vectorizer_info(vectorizer, X_train):
    """
    Возвращает информацию о векторизаторе с простым токенизатором.

    Args:
        vectorizer: Обученный векторизатор
        X_train: Преобразованные тренировочные данные

    Returns:
        dict: Словарь с информацией о векторизаторе
    """
    vocabulary = vectorizer.vocabulary_

    # Рассчитываем плотность матрицы
    import numpy as np
    if hasattr(X_train, 'toarray'):
        density = (X_train != 0).sum() / np.prod(X_train.shape)
    else:
        density = np.count_nonzero(X_train) / X_train.size

    return {
        'name': 'CountVectorizer с простым токенизатором',
        'vocabulary_size': len(vocabulary),
        'density_percent': density * 100,
        'shape': X_train.shape,
        'tokenizer_type': 'simple_tokenizer (split)',
        'example_features': list(vocabulary.keys())[:10],
        'note': (
            'Внимание: простой split() не обрабатывает пунктуацию правильно'
        )
    }


if __name__ == "__main__":
    # Пример использования
    print("Модуль CountVectorizer с простым токенизатором")
    sample_text = "I'm going to the park. It's beautiful!"
    print(f"Пример токенизации: '{sample_text}'")
    print(f"Результат: {simple_tokenizer(sample_text)}")
    print("Примечание: 'I'm' не разделяется на 'I' и 'am'")
