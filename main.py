"""
Основной скрипт для сравнения различных методов CountVectorizer.
Сравнивает 5 подходов к предобработке текста для классификации новостей BBC.
"""

import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import time

# Импортируем наши методы векторизации
from methods.base_vectorizer import (
    create_base_vectorizer,
    get_vectorizer_info as get_base_info,
)
from methods.stopwords_vectorizer import (
    create_stopwords_vectorizer,
    get_vectorizer_info as get_stopwords_info,
)
from methods.lemmatization_vectorizer import (
    create_lemmatization_vectorizer,
    get_vectorizer_info as get_lemmatization_info,
)
from methods.stemming_vectorizer import (
    create_stemming_vectorizer,
    get_vectorizer_info as get_stemming_info,
)
from methods.simple_tokenizer_vectorizer import (
    create_simple_tokenizer_vectorizer,
    get_vectorizer_info as get_simple_info,
)

# Импортируем утилиты NLTK
from utils.nltk_utils import download_nltk_resources


def load_data(data_path='data/bbc_text_cls.csv'):
    """
    Загружает датасет BBC News для классификации текста.

    Args:
        data_path (str): Путь к файлу с данными

    Returns:
        tuple: (inputs, labels) - тексты и метки
    """
    print(f"Загрузка данных из {data_path}")
    df = pd.read_csv(data_path)

    # Проверяем структуру данных
    print(f"Размер датасета: {df.shape}")
    print(f"Колонки: {df.columns.tolist()}")

    # Разделяем на признаки и метки
    inputs = df['text']
    labels = df['labels']

    # Выводим информацию о распределении классов
    print("\nРаспределение меток:")
    label_counts = labels.value_counts()
    for label, count in label_counts.items():
        print(f"  {label}: {count} документов ({count/len(labels)*100:.1f}%)")

    return inputs, labels


def evaluate_method(
        vectorizer_creator, info_getter, inputs_train,
        inputs_test, Ytrain, Ytest, method_name
):
    """
    Оценивает метод векторизации на данных.

    Args:
        vectorizer_creator: Функция создания векторизатора
        info_getter: Функция получения информации о векторизаторе
        inputs_train: Тренировочные тексты
        inputs_test: Тестовые тексты
        Ytrain: Тренировочные метки
        Ytest: Тестовые метки
        method_name: Название метода

    Returns:
        dict: Результаты оценки метода
    """
    print(f"\n{'='*60}")
    print(f"Оценка метода: {method_name}")
    print(f"{'='*60}")

    start_time = time.time()

    # Создаем и обучаем векторизатор
    vectorizer = vectorizer_creator()
    print("Векторизация тренировочных данных...")
    X_train = vectorizer.fit_transform(inputs_train)

    # Преобразуем тестовые данные
    print("Векторизация тестовых данных...")
    X_test = vectorizer.transform(inputs_test)

    # Получаем информацию о векторизаторе
    vectorizer_info = info_getter(vectorizer, X_train)

    # Обучаем модель Naive Bayes
    print("Обучение модели MultinomialNB...")
    model = MultinomialNB()
    model.fit(X_train, Ytrain)

    # Оцениваем модель
    train_score = model.score(X_train, Ytrain)
    test_score = model.score(X_test, Ytest)

    end_time = time.time()
    execution_time = end_time - start_time

    # Собираем результаты
    results = {
        'method_name': method_name,
        'train_accuracy': train_score * 100,
        'test_accuracy': test_score * 100,
        'vocabulary_size': vectorizer_info['vocabulary_size'],
        'density_percent': vectorizer_info.get('density_percent', 0),
        'execution_time': execution_time,
        'vectorizer_info': vectorizer_info
    }

    print(f"  Точность на тренировочных данных: {train_score:.3%}")
    print(f"  Точность на тестовых данных: {test_score:.3%}")
    print(f"  Размер словаря: {vectorizer_info['vocabulary_size']} слов")
    print(f"  Плотность матрицы: {vectorizer_info.get('density_percent', 0):.2f}%")
    print(f"  Время выполнения: {execution_time:.2f} секунд")

    return results


def visualize_results(results_df):
    """
    Визуализирует результаты сравнения методов.

    Args:
        results_df: DataFrame с результатами
    """
    # Настройка стиля графиков
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")

    # Создаем фигуру с несколькими subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Сравнение методов CountVectorizer', fontsize=16, fontweight='bold')

    # 1. Точность на тестовых данных
    ax1 = axes[0, 0]
    bars = ax1.barh(results_df['method_name'], results_df['test_accuracy'])
    ax1.set_xlabel('Точность, %')
    ax1.set_title('Точность на тестовых данных')
    ax1.set_xlim([0, 100])

    # Добавляем значения на столбцы
    for bar, value in zip(bars, results_df['test_accuracy']):
        ax1.text(value + 0.5, bar.get_y() + bar.get_height()/2,
                 f'{value:.1f}%', va='center', fontweight='bold')

    # 2. Размер словаря
    ax2 = axes[0, 1]
    bars = ax2.barh(results_df['method_name'], results_df['vocabulary_size'])
    ax2.set_xlabel('Количество слов')
    ax2.set_title('Размер словаря (количество уникальных слов)')

    # Добавляем значения на столбцы
    for bar, value in zip(bars, results_df['vocabulary_size']):
        ax2.text(value + max(results_df['vocabulary_size'])*0.01,
                 bar.get_y() + bar.get_height()/2,
                 f'{value:,}', va='center', fontweight='bold')

    # 3. Время выполнения
    ax3 = axes[1, 0]
    bars = ax3.barh(results_df['method_name'], results_df['execution_time'])
    ax3.set_xlabel('Время, секунды')
    ax3.set_title('Время выполнения')

    # Добавляем значения на столбцы
    for bar, value in zip(bars, results_df['execution_time']):
        ax3.text(value + 0.05, bar.get_y() + bar.get_height()/2,
                 f'{value:.2f}с', va='center', fontweight='bold')

    # 4. Плотность матрицы (если есть данные)
    if 'density_percent' in results_df.columns:
        ax4 = axes[1, 1]
        bars = ax4.barh(results_df['method_name'], results_df['density_percent'])
        ax4.set_xlabel('Плотность, %')
        ax4.set_title('Плотность матрицы признаков')
        ax4.set_xlim([0, 1])  # Обычно плотность < 1%

        # Добавляем значения на столбцы
        for bar, value in zip(bars, results_df['density_percent']):
            ax4.text(value + 0.01, bar.get_y() + bar.get_height()/2,
                     f'{value:.3f}%', va='center', fontweight='bold')
    else:
        # Если нет данных о плотности, показываем заглушку
        ax4 = axes[1, 1]
        ax4.text(0.5, 0.5, 'Нет данных о плотности',
                 ha='center', va='center', fontsize=12)
        ax4.set_title('Плотность матрицы признаков')
        ax4.set_xticks([])
        ax4.set_yticks([])

    plt.tight_layout()
    plt.savefig('results/comparison_results.png', dpi=300, bbox_inches='tight')
    print("\nГрафики сохранены в 'results/comparison_results.png'")
    plt.show()


def print_detailed_comparison(results):
    """
    Выводит детальное сравнение методов в виде таблицы.

    Args:
        results: Список словарей с результатами
    """
    # Создаем DataFrame для удобного отображения
    df = pd.DataFrame(results)

    # Сортируем по точности на тестовых данных
    df = df.sort_values('test_accuracy', ascending=False)

    print("\n" + "="*80)
    print("ДЕТАЛЬНОЕ СРАВНЕНИЕ МЕТОДОВ COUNTVECTORIZER")
    print("="*80)

    # Создаем таблицу для отображения
    table_data = []
    for _, row in df.iterrows():
        table_data.append([
            row['method_name'],
            f"{row['train_accuracy']:.2f}%",
            f"{row['test_accuracy']:.2f}%",
            f"{row['vocabulary_size']:,}",
            f"{row.get('density_percent', 0):.4f}%" if 'density_percent' in row else "N/A",
            f"{row['execution_time']:.2f}с"
        ])

    headers = ["Метод", "Train Acc", "Test Acc", "Словарь", "Плотность", "Время"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Выводим выводы
    print("\n" + "="*80)
    print("ВЫВОДЫ:")
    print("="*80)

    best_method = df.iloc[0]
    print(f"Лучший метод: {best_method['method_name']} с точностью {best_method['test_accuracy']:.2f}%")

    print("\nНаблюдения:")
    print("1. Размер словаря влияет на время выполнения и качество модели")
    print("2. Удаление стоп-слов обычно уменьшает размерность")
    print("3. Лемматизация и стемминг могут улучшить или ухудшить качество")
    print("4. Простой токенизатор самый быстрый, но может быть менее точным")

    # Сохраняем результаты в CSV
    results_dir = 'results'
    import os
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    df.to_csv(f'{results_dir}/detailed_results.csv', index=False)
    print(f"\nДетальные результаты сохранены в '{results_dir}/detailed_results.csv'")


def main():
    """
    Основная функция для сравнения методов CountVectorizer.
    """
    print("="*80)
    print("СРАВНЕНИЕ МЕТОДОВ ПРЕДОБРАБОТКИ ТЕКСТА ДЛЯ COUNTVECTORIZER")
    print("="*80)

    # Загружаем ресурсы NLTK
    print("\nЗагрузка ресурсов NLTK...")
    download_nltk_resources()

    # Загружаем данные
    inputs, labels = load_data()

    # Разделяем на тренировочную и тестовую выборки
    print("\nРазделение данных на тренировочную и тестовую выборки...")
    inputs_train, inputs_test, Ytrain, Ytest = train_test_split(
        inputs, labels, test_size=0.25, random_state=123, stratify=labels
    )

    print(f"  Тренировочная выборка: {len(inputs_train)} документов")
    print(f"  Тестовая выборка: {len(inputs_test)} документов")

    # Определяем методы для сравнения
    methods = [
        {
            'name': 'Базовый',
            'vectorizer_creator': create_base_vectorizer,
            'info_getter': get_base_info
        },
        {
            'name': 'Со стоп-словами',
            'vectorizer_creator': create_stopwords_vectorizer,
            'info_getter': get_stopwords_info
        },
        {
            'name': 'С лемматизацией',
            'vectorizer_creator': create_lemmatization_vectorizer,
            'info_getter': get_lemmatization_info
        },
        {
            'name': 'Со стеммингом',
            'vectorizer_creator': create_stemming_vectorizer,
            'info_getter': get_stemming_info
        },
        {
            'name': 'С простым токенизатором',
            'vectorizer_creator': create_simple_tokenizer_vectorizer,
            'info_getter': get_simple_info
        }
    ]

    # Оцениваем каждый метод
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

    # Выводим детальное сравнение
    print_detailed_comparison(all_results)

    # Визуализируем результаты
    try:
        visualize_results(pd.DataFrame(all_results))
    except Exception as e:
        print(f"\nОшибка при визуализации: {e}")
        print("Продолжение без визуализации...")

    print("\n" + "="*80)
    print("СРАВНЕНИЕ ЗАВЕРШЕНО!")
    print("="*80)


if __name__ == "__main__":
    main()
