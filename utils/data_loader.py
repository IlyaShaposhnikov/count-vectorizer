import pandas as pd

DEFAULT_DATA_PATH = 'data/bbc_text_cls.csv'


def load_data(data_path=DEFAULT_DATA_PATH):
    """
    Loads the BBC News dataset for text classification.

    Args:
        data_path: Path to the data file

    Returns:
        tuple: (inputs, labels) - texts and labels


    Raises:
        FileNotFoundError: If the data file does not exist
        ValueError: If the file is empty or missing required columns
    """
    print(f"Loading data from {data_path}")

    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {data_path}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"Data file is empty: {data_path}")
    except Exception as e:
        raise ValueError(f"Error reading data file: {e}")

    # Check if DataFrame is empty
    if df.empty:
        raise ValueError(f"Loaded DataFrame is empty: {data_path}")

    # Check required columns
    required_columns = ['text', 'labels']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in data file: {missing_cols}"
        )

    # Check the data structure
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Separate features and labels
    inputs = df['text']
    labels = df['labels']

    # Print class distribution info
    print("\nLabel distribution:")
    label_counts = labels.value_counts()
    for label, count in label_counts.items():
        print(f"  {label}: {count} documents ({count/len(labels)*100:.1f}%)")

    return inputs, labels
