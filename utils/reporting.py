import os
import pandas as pd
from tabulate import tabulate

DEFAULT_CSV_PATH = 'results/detailed_results.csv'


def print_detailed_comparison(results, csv_path=DEFAULT_CSV_PATH):
    """
    Prints a detailed comparison of the methods as a table.

    Args:
        results: List of dictionaries with results
        csv_path: Path where the CSV report will be saved
    """
    # Create DataFrame for display
    df = pd.DataFrame(results)

    # Sort by test accuracy
    df = df.sort_values('test_accuracy', ascending=False)

    print("\n" + "="*80)
    print("DETAILED COUNTVECTORIZER METHODS COMPARISON")
    print("="*80)

    # Prepare table data
    table_data = []
    for _, row in df.iterrows():
        table_data.append([
            row['method_name'],
            f"{row['train_accuracy']:.2f}%",
            f"{row['test_accuracy']:.2f}%",
            f"{row['vocabulary_size']:,}",
            (
                f"{row.get('density_percent', 0):.4f}%"
                if 'density_percent' in row
                else "N/A"
            ),
            f"{row['execution_time']:.2f}s"
        ])

    headers = [
        "Method", "Train Acc", "Test Acc", "Vocab", "Density", "Time"
    ]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY:")
    print("="*80)

    best_method = df.iloc[0]
    print(
        f"Best method: {best_method['method_name']} "
        f"with accuracy {best_method['test_accuracy']:.2f}%"
    )

    # Ensure directory exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Save results to CSV
    df.to_csv(csv_path, index=False)
    print(f"\nDetailed results saved to '{csv_path}'")
