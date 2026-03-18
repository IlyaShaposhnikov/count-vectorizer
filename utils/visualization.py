import os
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_PATH = 'results/comparison_results.png'


def visualize_results(results_df, save_path=OUTPUT_PATH):
    """
    Visualizes the comparison results of the methods.

    Args:
        results_df: DataFrame containing the results
        save_path: Path where the chart image will be saved
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Set plot style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('CountVectorizer Methods', fontsize=16, fontweight='bold')

    # 1. Test accuracy
    ax1 = axes[0, 0]
    bars = ax1.barh(results_df['method_name'], results_df['test_accuracy'])
    ax1.set_xlabel('Accuracy, %')
    ax1.set_title('Test Accuracy')
    ax1.set_xlim([0, 100])

    # Add values to the bars
    for bar, value in zip(bars, results_df['test_accuracy']):
        ax1.text(value + 0.5, bar.get_y() + bar.get_height()/2,
                 f'{value:.1f}%', va='center', fontweight='bold')

    # 2. Vocabulary size
    ax2 = axes[0, 1]
    bars = ax2.barh(results_df['method_name'], results_df['vocabulary_size'])
    ax2.set_xlabel('Number of words')
    ax2.set_title('Vocabulary Size (unique words)')

    # Add values to the bars
    max_vocabulary_size = results_df['vocabulary_size'].max()
    for bar, value in zip(bars, results_df['vocabulary_size']):
        ax2.text(value + max_vocabulary_size * 0.01,
                 bar.get_y() + bar.get_height()/2,
                 f'{value:,}', va='center', fontweight='bold')

    # 3. Execution time
    ax3 = axes[1, 0]
    bars = ax3.barh(results_df['method_name'], results_df['execution_time'])
    ax3.set_xlabel('Time, seconds')
    ax3.set_title('Execution Time')

    # Add values to the bars
    for bar, value in zip(bars, results_df['execution_time']):
        ax3.text(value + 0.05, bar.get_y() + bar.get_height()/2,
                 f'{value:.2f}s', va='center', fontweight='bold')

    # 4. Matrix density (if data exists)
    density_data = results_df.get('density_percent')
    if density_data is not None:
        ax4 = axes[1, 1]
        bars = ax4.barh(
            results_df['method_name'], results_df['density_percent']
        )
        ax4.set_xlabel('Density, %')
        ax4.set_title('Feature Matrix Density')
        max_density = results_df['density_percent'].max()
        ax4.set_xlim([0, max_density * 1.1])

        # Add values to the bars
        for bar, value in zip(bars, results_df['density_percent']):
            ax4.text(value + 0.01, bar.get_y() + bar.get_height()/2,
                     f'{value:.3f}%', va='center', fontweight='bold')
    else:
        # Show placeholder if no density data
        ax4 = axes[1, 1]
        ax4.text(0.5, 0.5, 'No density data',
                 ha='center', va='center', fontsize=12)
        ax4.set_title('Feature Matrix Density')
        ax4.set_xticks([])
        ax4.set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nCharts saved to '{save_path}'")
    plt.show()
