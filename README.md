**[Russian Version / На русском](README.ru.md)**

# CountVectorizer Comparison Project

A project for comparing various text preprocessing methods when using CountVectorizer in machine learning.

## Project Description

This project compares 5 different approaches to text vectorization using `CountVectorizer` from the scikit-learn library:
1.  **Base Method** - Standard CountVectorizer without additional processing
2.  **With Stop Words Removal** - Removal of common words (e.g., "the", "and", "is")
3.  **With Lemmatization** - Reducing words to their normal form (lemma)
4.  **With Stemming** - Trimming word endings to their root (stem)
5.  **With Simple Tokenizer** - Fast splitting by spaces without punctuation handling

The project uses the BBC news dataset for classification and compares the effectiveness of each method across several metrics.

## Quick Start

### Installation

1. Clone the repository:

```bash
git clone https://github.com/IlyaShaposhnikov/count-vectorizer.git
cd count-vectorizer
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3.  Download the dataset and place it into the `data` folder:

Download the dataset on [Kaggle](https://www.kaggle.com/datasets/abdulraffayali/bbc-text-cls)

### Usage
Run the main comparison script:

```bash
python main.py
```

The script will automatically:
1. Download necessary NLTK resources (_On the first run, this may take a few minutes depending on your internet connection speed. Subsequent runs will be instantaneous._) 
2. Load and prepare the data
3. Train models with different vectorization methods
4. Print the comparison results
5. Generate visualizations

## Project Structure
```text
count-vectorizer/
├── data/                               # Data folder
│   └── bbc_text_cls.csv                # BBC News Dataset
├── methods/                            # Implementations of various vectorization methods
│   ├── base_vectorizer.py              # Base method
│   ├── stopwords_vectorizer.py         # With stop words removal
│   ├── lemmatization_vectorizer.py     # With lemmatization
│   ├── stemming_vectorizer.py          # With stemming
│   └── simple_tokenizer_vectorizer.py  # With simple tokenizer
├── utils/                              # Utility scripts
│   └── nltk_utils.py                   # Utilities for working with NLTK
├── results/                            # Folder for saving results
├── main.py                             # Main comparison script
├── requirements.txt                    # Project dependencies
├── README.md                           # Project documentation (English)
└── README.ru.md                        # Project documentation (Russian)
```

## Comparison Metrics
Each method is evaluated based on the following metrics:
* Training Accuracy - Model accuracy on the training data
* Test Accuracy - Model accuracy on new data (primary metric)
* Vocabulary Size - Number of unique words after processing
* Matrix Density - Percentage of non-zero elements in the feature matrix
* Execution Time - Total time for training and evaluation of the method

## Technical Details
### Libraries Used
* numpy - Working with numerical arrays
* pandas - Data processing and analysis
* scikit-learn - Machine learning and CountVectorizer
* nltk - Natural language processing
* matplotlib/seaborn - Result visualization
* tabulate - Human-readable table display

### Dataset
The project uses the BBC News dataset, containing 2225 documents across 5 categories:
* business (510 documents)
* entertainment (386 documents)
* politics (417 documents)
* sport (511 documents)
* tech (401 document)

### Classification Algorithm
Multinomial Naive Bayes is used for all methods - an algorithm well-suited for text classification.

## Result Output
After running `main.py`, you will see:
1. Progress of each method execution
2. A detailed comparison table
3. Graphs with result visualizations
4. Saved results: `detailed_results.csv` table and `comparison_results.png` graph in the `results` folder

## Key Findings
The comparison demonstrates the following trends:
1.  **Stop-word Removal:** Generally reduces dimensionality (vocabulary size) and often improves classification accuracy compared to the base method.
2.  **Lemmatization vs. Stemming:** Lemmatization usually achieves higher accuracy than stemming (e.g., ~97.3% vs ~96.8% in typical runs), but at a significant computational cost, resulting in much longer execution times.
3.  **Speed vs. Accuracy:** Simpler methods (like removing stop-words or using a simple tokenizer) are generally much faster than complex ones (like lemmatization or stemming). The stop-words removal method often provides a good balance, frequently achieving high accuracy while remaining relatively fast. However, the absolute fastest method might vary slightly between runs depending on the system.
4.  **Vocabulary Size & Density:** More aggressive text processing (like stemming or lemmatization) tends to reduce vocabulary size compared to simpler methods. This often leads to a *higher* feature matrix density (as the same concepts are represented by fewer unique terms), although the density remains quite low overall (typically well below 2%).


## Author

Ilya Shaposhnikov | [E-mail](mailto:ilia.a.shaposhnikov@gmail.com) | [LinkedIn](https://linkedin.com/in/iliashaposhnikov)

**[Russian Version / На русском](README.ru.md)**