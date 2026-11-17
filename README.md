# Sentiment Analysis on Movie Reviews (LSTM Pipeline)

A supervised Deep Learning pipeline built with **spaCy**, **TensorFlow/Keras**, and **Hugging Face Datasets** to perform binary sentiment classification (Positive vs. Negative) on the Stanford Large Movie Review Dataset (IMDB).

## Repository Structure

* **`NLP_project.ipynb`** — Complete end-to-end Jupyter Notebook containing dataset loading, spaCy text normalization, tensor conversion, progressive training (2,000-review subset and full 25,000-review corpus), and final test evaluation outputs.
* **`data_collection.py`** — Initial standalone script for downloading and extracting the raw `aclImdb_v1.tar.gz` archive locally (superseded in the main notebook by Hugging Face `datasets` streaming to avoid local I/O bottlenecks).
* **`data_preprocessing.py`** — Modular spaCy preprocessing script implementing stop-word removal, punctuation/whitespace filtering, and lemmatization via `nlp.pipe`.

## Pipeline & Architecture

1. **Data Ingestion & Shuffling:** Loads the 50,000 balanced reviews (`stanfordnlp/imdb`: 25,000 train / 25,000 test) and shuffles the training set (`random_state=42`) prior to validation splitting to prevent class-order bias.
2. **Text Normalization (`spaCy`):** Uses `en_core_web_sm` with `nlp.pipe(batch_size=500)` to remove stop words, punctuation, and whitespace, converting remaining tokens to lowercase lemmas.
3. **Tensor Conversion (`TensorFlow`):** Explicitly casts variable-length text arrays into `tf.string` tensors (`tf.convert_to_tensor`) to ensure memory compatibility with Keras 3 validation splits.
4. **Neural Network (`Keras Sequential`):**
   * `TextVectorization` (`max_tokens=10000`, `output_sequence_length=200`)
   * `Embedding` (`input_dim=10000`, `output_dim=64`)
   * `LSTM` (`64` units)
   * `Dense` (`1` unit, `sigmoid` activation)

## Test Dataset Performance (25,000 Unseen Reviews)

| Metric | Negative Class | Positive Class | Overall / Macro Avg |
| :--- | :---: | :---: | :---: |
| **Precision** | 0.84 | 0.85 | 0.84 |
| **Recall** | 0.85 | 0.84 | 0.84 |
| **F1-Score** | 0.84 | 0.84 | 0.84 |
| **Accuracy** | — | — | **84.0%** |
| **ROC AUC Score** | — | — | **0.9167** |

**Confusion Matrix:**
```text
[[10636  1864]   # True Negatives: 10,636 | False Positives: 1,864
 [ 2040 10460]]  # False Negatives: 2,040 | True Positives: 10,460
```

## Requirements & Setup

* **Python 3.12** (tested on Python 3.12.10)

```bash
pip install datasets spacy tensorflow scikit-learn
python -m spacy download en_core_web_sm
