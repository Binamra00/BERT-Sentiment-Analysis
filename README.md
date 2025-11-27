# Understanding the Impact of BERT Layer Freezing on Accuracy and Speed of Sentiment Analysis

This repository has two tightly connected components:

1. **IMDb Data Preprocessing Pipeline** – a reusable, modular pipeline for preparing the Stanford Large Movie Review (IMDb) dataset.  
2. **Sentiment Analysis Experiments** – code to study how freezing different numbers of BERT layers affects accuracy, training speed, and downstream tasks.

The typical workflow is:

**Raw IMDb data → Preprocessing Pipeline → Clean CSV files → Sentiment Analysis & BERT experiments**

---

## 1. IMDb Movie Review Preprocessing Pipeline

A robust, modular Python pipeline for cleaning, structuring, and preparing the **Stanford Large Movie Review (IMDb) Dataset** for sentiment analysis.  
It is built with software engineering best practices, including the **Strategy** and **Facade** design patterns, to keep the codebase maintainable and extensible.

### 1.1 Features

- **Modular architecture**  
  Each step (loading, cleaning, partitioning) is handled by its own single-responsibility module.

- **Flexible cleaning pipeline (Strategy Pattern)**  
  Text cleaning steps are encapsulated as strategies, making it easy to add, remove, or modify operations.

- **Simple execution (Facade Pattern)**  
  A single entry point (`main.py`) and a facade (`pipeline.py`) hide internal complexity and orchestrate the workflow.

- **Data-driven design**  
  The pipeline preserves critical linguistic features for sentiment analysis  
  (e.g., **no stop word removal**, **no stemming**).

- **Google Colab friendly**  
  Includes instructions and path handling tailored for Google Colab + Google Drive.

### 1.2 Directory Structure

The preprocessing code is organized into two main packages (`preprocess` and `resources`) and a main script:

```text
Data Preprocessor/
|
|--- preprocess/
|    |--- __init__.py        
|    |--- data_loader.py      # Load and structure raw IMDb data
|    |--- partitioner.py      # Split data into train/val/test sets
|    |--- pipeline.py         # Facade that orchestrates the full workflow
|    |--- text_cleaner.py     # Implements the text cleaning Strategy Pattern
|
|--- resources/
|    |--- __init__.py         
|    |--- contractions.py     # Static dictionary of English contractions
|
|--- main.py                  # Main script to run the entire preprocessing pipeline
```
### 1.3 Prerequisites

The pipeline runs in a Python 3 environment and uses:

- Python 3.x  
- `pandas`  
- `tqdm`  

These are typically available by default in **Google Colab**, so no extra installation is usually required.

---

### 1.4 Setup (Google Drive / Colab)

#### 1.4.1 Get the repo code

1. Download or clone the **BERT-Sentiment-Analysis** repository.
2. Upload it to your Google Drive.
3. The `Data Preprocessor/` folder lives inside this repository.

#### 1.4.2 Download the IMDb dataset

1. Download the **Large Movie Review Dataset** (`aclImdb_v1.tar.gz`) from its official source (e.g., Stanford AI Lab).

#### 1.4.3 Unzip the dataset

1. Extract the archive; you should get a folder named `aclImdb`.

#### 1.4.4 Place the data in the repo

1. Move the `aclImdb` folder into your `BERT-Sentiment-Analysis` (or equivalent) folder in Google Drive.

---

### 1.5 Running the Preprocessing Pipeline

1. Open `main.py` inside the `Data Preprocessor/` folder in **Google Colab**.
2. If prompted, **mount your Google Drive** and authorize access.
3. Run all cells (e.g., `Runtime > Run all` in Colab).

The script will:

- Locate the raw `aclImdb` data  
- Run the complete preprocessing pipeline  
- Save clean, structured datasets to a new output directory  

---

### 1.6 Preprocessing Output

After successful execution, the pipeline will create:

```text
Data Preprocessor/processed_data/
```
Inside, you will find three CSV files, ready for model training:

- `train_clean.csv` – training set (~22,500 reviews)
- `val_clean.csv` – validation set for hyperparameter tuning (~2,500 reviews)
- `test_clean.csv` – held-out test set (~25,000 reviews)

Each CSV contains:

- `review_text`
- `sentiment_label`
- `star_rating`

These files are the inputs used by the **Sentiment Analysis** component described next.

---

## 2. Sentiment Analysis & BERT Layer Freezing Experiments

This part of the project investigates the trade-off between **model accuracy** and **training efficiency** in sentiment analysis.

It uses the preprocessed IMDb reviews to:

- Compare a **CNN baseline** to various **BERT fine-tuning** strategies  
- Evaluate the impact of **freezing 0, 4, 8, or 11 of BERT’s 12 encoder layers**  
- Study **probability calibration** and downstream **ordinal prediction (1–5 star ratings)**  

---

### 2.1 Experimental Phases

The analysis is divided into three sequential phases:

#### Phase 1 – Model Training

Quantify how freezing different numbers of BERT layers affects:

- **Accuracy** (F1-score)
- **Training speed** (time per epoch)

Models include:

- Fully fine-tuned BERT (`bert_full_finetune`)
- BERT with bottom 4, 8, or 11 layers frozen
- CNN baselines (static and non-static embeddings)

---

#### Phase 2 – Probability Calibration

Study how “honest” the best model’s probability scores are and correct overconfidence using **Isotonic Regression**.

This phase:

- Measures calibration with **Expected Calibration Error (ECE)**
- Applies post-hoc calibration to outputs of the best model
- Produces **reliability diagrams** and **calibration plots**

---

#### Phase 3 – Ordinal Mapping (1–5 Star Ratings)

Use the calibrated probabilities in a downstream ordinal prediction task:

- Map sentiment probabilities to **1–5 star ratings**
- Evaluate performance using **RMSE** and **MAE**
- Compare **calibrated vs. uncalibrated** predictions

---

### 2.2 Key Findings (Summary)

#### Efficiency vs. Accuracy

- Fully fine-tuned BERT (`bert_full_finetune`) achieved the highest F1-score:  
  **93.96% (average)**.
- Freezing the bottom 8 layers (`bert_frozen_8`) retained about **99.5%** of this performance  
  (**93.51% F1 on average**) while **reducing training time per epoch by ~40%**.

#### Probability Calibration

- The best model (`bert_full_finetune_seed123`) was **highly overconfident**.

Post-hoc calibration with **Isotonic Regression**:

- Reduced **ECE** by **64.23%**
- Yielded significantly more reliable confidence scores

#### Downstream Ordinal Task

Using calibrated probabilities instead of raw ones improved star-rating predictions:

- **RMSE** decreased by **3.77%**
- **MAE** decreased by **2.21%**

---

### 2.3 Project Structure (Sentiment Analysis)

```text
Sentiment Analysis
├── configs/
│   ├── cnn_baseline.yaml
│   ├── cnn_non_static.yaml
│   ├── bert_full_finetune.yaml
│   ├── bert_frozen_4.yaml
│   ├── bert_frozen_8.yaml
│   └── bert_frozen_11.yaml
│
├── data/
│   ├── processed/
│   │   ├── train_clean.csv
│   │   ├── val_clean.csv
│   │   └── test_clean.csv
│   └── embeddings/
│       └── glove.6B.300d.txt
│
├── outputs/
│   ├── models/           # (Generated) Trained .pt model checkpoints
│   ├── metrics/          # (Generated) Experiment .json logs
│   ├── probabilities/    # (Generated) .npz probability/label files
│   └── plots/            # (Generated) .png analysis plots
│
├── src/
│   ├── models/
│   │   ├── bert.py           # BERT model with layer freezing options
│   │   └── kim_cnn.py        # Kim (2014) CNN implementation
│   ├── engine/
│   │   ├── trainer.py        # train_epoch function
│   │   └── evaluator.py      # evaluate function (F1, Acc, etc.)
│   ├── utils/
│   │   └── cnn_utils.py      # GloVe/vocabulary helpers for CNN
│   ├── data/
│   │   └── dataset.py        # PyTorch SentimentDataset
│   └── postprocessing/
│       ├── calibrate.py      # Phase 2: ECE / calibration analysis
│       └── ordinal.py        # Phase 3: 1–5 star rating analysis
│
├── notebooks/                # Google Colab notebooks for the experiments
│   └── ...                   # (Added) Colab-ready experiment notebooks
│
├── run_experiment.py               # Phase 1: Train and evaluate models
├── run_probability_generation.py   # Phase 2/3: Generate probabilities
└── Readme.md
```
### 2.4 Phase 4 – Statistical Model Validation

To ensure that the performance gap between the baseline CNN (`cnn_baseline`) and the best BERT model (`bert_full_finetune_seed123`) is not due to chance, we ran a dedicated statistical validation phase:

- **McNemar’s Test (paired classification test)**  
  - Built a 2×2 contingency table over the full IMDb test set (25,000 reviews).  
  - CNN-correct / BERT-wrong: 588 cases  
  - CNN-wrong / BERT-correct: 2,327 cases  
  - Test statistic: χ² ≈ 1036.24, p ≈ 2.38 × 10⁻²²⁷  
  - This overwhelmingly rejects the null hypothesis of equal error rates and confirms that BERT’s improvements over the CNN baseline are statistically significant.

- **Bootstrap accuracy confidence interval (BERT)**  
  - Resampled the 25,000-example test set 1,000 times with replacement and recomputed accuracy on each sample.  
  - Mean accuracy: **94.25%**  
  - 95% confidence interval: **[93.96%, 94.53%]** (±0.29%).  
  - This narrow interval indicates that the reported accuracy of the BERT model is stable and robust.

- **Seed-variance sanity check (earlier phases)**  
  - Layer-freezing configurations were also evaluated across multiple random seeds to verify that trends (e.g., the efficiency gains of freezing 8 layers) are consistent and not tied to a single initialization.

Taken together, these checks show that the fine-tuned BERT model delivers a **real, statistically significant** improvement over the CNN baselines on IMDb, rather than a result of random variation in the test set.

## 3. How to Run the Analysis

The **Sentiment Analysis** part is meant to run **after preprocessing**, in the following order:

1. Prepare data & embeddings  
2. Phase 1 – Train models  
3. Phase 2 – Run calibration  
4. Phase 3 – Run ordinal mapping  

---

### 3.1 Prerequisites

#### Preprocessed data

Make sure the CSVs produced by the preprocessing pipeline are available in:

```text
Sentiment Analysis/data/processed/
    ├── train_clean.csv
    ├── val_clean.csv
    └── test_clean.csv
```
#### Word embeddings

Download **GloVe 6B 300d** embeddings (`glove.6B.300d.txt`) and place them in:

```text
Sentiment Analysis/data/embeddings/glove.6B.300d.txt
```
#### Python dependencies

Install the required libraries:

```bash
pip install torch transformers pandas numpy scikit-learn pyyaml tqdm matplotlib seaborn
```
### 3.2 Phase 1 – Train Models

Use `run_experiment.py` to train models from YAML configurations.  
The script:

- Reads a config file from `configs/`
- Trains the specified model
- Saves:
  - Best model checkpoint (by validation loss) to `outputs/models/`
  - Full metrics log to `outputs/metrics/`

**Examples:**

```bash
# Train fully fine-tuned BERT with seed 42
python run_experiment.py --config configs/bert_full_finetune.yaml --seed 42

# Train BERT with 8 frozen layers and seed 123
python run_experiment.py --config configs/bert_frozen_8.yaml --seed 123

# Train static CNN baseline with seed 2025
python run_experiment.py --config configs/cnn_baseline.yaml --seed 2025

# Train non-static CNN with seed 42
python run_experiment.py --config configs/cnn_non_static.yaml --seed 42
```
### 3.3 Phase 2 – Probability Calibration

Use `calibrate.py` to:

- Load `.npz` probability/label files produced during experiments  
- Fit an **Isotonic Regression** calibrator on validation data  
- Report **ECE** and **Brier scores** before vs. after calibration  

It also saves:

- Reliability diagram  
- Calibration (correction function) plot  

to `outputs/plots/`.

**Example:**

```bash
python src/postprocessing/calibrate.py --run_name "bert_full_finetune_seed123.pt"
```
### 3.4 Phase 3 – Ordinal Mapping (1–5 Stars)

Use `ordinal.py` to evaluate calibrated probabilities in a **star-rating prediction** task.

This script:

- Repeats the calibration step  
- Maps probabilities to **1–5 star ratings**  
- Compares **uncalibrated vs. calibrated** performance (MAE, RMSE, etc.)  
- Saves a final comparison boxplot to `outputs/plots/`  

**Example:**

```bash
python src/postprocessing/ordinal.py --run_name "bert_full_finetune_seed123.pt"
```
