# Understanding the Impact of BERT Layer Freezing on Accuracy and Speed of Sentiment Analysis

## **IMDb Movie Review Preprocessing Pipeline**
**This project contains a robust, modular, and reusable Python pipeline for cleaning, structuring, and preparing the Stanford Large Movie Review (IMDb) Dataset for sentiment analysis tasks. The entire pipeline is built with a focus on software engineering best practices, including the use of the Strategy and Facade design patterns to ensure the code is maintainable and extensible.**

## Features
Modular Architecture: Each step of the process (loading, cleaning, partitioning) is handled by a separate, single-responsibility module.

- Flexible Cleaning Pipeline: Utilizes the Strategy Pattern to allow for easy addition, removal, or modification of text cleaning steps.

- Simple Execution: A single entry point (main.py) and a Facade Pattern (pipeline.py) hide the system's complexity, making it easy to run the entire workflow.

- Data-Driven: The pipeline is designed to preserve critical linguistic features for sentiment analysis (e.g., no stop word removal, no stemming).

- Google Colab Ready: Includes instructions and path management tailored for a Google Colab and Google Drive environment.

## Directory Structure
The project is organized into two main Python packages (preprocess and resources) and a main execution script.

```
Data Preprocessor/
|
|--- preprocess/
|    |--- init.py        
|    |--- data_loader.py      # Module for loading and structuring raw data
|    |--- partitioner.py      # Module for splitting data into train/val/test sets
|    |--- pipeline.py         # The main Facade that orchestrates the workflow
|    |--- text_cleaner.py     # Implements the text cleaning Strategy Pattern
|
|--- resources/
|    |--- init.py         
|    |--- contractions.py     # Static resource: a dictionary of English contractions
|
|--- main.py                 # The main script to execute the entire pipeline
```
## Prerequisites
This project is designed to run in a Python 3 environment. The primary dependency is the Pandas library for data manipulation and tqdm for progress bars.

Python 3.x

Pandas

tqdm

These are standard in Google Colab environments. No special installation is typically required.

## Setup and Execution
This pipeline is designed to be run from Google Colab, using data stored in your Google Drive.

## Folder Setup
Download or clone the CSE 6363 ML Project repository folder and upload it to your Google Drive. This folder contains the Data Preprocessor codebase.

Download the Large Movie Review Dataset (aclImdb_v1.tar.gz) from its source (e.g., Stanford AI Lab).

Unzip the dataset. You will get a folder named aclImdb.

Place the unzipped aclImdb folder inside the CSE 6363 ML Project folder on your Google Drive.

Your final Google Drive structure should look like this:
```
My Drive/
|
└── CSE 6363 ML Project/
|--- aclImdb/            <-- Raw dataset folder
└── Data Preprocessor/  <-- Your codebase folder
|--- main.py
|--- preprocess/
└── resources/
```
## Running the Pipeline
Open main.py (located inside the Data Preprocessor folder) in Google Colaboratory.

If it's your first time in the session, you may be prompted to mount your Google Drive. Authorize it when asked.

Run all the cells in the notebook (Runtime > Run all).

The script will automatically locate the raw data, execute the entire preprocessing pipeline, and save the output to a new folder.

## Output
Upon successful execution, the pipeline will create a new directory: Data Preprocessor/processed_data/. This directory will contain three structured, clean datasets in CSV format, ready for the next phase of model training:

train_clean.csv: The training set (22,500 reviews).

validation_clean.csv: The validation set for hyperparameter tuning (2,500 reviews).

test_clean.csv: The final, held-out test set for unbiased evaluation (25,000 reviews).

Each CSV file contains the following columns: review_text, sentiment_label, and star_rating.

## **Sentiment Analysis**
**This project analyzes the trade-off between model accuracy and training efficiency in sentiment analysis. It uses the IMDb movie review dataset to compare a CNN baseline against various BERT fine-tuning strategies, with a focus on layer freezing.**

---

Beyond accuracy, the project is divided into three phases:

- **Phase 1: Model Training**  
  Quantify the accuracy (F1-score) and speed (train time) trade-off when freezing 0, 4, 8, or 11 of BERT's 12 encoder layers.

- **Phase 2: Probability Calibration**  
  Analyze the "honesty" of the best model's probability scores and correct for overconfidence using Isotonic Regression, measuring success with Expected Calibration Error (ECE).

- **Phase 3: Ordinal Mapping**  
  Test the usefulness of the calibrated probabilities on a downstream task by mapping them to 1–5 star ratings, measured by RMSE and MAE.

---

## Key Findings

### Efficiency vs. Accuracy

A fully fine-tuned BERT (`bert_full_finetune`) achieved the highest F1-score (avg. **93.96%**). However, freezing the bottom 8 layers (`bert_frozen_8`) retained ~**99.5%** of this performance (avg. **93.51% F1**) while reducing training time per epoch by ~**40%**.

### Probability Calibration

The best-performing model (`bert_full_finetune_seed123`) was highly overconfident.  
Post-hoc calibration with **Isotonic Regression** reduced the Expected Calibration Error (**ECE**) by **64.23%**, making the model's confidence scores significantly more reliable.

### Downstream Task Improvement

The more "honest" calibrated probabilities were more useful for nuanced predictions.  
When mapped to 1–5 star ratings, the calibrated probabilities reduced:

- **RMSE** by **3.77%**
- **MAE** by **2.21%**

compared to the uncalibrated ones.

---

## Project Structure

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
│   ├── models/           # (Generated) Stores trained .pt model checkpoints
│   ├── metrics/          # (Generated) Stores experiment .json log files
│   ├── probabilities/    # (Generated) Stores .npz probability/label files
│   └── plots/            # (Generated) Stores .png analysis plots
│
├── src/
│   ├── models/
│   │   ├── bert.py           # BERT model definition with layer freezing
│   │   └── kim_cnn.py        # Kim (2014) CNN model definition
│   ├── engine/
│   │   ├── trainer.py        # train_epoch function
│   │   └── evaluator.py      # evaluate function (calculates F1, Acc, etc.)
│   ├── utils/
│   │   └── cnn_utils.py      # GloVe/Vocab helpers for the CNN
│   ├── data/
│   │   └── dataset.py        # PyTorch SentimentDataset class
│   └── postprocessing/
│       ├── calibrate.py      # Phase 2: Runs ECE analysis
│       └── ordinal.py        # Phase 3: Runs 1–5 star rating analysis
│
├── run_experiment.py               # Phase 1: Main script to train models
├── run_probability_generation.py   # Phase 2/3: Script to generate probabilities
└── Readme.md
```
#### Added `Sentiment Analysis/notebooks` folder. Contains google colab notebooks for the experiments.

## How to Run the Analysis

This project is designed to be run in a sequential, 3-phase workflow.

---

### Prerequisites

You will need to have your data prepared in `data/processed/` and your GloVe embeddings in `data/embeddings/`. Ensure all paths in the `.yaml` config files point to the correct locations.

Download the Glove embeddings i.e. glove.6B.300d.txt and place it inside the `data/embeddings` folder.

Install the required libraries:

```bash
pip install torch transformers pandas numpy scikit-learn pyyaml tqdm matplotlib seaborn
```
### Phase 1: Train Models

Use `run_experiment.py` to train your models. This script reads a config file, trains the specified model, and saves the best checkpoint (based on validation loss) to `outputs/models/` and a full metrics log to `outputs/metrics/`.

Run a specific configuration with a seed:

```bash
# Train the full fine-tuned BERT model with seed 42
python run_experiment.py --config configs/bert_full_finetune.yaml --seed 42

# Train the 8-layer frozen BERT model with seed 123
python run_experiment.py --config configs/bert_frozen_8.yaml --seed 123

# Train the static CNN baseline with seed 2025
python run_experiment.py --config configs/cnn_baseline.yaml --seed 2025

# Train the non-static CNN with seed 42
python run_experiment.py --config configs/cnn_non_static.yaml --seed 42
```
#### A) Run Calibration Analysis (Phase 2)

Use `calibrate.py` to test for overconfidence. This script loads the `.npz` files, fits an Isotonic Regression calibrator on the validation data, and reports the "Before" vs. "After" ECE and Brier scores on the test data. It saves a Reliability Diagram and Correction Function plot to `outputs/plots/`.

Example:

```bash
python src/postprocessing/calibrate.py --run_name "bert_full_finetune_seed123.pt"
```
#### B) Run Ordinal Mapping Analysis (Phase 3)

Use `ordinal.py` to test the calibrated probabilities on the star-rating task. This script repeats the calibration step and then compares the performance (MAE, RMSE, etc.) of 1–5 star predictions from both uncalibrated and calibrated probabilities. It saves a final box plot to `outputs/plots/`.

Example:

```bash
python src/postprocessing/ordinal.py --run_name "bert_full_finetune_seed123.pt"
```
## Acknowledgement
This project was developed in a collaborative partnership between the author and Google's Gemini. The overall architecture, design patterns, and strategic direction were conceived by the author, who then leveraged Gemini as an AI programming partner. Through an iterative process of instruction and feedback, Gemini assisted in generating, refactoring, and documenting the code to bring the author's vision to life.
