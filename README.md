# Understanding the Impact of BERT Layer Freezing on Accuracy and Speed of Sentiment Analysis

This project analyzes the trade-off between model accuracy and training efficiency in sentiment analysis. It uses the IMDb movie review dataset to compare a CNN baseline against various BERT fine-tuning strategies, with a focus on layer freezing.

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
.
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
