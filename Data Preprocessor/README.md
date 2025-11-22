# **IMDb Movie Review Preprocessing Pipeline**
This project contains a robust, modular, and reusable Python pipeline for cleaning, structuring, and preparing the Stanford Large Movie Review (IMDb) Dataset for sentiment analysis tasks. The entire pipeline is built with a focus on software engineering best practices, including the use of the Strategy and Facade design patterns to ensure the code is maintainable and extensible.

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

## Acknowledgement
This project was developed in a collaborative partnership between the author and Google's Gemini. The overall architecture, design patterns, and strategic direction were conceived by the author, who then leveraged Gemini as an AI programming partner. Through an iterative process of instruction and feedback, Gemini assisted in generating, refactoring, and documenting the code to bring the author's vision to life.
