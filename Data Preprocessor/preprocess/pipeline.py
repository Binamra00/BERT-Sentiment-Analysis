
import os
import pandas as pd

# Import the individual components of our pipeline
from .data_loader import DataLoader
from .text_cleaner import (
    TextCleaner,
    RemoveHTMLStrategy,
    LowercaseStrategy,
    ExpandContractionsStrategy,
    IsolatePunctuationStrategy,
    RemoveNoiseStrategy,
)
from .partitioner import DataPartitioner


class PreprocessingPipeline:
    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path
        
        # 1. Instantiate the DataLoader
        self.loader = DataLoader(dataset_path=self.input_path)
        
        # 2. Define and instantiate the TextCleaner with our chosen strategies
        cleaning_strategies = [
            RemoveHTMLStrategy(),
            ExpandContractionsStrategy(),
            LowercaseStrategy(),
            IsolatePunctuationStrategy(),
            RemoveNoiseStrategy(),
        ]
        self.cleaner = TextCleaner(strategies=cleaning_strategies)
        
        # 3. Instantiate the DataPartitioner
        self.partitioner = DataPartitioner(validation_split_ratio=0.1, random_state=42)

        print("-" * 50)
        print("Preprocessing Pipeline Initialized and Ready.")
        print(f"Input path:  {self.input_path}")
        print(f"Output path: {self.output_path}")
        print("-" * 50)

    def run(self):
        # Step 1: Load the raw data
        print("\n--- Step 1: Loading Data ---")
        raw_df = self.loader.load_and_structure()

        # Step 2: Clean the review text
        print("\n--- Step 2: Cleaning Text ---")
        clean_df = self.cleaner.apply(raw_df, column='review_text')

        # Step 3: Partition the data
        print("\n--- Step 3: Partitioning Data ---")
        train_df, val_df, test_df = self.partitioner.partition(clean_df)

        # Step 4: Save the processed data
        print("\n--- Step 4: Saving Processed Data ---")
        os.makedirs(self.output_path, exist_ok=True)
        
        # Define file paths
        train_path = os.path.join(self.output_path, 'train_clean.csv')
        val_path = os.path.join(self.output_path, 'validation_clean.csv')
        test_path = os.path.join(self.output_path, 'test_clean.csv')

        # Save to CSV
        train_df.to_csv(train_path, index=False, encoding='utf-8')
        val_df.to_csv(val_path, index=False, encoding='utf-8')
        test_df.to_csv(test_path, index=False, encoding='utf-8')
        
        print(f"\nSuccessfully saved datasets to:")
        print(f"  - {train_path}")
        print(f"  - {val_path}")
        print(f"  - {test_path}")
        print("\nPipeline execution complete.")
