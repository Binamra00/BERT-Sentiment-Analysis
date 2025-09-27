"""
Data Partitioner Module.

This module is responsible for splitting the full, loaded dataset into
training, validation, and test sets according to the project specifications.
It respects the original train/test split provided by the dataset creators.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

class DataPartitioner:
    """
    Partitions a DataFrame into training, validation, and test sets.
    """
    def __init__(self, validation_split_ratio: float = 0.1, random_state: int = 42):
        """
        Initializes the DataPartitioner.

        Args:
            validation_split_ratio (float): The proportion of the original training
                                            data to be used for validation.
            random_state (int): A seed for the random number generator to ensure
                                reproducibility of the split.
        """
        self.validation_split_ratio = validation_split_ratio
        self.random_state = random_state
        print(f"DataPartitioner initialized with a {validation_split_ratio:.0%} validation split ratio.")

    def partition(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits the DataFrame into three sets: train, validation, and test.

        It first separates the original held-out test set. Then, it splits
        the original training set into a new, smaller training set and a
        validation set.

        Args:
            df (pd.DataFrame): The full DataFrame loaded by DataLoader,
                               containing the 'split' column.

        Returns:
            A tuple containing three DataFrames: (train_df, validation_df, test_df).
        
        Raises:
            ValueError: If the 'split' column is not found in the DataFrame.
        """
        if 'split' not in df.columns:
            raise ValueError("Input DataFrame must contain a 'split' column.")

        # 1. Separate the original held-out test set
        test_df = df[df['split'] == 'test'].copy()
        original_train_df = df[df['split'] == 'train'].copy()

        # 2. Split the original training data into a new training set and a validation set
        train_df, validation_df = train_test_split(
            original_train_df,
            test_size=self.validation_split_ratio,
            random_state=self.random_state,
            stratify=original_train_df['sentiment_label'] # Ensures balanced split
        )
        
        # 3. Drop the now-redundant 'split' column from all partitions
        train_df = train_df.drop(columns=['split'])
        validation_df = validation_df.drop(columns=['split'])
        test_df = test_df.drop(columns=['split'])

        print("Partitioning complete:")
        print(f"  - Training set size:   {len(train_df)}")
        print(f"  - Validation set size: {len(validation_df)}")
        print(f"  - Test set size:       {len(test_df)}")
        
        return train_df.reset_index(drop=True), validation_df.reset_index(drop=True), test_df.reset_index(drop=True)
