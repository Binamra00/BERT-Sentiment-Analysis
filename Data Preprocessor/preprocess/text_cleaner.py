"""
Text Cleaner Module using the Strategy Design Pattern.

This module defines the text cleaning pipeline. It consists of:
1. An abstract base class, `CleaningStrategy`, which defines the interface
   for all cleaning operations.
2. Concrete strategy classes that inherit from `CleaningStrategy` and
   implement specific cleaning steps (e.g., removing HTML, lowercasing).
3. A `TextCleaner` context class that takes a list of strategies and applies
   them sequentially to a DataFrame column.
"""

import re
import pandas as pd
from abc import ABC, abstractmethod
from tqdm import tqdm

# Import the contractions dictionary from our resources module
from resources.contractions import CONTRACTIONS

# --- 1. Abstract Base Class for Strategies ---

class CleaningStrategy(ABC):
    """Abstract base class for all cleaning strategies."""

    @abstractmethod
    def clean(self, text_series: pd.Series) -> pd.Series:
        """
        Applies the cleaning logic to a pandas Series.

        Args:
            text_series (pd.Series): The series containing the text data.

        Returns:
            pd.Series: The cleaned series.
        """
        pass

# --- 2. Concrete Strategy Implementations ---

class RemoveHTMLStrategy(CleaningStrategy):
    """Removes HTML tags like <br /> from the text."""
    def clean(self, text_series: pd.Series) -> pd.Series:
        # First, handle the specific <br> tags by replacing them with a space
        series = text_series.str.replace(r'<br\s*/?>', ' ', regex=True)
        # Then, remove any other remaining HTML tags
        series = series.str.replace(r'<.*?>', ' ', regex=True)
        return series

class LowercaseStrategy(CleaningStrategy):
    """Converts all text to lowercase."""
    def clean(self, text_series: pd.Series) -> pd.Series:
        return text_series.str.lower()

class ExpandContractionsStrategy(CleaningStrategy):
    """Expands common English contractions (e.g., "don't" -> "do not")."""
    def __init__(self):
        # Compile the regex for efficiency
        self.contraction_re = re.compile('(%s)' % '|'.join(CONTRACTIONS.keys()))

    def _expand_match(self, match):
        """Helper function to be used with re.sub for replacement."""
        return CONTRACTIONS[match.group(0)]

    def clean(self, text_series: pd.Series) -> pd.Series:
        # We use .apply() here because re.sub works on one string at a time
        return text_series.apply(lambda text: self.contraction_re.sub(self._expand_match, text))

class IsolatePunctuationStrategy(CleaningStrategy):
    """Adds whitespace around key punctuation and emoticons."""
    def clean(self, text_series: pd.Series) -> pd.Series:
        # Isolate common punctuation marks
        series = text_series.str.replace(r'([!?.(),:])', r' \1 ', regex=True)
        # Isolate common emoticons
        series = series.str.replace(r'(:\)|:-\)|:\(|:-\(|;\)|;-\))', r' \1 ', regex=True)
        return series

class RemoveNoiseStrategy(CleaningStrategy):
    """Removes URLs, non-ASCII characters, and consolidates whitespace."""
    def clean(self, text_series: pd.Series) -> pd.Series:
        # Remove URLs
        series = text_series.str.replace(r'https?://\S+|www\.\S+', '', regex=True)
        # Remove any characters that are not standard ASCII
        series = series.str.replace(r'[^\x00-\x7F]+', ' ', regex=True)
        # Replace multiple whitespace characters with a single space and strip ends
        series = series.str.replace(r'\s+', ' ', regex=True).str.strip()
        return series

# --- 3. The 'Context' Class ---

class TextCleaner:
    """
    The 'Context' class that applies a list of cleaning strategies.
    This class orchestrates the execution of the individual strategies.
    """
    def __init__(self, strategies: list[CleaningStrategy]):
        self.strategies = strategies
        print(f"TextCleaner initialized with {len(self.strategies)} strategies.")

    def apply(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Applies all configured cleaning strategies sequentially to a DataFrame column.

        Args:
            data (pd.DataFrame): The input DataFrame.
            column (str): The name of the text column to clean.

        Returns:
            pd.DataFrame: The DataFrame with the cleaned column.
        """
        df = data.copy()
        
        # Use tqdm to show a progress bar for the strategies
        for strategy in tqdm(self.strategies, desc="Applying Cleaning Strategies"):
            df[column] = strategy.clean(df[column])
        
        print("All cleaning strategies have been applied successfully.")
        return df
