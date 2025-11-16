import re
import pandas as pd
from abc import ABC, abstractmethod
from tqdm import tqdm

# Import the contractions dictionary from our resources module
from resources.contractions import CONTRACTIONS


class CleaningStrategy(ABC):

    @abstractmethod
    def clean(self, text_series: pd.Series) -> pd.Series:
        pass


class RemoveHTMLStrategy(CleaningStrategy):
    def clean(self, text_series: pd.Series) -> pd.Series:
        # First, handle the specific <br> tags by replacing them with a space
        series = text_series.str.replace(r'<br\s*/?>', ' ', regex=True)
        # Then, remove any other remaining HTML tags
        series = series.str.replace(r'<.*?>', ' ', regex=True)
        return series


class LowercaseStrategy(CleaningStrategy):
    def clean(self, text_series: pd.Series) -> pd.Series:
        return text_series.str.lower()


class ExpandContractionsStrategy(CleaningStrategy):
    def __init__(self):
        # Compile the regex for efficiency
        self.contraction_re = re.compile('(%s)' % '|'.join(CONTRACTIONS.keys()))

    def _expand_match(self, match):
        return CONTRACTIONS[match.group(0)]

    def clean(self, text_series: pd.Series) -> pd.Series:
        return text_series.apply(lambda text: self.contraction_re.sub(self._expand_match, text))


class IsolatePunctuationStrategy(CleaningStrategy):
    def clean(self, text_series: pd.Series) -> pd.Series:
        # Isolate common punctuation marks
        series = text_series.str.replace(r'([!?.(),:])', r' \1 ', regex=True)
        # Isolate common emoticons
        series = series.str.replace(r'(:\)|:-\)|:\(|:-\(|;\)|;-\))', r' \1 ', regex=True)
        return series


class RemoveNoiseStrategy(CleaningStrategy):
    def clean(self, text_series: pd.Series) -> pd.Series:
        # Remove URLs
        series = text_series.str.replace(r'https?://\S+|www\.\S+', '', regex=True)
        # Remove any characters that are not standard ASCII
        series = series.str.replace(r'[^\x00-\x7F]+', ' ', regex=True)
        # Replace multiple whitespace characters with a single space and strip ends
        series = series.str.replace(r'\s+', ' ', regex=True).str.strip()
        return series


class TextCleaner:
    def __init__(self, strategies: list[CleaningStrategy]):
        self.strategies = strategies
        print(f"TextCleaner initialized with {len(self.strategies)} strategies.")

    def apply(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        df = data.copy()

        # Use tqdm to show a progress bar for the strategies
        for strategy in tqdm(self.strategies, desc="Applying Cleaning Strategies"):
            df[column] = strategy.clean(df[column])

        print("All cleaning strategies have been applied successfully.")
        return df
