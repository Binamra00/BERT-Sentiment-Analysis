"""
Defines the unified SentimentDataset class.
"""

import torch
import pandas as pd
from torch.utils.data import Dataset

class SentimentDataset(Dataset):
    """
    Unified PyTorch Dataset for sentiment analysis.
    
    Can be initialized with either a Hugging Face tokenizer (for BERT)
    or a simple tokenizer function (for Kim CNN).
    
    It behaves differently based on the tokenizer provided.
    """
    def __init__(self, csv_file, tokenizer, max_length):
        self.df = pd.read_csv(csv_file)
        
        # Store data columns
        self.reviews = self.df['review_text']
        self.labels = self.df['sentiment_label']
        self.ratings = self.df['star_rating'] # For later post-processing
        
        # Store processing tools
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get the raw data for one sample
        text = self.reviews.iloc[idx]
        label = self.labels.iloc[idx]
        rating = self.ratings.iloc[idx]

        # Check if we are using a Hugging Face tokenizer or our simple function
        if hasattr(self.tokenizer, 'encode_plus'):
            
            # --- Path 1: BERT (Hugging Face) Tokenizer ---
            # It has .encode_plus() and handles everything.
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                truncation=True,
                padding='max_length', # BERT needs padding *before* batching
                return_tensors='pt'
            )
            
            # Return tensors for the default DataLoader
            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'label': torch.tensor(label, dtype=torch.long),
                # 'star_rating' will be used later
                # --- ADDITION FOR PHASE 3 ---
                'star_rating': torch.tensor(rating, dtype=torch.int)
                # --- END ADDITION ---
            }
        
        else:
            
            # --- Path 2: Kim CNN (simple tokenizer function) ---
            # We just call the function.
            token_ids = self.tokenizer(text)
            
            # Apply truncation manually
            token_ids = token_ids[:self.max_length]
            
            # Return lists/ints. Padding will be handled by our custom collate_fn.
            return {
                'ids': token_ids,
                'label': label,
                # 'star_rating' will be used later
                # --- ADDITION FOR PHASE 3 ---
                'star_rating': rating
                # --- END ADDITION ---
            }