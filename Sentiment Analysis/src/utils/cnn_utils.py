"""
Utility functions for the Kim CNN pipeline.
Includes:
- Vocabulary building from the training data.
- Loading and mapping GloVe embeddings.
- A custom collate function for padding CNN batches.
"""

import torch
import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm

def build_vocab_and_tokenizer(train_csv_path, unk_token="<unk>", pad_token="<pad>"):
    """
    Builds a vocabulary and a simple tokenizer function from the training data.
    
    Args:
        train_csv_path (str): Path to the train_clean.csv file.
        unk_token (str): The token to use for unknown words.
        pad_token (str): The token to use for padding.
        
    Returns:
        tuple: (vocab, tokenizer_fn)
            - vocab (dict): A word-to-index dictionary.
            - tokenizer_fn (function): A function that takes text and returns a list of token IDs.
    """
    print(f"[INFO] Building vocabulary from {train_csv_path}...")
    df = pd.read_csv(train_csv_path)
    
    # 1. Build a simple tokenizer and count word frequencies
    word_counts = Counter()
    for text in tqdm(df['review_text'], desc="Building Vocab"):
        word_counts.update(text.split())
        
    # 2. Create the vocabulary (word-to-index mapping)
    # Start with special tokens. PAD should be index 0 for nn.Embedding.
    vocab = {pad_token: 0, unk_token: 1}
    
    # Add all other words
    for word, _ in word_counts.items():
        if word not in vocab:
            vocab[word] = len(vocab)
            
    print(f"[INFO] Vocabulary built. Total size: {len(vocab)} tokens.")
    
    # 3. Create the tokenizer function
    def tokenizer_fn(text_str):
        """Converts a string of text into a list of token IDs."""
        token_ids = []
        for word in text_str.split():
            token_ids.append(vocab.get(word, vocab[unk_token]))
        return token_ids
        
    return vocab, tokenizer_fn


def load_glove_embeddings(vocab, glove_path, embed_dim):
    """
    Loads GloVe embeddings and creates an embedding matrix for our vocabulary.
    
    Args:
        vocab (dict): The word-to-index vocabulary.
        glove_path (str): Path to the glove.6B.300d.txt file.
        embed_dim (int): The embedding dimension (e.g., 300).
        
    Returns:
        torch.Tensor: An embedding matrix of shape (vocab_size, embed_dim).
    """
    print(f"[INFO] Loading GloVe embeddings from {glove_path}...")
    
    # Initialize the embedding matrix with zeros
    vocab_size = len(vocab)
    embedding_matrix = np.zeros((vocab_size, embed_dim))
    
    words_found = 0
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading GloVe"):
            parts = line.split()
            word = parts[0]
            if word in vocab:
                vector = np.array(parts[1:], dtype=np.float32)
                embedding_matrix[vocab[word]] = vector
                words_found += 1
                
    print(f"[INFO] GloVe embeddings loaded.")
    print(f"[INFO] {words_found} / {vocab_size} words found in GloVe vocab.")
    
    return torch.tensor(embedding_matrix, dtype=torch.float32)


def collate_fn_cnn(batch, pad_idx):
    """
    Custom collate function for the CNN DataLoader.
    
    This function takes a list of dictionaries (from SentimentDataset),
    pads the 'ids' to the maximum length *in the batch*,
    and stacks 'ids' and 'labels' into tensors.
    
    Args:
        batch (list): A list of dictionaries, e.g., [{'ids': [1,2,3], 'label': 0}, ...]
        pad_idx (int): The index of the <pad> token.
        
    Returns:
        dict: A dictionary of batched and padded tensors: {'ids': tensor, 'label': tensor}
    """
    labels = [item['label'] for item in batch]
    ids_list = [item['ids'] for item in batch]
    
    # Find the max length in this specific batch
    max_len = max(len(ids) for ids in ids_list)
    
    # Pad all sequences to max_len
    padded_ids = []
    for ids in ids_list:
        padding_needed = max_len - len(ids)
        padded_ids.append(ids + [pad_idx] * padding_needed)
        
    # Convert to tensors
    ids_tensor = torch.tensor(padded_ids, dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    return {
        'ids': ids_tensor,
        'label': labels_tensor
    }