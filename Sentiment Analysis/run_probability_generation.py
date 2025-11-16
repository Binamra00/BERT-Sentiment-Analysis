"""
run_probability_generation.py

This script loads a pre-trained model checkpoint from an experiment
and runs inference on the validation and test datasets. It saves the
raw (uncalibrated) probabilities and true labels to disk for
use in post-processing (e.g., calibration, ordinal mapping).

Usage:
    python run_probability_generation.py --config configs/bert_full_finetune.yaml --seed 123
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
import sys
import yaml
import argparse
from tqdm import tqdm
from transformers import BertTokenizer

# --- 1. Import Custom Modules ---

try:
    from src.data.dataset import SentimentDataset
    from src.models.bert import BERTModel

except ImportError as e:
    print("Error: Could not import custom modules from 'src'.")
    print(f"Details: {e}")
    print(f"Current Working Directory: {os.getcwd()}")
    print("Please ensure you are running this script from the project root directory.")
    sys.exit(1)

# --- 2. Helper Functions ---

def load_model_and_tokenizer(model_config, model_path, device):
    """Loads the BERT model and tokenizer."""
    print(f"Loading tokenizer: {model_config['pretrained_model_name']}")
    tokenizer = BertTokenizer.from_pretrained(model_config['pretrained_model_name'])
    
    print("Initializing model...")
    model = BERTModel(
        pretrained_model_name=model_config['pretrained_model_name'],
        num_classes=model_config['num_classes'],
        dropout_prob=model_config['dropout_prob'],
        freeze_embed=model_config['freeze_embed'],
        freeze_layers=model_config['freeze_layers']
    )
    
    print(f"Loading model state from: {model_path}")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Error loading model state: {e}")
        print("Please ensure the model path is correct.")
        sys.exit(1)
        
    model.to(device)
    model.eval() 
    return model, tokenizer

def create_dataloaders(data_config, train_config, tokenizer):
    """Creates validation and test dataloaders."""
    print("Creating DataLoaders...")
    

    val_dataset = SentimentDataset(
        csv_file=data_config['val_path'],
        tokenizer=tokenizer,
        max_length=train_config['max_length']
    )

    test_dataset = SentimentDataset(
        csv_file=data_config['test_path'],
        tokenizer=tokenizer,
        max_length=train_config['max_length']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config['batch_size'],
        shuffle=False 
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=train_config['batch_size'],
        shuffle=False
    )
    return val_loader, test_loader

def get_probs_and_labels(model, data_loader, device):
    """
    Runs inference and returns raw probabilities (for class 1), true labels,
    and the original star ratings.
    """
    all_probs = []
    all_labels = []
    all_ratings = [] 
    
    with torch.no_grad(): 
        for batch in tqdm(data_loader, desc="Getting Probabilities"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            ratings = batch['star_rating'].to(device)
            
            outputs = model(input_ids, attention_mask)
            
            probs = F.softmax(outputs, dim=1)[:, 1] 
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_ratings.append(ratings.cpu().numpy())
            
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    all_ratings = np.concatenate(all_ratings)
    
    return all_probs, all_labels, all_ratings

# --- 3. Main Execution ---

def main(args):
    project_root = os.path.dirname(os.path.abspath(__file__))

    print(f"Loading config: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # --- Setup paths and device ---
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    run_name = config['training']['run_name']
    
    model_name = f"{run_name}_seed{args.seed}.pt"
    
    model_path = os.path.join(project_root, "outputs", "models", model_name)
    
    output_dir = os.path.join(project_root, "outputs", "probabilities")
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Load Model and Tokenizer ---
    model, tokenizer = load_model_and_tokenizer(
        config['model'], model_path, device
    )
    
    # --- Create DataLoaders ---
    val_loader, test_loader = create_dataloaders(
        config['data'], config['training'], tokenizer
    )
    
    # --- Process Validation Set ---
    print("Processing Validation Set...")
    val_probs, val_labels, val_ratings = get_probs_and_labels(model, val_loader, device)
    val_output_path = os.path.join(output_dir, f"{model_name}_validation_outputs.npz")
    np.savez(
        val_output_path, 
        probs=val_probs, 
        labels=val_labels,
        ratings=val_ratings
    )
    print(f"Saved validation outputs to {val_output_path}")

    # --- Process Test Set ---
    print("Processing Test Set...")
    test_probs, test_labels, test_ratings = get_probs_and_labels(model, test_loader, device)
    test_output_path = os.path.join(output_dir, f"{model_name}_test_outputs.npz")
    np.savez(
        test_output_path,
        probs=test_probs,
        labels=test_labels,
        ratings=test_ratings
    )
    print(f"Saved test outputs to {test_output_path}")

    print("\n--- Probability Generation Complete! ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate and save model probabilities for post-processing."
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the model configuration YAML file (e.g., configs/bert_full_finetune.yaml)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="The seed of the trained model checkpoint to load (e.g., 123)"
    )
    
    args = parser.parse_args()
    main(args)