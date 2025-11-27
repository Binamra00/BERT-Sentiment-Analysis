"""
run_probability_generation.py

This is the main client script for generating probabilities.
It uses the Strategy Pattern to support multiple model types (BERT, CNN)
without needing hard-coded if/else logic for each one.

Usage:
    python generate_probabilities.py --config configs/cnn_non_static.yaml --seed 2025
    python generate_probabilities.py --config configs/cnn_baseline.yaml
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import yaml
import argparse
from tqdm import tqdm

# --- Import Custom Modules ---
try:
    from src.data.dataset import SentimentDataset
    # We import the Factory function from the inference package
    from src.inference import get_inference_strategy
except ImportError as e:
    print("Error: Could not import custom modules.")
    print(f"Details: {e}")
    sys.exit(1)

def create_dataloaders(data_config, train_config, tokenizer, collate_fn):
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
        shuffle=False,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=train_config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )
    return val_loader, test_loader

def get_probs_and_labels(model, data_loader, device, strategy):
    """
    Runs inference using the provided strategy to handle batch preparation.
    """
    all_probs = []
    all_labels = []
    all_ratings = [] 
    
    with torch.no_grad(): 
        for batch in tqdm(data_loader, desc="Getting Probabilities"):
            # Strategy handles moving to device and formatting inputs
            inputs, labels, ratings = strategy.prepare_batch(batch, device)
            
            # Forward pass (inputs is a dict, so we unpack it)
            outputs = model(**inputs)
            
            probs = F.softmax(outputs, dim=1)[:, 1] 
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_ratings.append(ratings.cpu().numpy())
            
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    all_ratings = np.concatenate(all_ratings)
    
    return all_probs, all_labels, all_ratings

def main(args):
    project_root = os.path.dirname(os.path.abspath(__file__))

    print(f"Loading config: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    run_name = config['training']['run_name']
    
    # --- LOGIC CHANGE FOR OPTIONAL SEED ---
    if args.seed is not None:
        model_name = f"{run_name}_seed{args.seed}.pt"
    else:
        model_name = f"{run_name}.pt"
    
    model_path = os.path.join(project_root, "outputs", "models", model_name)
    output_dir = os.path.join(project_root, "outputs", "probabilities")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Get Strategy (The Factory call)
    model_type = config['model']['type']
    print(f"Selected Inference Strategy: {model_type}")
    
    # This line is the magic of the Strategy Pattern:
    # We get a strategy object that knows how to handle the specific model type.
    strategy = get_inference_strategy(model_type)
    
    # 2. Load Model & Tools via Strategy
    model, tokenizer = strategy.load_model(config, model_path, device)
    collate_fn = strategy.get_collate_fn()
    
    # 3. Create DataLoaders
    val_loader, test_loader = create_dataloaders(config['data'], config['training'], tokenizer, collate_fn)
    
    # 4. Process
    for name, loader in [("validation", val_loader), ("test", test_loader)]:
        print(f"Processing {name} set...")
        probs, labels, ratings = get_probs_and_labels(model, loader, device, strategy)
        
        output_path = os.path.join(output_dir, f"{model_name}_{name}_outputs.npz")
        np.savez(output_path, probs=probs, labels=labels, ratings=ratings)
        print(f"Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    # Changed required=True to required=False (default=None is implicit)
    parser.add_argument("--seed", type=int, help="Seed used for training (optional)")
    args = parser.parse_args()
    main(args)