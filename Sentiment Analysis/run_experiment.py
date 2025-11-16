#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Experiment Runner for Sentiment Analysis (Kim CNN and BERT)
-------------------------------------------------------------------
Loads a YAML config, initializes the correct model (CNN or BERT),
tokenizer, dataset, and executes training/evaluation via the engine.

Saves the best model based on validation loss and logs all metrics to JSON.

Usage:
    # Run WITHOUT a specific seed (uses system randomness)
    python run_experiment.py --config configs/cnn_baseline.yaml

    # Run WITH a specific seed for reproducibility
    python run_experiment.py --config configs/bert_full_finetune.yaml --seed 42
"""

import argparse
import yaml
import os
import json
import torch
import torch.nn as nn
import numpy as np
import random
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader

# --- Import all our project components ---
from src.models.bert import BERTModel
from src.models.kim_cnn import KimCNN
from src.data.dataset import SentimentDataset
from src.engine.trainer import train_epoch
from src.engine.evaluator import evaluate
from src.utils.cnn_utils import (
    build_vocab_and_tokenizer,
    load_glove_embeddings,
    collate_fn_cnn
)

def set_seed(seed_value):
    """Sets the seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    # Ensure deterministic behavior for cuDNN (can impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Set random seed to {seed_value}")

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Experiment Runner")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    # Default is None
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility. If None, uses system randomness.")
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # 1. Setup Environment
    print(f"[INFO] Using configuration file: {args.config}")

    # Set seed only if provided
    if args.seed is not None:
        set_seed(args.seed)
    else:
        print("[INFO] No seed provided, running with system randomness.")
        # Ensure non-deterministic behavior if no seed
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


    device = torch.device(config["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Using device: {device}")

    # Get run name for saving outputs
    run_name = config["training"]["run_name"]
    model_type = config["model"]["type"]

    # Create output directories
    model_save_dir = os.path.join("outputs", "models")
    metrics_save_dir = os.path.join("outputs", "metrics")
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(metrics_save_dir, exist_ok=True)

    # Create save paths based on seed
    if args.seed is not None:
        model_save_path = os.path.join(model_save_dir, f"{run_name}_seed{args.seed}.pt")
        metrics_save_path = os.path.join(metrics_save_dir, f"{run_name}_seed{args.seed}_metrics.json")
    else:
        # Original paths if no seed
        model_save_path = os.path.join(model_save_dir, f"{run_name}.pt")
        metrics_save_path = os.path.join(metrics_save_dir, f"{run_name}_metrics.json")
    print(f"[INFO] Saving best model to: {model_save_path}")
    print(f"[INFO] Saving metrics to: {metrics_save_path}")


    # 2. Load Tokenizer, Model, and DataLoaders based on type
    if model_type == "bert":
        print(f"[INFO] Initializing BERT pipeline for model: {config['model']['pretrained_model_name']}")
        tokenizer = BertTokenizer.from_pretrained(config["model"]["pretrained_model_name"])
        train_ds = SentimentDataset(config["data"]["train_path"], tokenizer, config["training"]["max_length"])
        val_ds = SentimentDataset(config["data"]["val_path"], tokenizer, config["training"]["max_length"])
        test_ds = SentimentDataset(config["data"]["test_path"], tokenizer, config["training"]["max_length"])
        train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config["training"]["batch_size"])
        test_loader = DataLoader(test_ds, batch_size=config["training"]["batch_size"])
        model = BERTModel(
            num_classes=config["model"]["num_classes"],
            dropout_prob=config["model"]["dropout_prob"],
            freeze_embed=config["model"]["freeze_embed"],
            freeze_layers=config["model"]["freeze_layers"],
            pretrained_model_name=config["model"]["pretrained_model_name"],
        ).to(device)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config["training"]["learning_rate"],
        )
        total_steps = len(train_loader) * config["training"]["num_epochs"]
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * config["training"]["warmup_proportion"]),
            num_training_steps=total_steps,
        )
    elif model_type == "cnn":
        print("[INFO] Initializing Kim CNN pipeline...")
        vocab, cnn_tokenizer = build_vocab_and_tokenizer(
            config["data"]["train_path"], unk_token="<unk>", pad_token="<pad>"
        )
        embedding_matrix = load_glove_embeddings(
            vocab, config["data"]["embedding_path"], config["model"]["embed_dim"]
        )
        train_ds = SentimentDataset(config["data"]["train_path"], cnn_tokenizer, config["training"]["max_length"])
        val_ds = SentimentDataset(config["data"]["val_path"], cnn_tokenizer, config["training"]["max_length"])
        test_ds = SentimentDataset(config["data"]["test_path"], cnn_tokenizer, config["training"]["max_length"])
        train_loader = DataLoader(
            train_ds, batch_size=config["training"]["batch_size"], shuffle=True,
            collate_fn=lambda b: collate_fn_cnn(b, vocab["<pad>"])
        )
        val_loader = DataLoader(
            val_ds, batch_size=config["training"]["batch_size"],
            collate_fn=lambda b: collate_fn_cnn(b, vocab["<pad>"])
        )
        test_loader = DataLoader(
            test_ds, batch_size=config["training"]["batch_size"],
            collate_fn=lambda b: collate_fn_cnn(b, vocab["<pad>"])
        )
        model = KimCNN(
            vocab_size=len(vocab),
            embed_dim=config["model"]["embed_dim"],
            num_classes=config["model"]["num_classes"],
            num_filters=config["model"]["num_filters"],
            filter_sizes=config["model"]["filter_sizes"],
            dropout_prob=config["model"]["dropout_prob"]
        ).to(device)
        model.load_pretrained_embeddings(embedding_matrix, freeze=config["model"]["freeze_embed"])
        optimizer = torch.optim.Adadelta(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"]
        )
        scheduler = None
    else:
        raise ValueError(f"Unknown model type in config: {model_type}")

    model.count_trainable_parameters()
    loss_fn = nn.CrossEntropyLoss()

    # 3. Training Loop
    best_val_loss = float("inf")
    all_metrics = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_f1": [], "val_precision": [], "val_recall": [],
        "test": {}
    }
    print("\n[INFO] Starting training...")
    for epoch in range(config["training"]["num_epochs"]):
        print(f"\n===== Epoch {epoch + 1}/{config['training']['num_epochs']} =====")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, loss_fn, device, scheduler)
        val_loss, val_acc, val_f1, val_precision, val_recall = evaluate(model, val_loader, loss_fn, device)
        all_metrics["train_loss"].append(train_loss)
        all_metrics["train_acc"].append(train_acc)
        all_metrics["val_loss"].append(val_loss)
        all_metrics["val_acc"].append(val_acc)
        all_metrics["val_f1"].append(val_f1)
        all_metrics["val_precision"].append(val_precision)
        all_metrics["val_recall"].append(val_recall)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"[INFO] New best model saved to {model_save_path}")

    # 4. Final Test
    print("\n[INFO] Training complete. Loading best model for final test evaluation...")
    # Load the best model saved during training
    if os.path.exists(model_save_path):
      model.load_state_dict(torch.load(model_save_path))
      print(f"[INFO] Loaded best model from {model_save_path}")
    else:
      print(f"[WARNING] No best model found at {model_save_path}. Evaluating the final model state.")

    test_loss, test_acc, test_f1, test_precision, test_recall = evaluate(model, test_loader, loss_fn, device)

    # Conditional print statement
    if args.seed is not None:
        print(f"===== Final Test Results (Seed {args.seed}) =====")
    else:
        print(f"===== Final Test Results (Random Seed) =====")
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")

    all_metrics["test"] = {
        "loss": test_loss,
        "accuracy": test_acc,
        "f1": test_f1,
        "precision": test_precision,
        "recall": test_recall
    }

    # 5. Save Metrics
    with open(metrics_save_path, "w") as f:
        json.dump(all_metrics, f, indent=4)
    print(f"[INFO] All metrics saved to {metrics_save_path}")

if __name__ == "__main__":
    main()