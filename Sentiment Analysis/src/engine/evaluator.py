"""
Contains the function for evaluating the model on a dataset.
"""

import torch
import torch.nn as nn
from tqdm import tqdm

# Import sklearn for F1, Precision, and Recall
from sklearn.metrics import f1_score, precision_score, recall_score

def evaluate(model, dataloader, loss_fn, device):
    """
    Performs evaluation on a given dataset (validation or test).
    
    Args:
        model (nn.Module): The PyTorch model to evaluate.
        dataloader (DataLoader): DataLoader for the evaluation data.
        loss_fn (nn.Module): The loss function.
        device (torch.device): The device to run evaluation on.
        
    Returns:
        tuple: (average_loss, accuracy, f1, precision, recall)
    """
    model.eval()  # Set the model to evaluation mode
    
    total_loss = 0.0
    
    
    # Store all labels and preds for sklearn metrics
    all_labels = []
    all_preds = []

    # No gradients needed for evaluation
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            
            # 1. Prepare data
            labels = batch.pop("label").to(device)
            inputs = {key: val.to(device) for key, val in batch.items()}

            # 2. Forward pass
            outputs = model(**inputs)

            # 3. Calculate loss
            loss = loss_fn(outputs, labels)
            
            # 4. Get predictions
            preds = torch.argmax(outputs, dim=1)
            
            # 5. Store metrics
            total_loss += loss.item()
            
            
            # Move labels and preds to CPU for sklearn
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Calculate average loss
    avg_loss = total_loss / len(dataloader)
    
    
    # Calculate all sklearn metrics
    accuracy = (torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean().item()
    f1 = f1_score(all_labels, all_preds, average='binary')
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    
    # Updated print statement to include F1
    print(f"Val/Test Loss: {avg_loss:.4f} | Acc: {accuracy:.4f} | F1: {f1:.4f}")
    
    
    # Return all metrics
    return avg_loss, accuracy, f1, precision, recall