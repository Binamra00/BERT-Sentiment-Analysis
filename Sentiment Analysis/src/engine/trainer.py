"""
Contains the function for a single training epoch.
"""

import torch
import torch.nn as nn
from tqdm import tqdm

def train_epoch(model, dataloader, optimizer, loss_fn, device, scheduler=None):
    """
    Performs a single training epoch.
    
    Args:
        model (nn.Module): The PyTorch model to train.
        dataloader (DataLoader): DataLoader for the training data.
        loss_fn (nn.Module): The loss function.
        optimizer (Optimizer): The optimizer.
        device (torch.device): The device to run training on (e.g., 'cuda' or 'cpu').
        scheduler (Scheduler, optional): Learning rate scheduler.
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()  # Set the model to training mode
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()

        # 1. Prepare data
        # Pop 'label' and move it to the device.
        labels = batch.pop("label").to(device)
        
        # Move all remaining items in the batch (model inputs) to the device.
        inputs = {key: val.to(device) for key, val in batch.items()}

        # 2. Forward pass
        # Use dictionary unpacking. This works for both models:
        # - KimCNN will receive model(ids=...)
        # - BERTModel will receive model(input_ids=..., attention_mask=...)
        outputs = model(**inputs)

        # 3. Calculate loss
        loss = loss_fn(outputs, labels)
        
        # 4. Backward pass
        loss.backward()
        optimizer.step()
        
        # 5. Update scheduler (if one exists)
        if scheduler:
            scheduler.step()
            
        # 6. Calculate metrics
        # Get the predictions (class with the highest logit)
        preds = outputs.argmax(dim=1)
        correct = (preds == labels).sum().item()
        
        # Accumulate metrics
        total_loss += loss.item()
        total_correct += correct
        total_samples += labels.size(0)

    # Calculate average loss and accuracy for the epoch
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples
    
    print(f"Train Loss: {avg_loss:.4f} | Train Acc: {accuracy:.4f}")
    
    return avg_loss, accuracy