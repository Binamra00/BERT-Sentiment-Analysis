"""
Implementation of the Kim CNN model (2014) for text classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class KimCNN(nn.Module):
    """
    KimCNN model architecture.
    
    Args:
        vocab_size (int): Size of the vocabulary.
        embed_dim (int): Dimension of the word embeddings.
        num_classes (int): Number of output classes.
        num_filters (int): Number of filters for each filter size (out_channels).
        filter_sizes (list): List of filter sizes (e.g., [3, 4, 5]).
        dropout_prob (float): Dropout probability.
    """
    def __init__(self, vocab_size, embed_dim, num_classes, num_filters, filter_sizes, dropout_prob):
        super(KimCNN, self).__init__()
        
        # 1. Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # 2. Convolutional Layers
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=1,
                out_channels=num_filters,
                kernel_size=(fs, embed_dim)
            )
            for fs in filter_sizes
        ])
        
        # 3. Dropout Layer
        self.dropout = nn.Dropout(dropout_prob)
        
        # 4. Fully Connected Layer
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, ids):
        """
        Forward pass of the model.
        
        Args:
            ids (torch.Tensor): Input tensor of shape (batch_size, max_length)
            
        Returns:
            torch.Tensor: Output tensor (logits) of shape (batch_size, num_classes)
        """
        x_embed = self.embedding(ids)
        x_embed = x_embed.unsqueeze(1)
        
        conv_pooled_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x_embed)).squeeze(3)
            pooled_out = F.max_pool1d(conv_out, conv_out.shape[2]).squeeze(2)
            conv_pooled_outputs.append(pooled_out)
            
        x_cat = torch.cat(conv_pooled_outputs, dim=1)
        x_cat = self.dropout(x_cat)
        logits = self.fc(x_cat)
        
        return logits

    def load_pretrained_embeddings(self, embeddings_tensor, freeze=True):
        """
        Loads pre-trained embeddings into the embedding layer.

        Args:
            embeddings_tensor (torch.Tensor): The tensor of pre-trained embeddings.
            freeze (bool): Whether to freeze the embedding layer weights.
        """
        # Get the device of the model's existing embedding layer
        device = self.embedding.weight.device

        # Move the new embeddings to the same device BEFORE assigning them
        self.embedding.weight = nn.Parameter(embeddings_tensor.to(device))

        if freeze:
            self.embedding.weight.requires_grad = False

    def count_trainable_parameters(self):
        """Counts and prints the total and trainable parameters of the model."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[INFO] Trainable parameters: {trainable:,} / {total:,}")
        return trainable, total