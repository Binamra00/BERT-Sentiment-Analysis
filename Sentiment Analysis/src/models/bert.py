#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERT Sentiment Classification — Sun et al. (2019) configuration
Implements BERT-base-uncased fine-tuning for sentiment classification
with optional layer freezing (embedding + N encoder layers).

Reference:
    Sun, C., et al. (2019). "How to Fine-Tune BERT for Text Classification?"
    Proceedings of CCL 2019.


"""

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig


class BERTModel(nn.Module):
    """BERT sentiment classifier with configurable layer freezing."""

    def __init__(
        self,
        num_classes: int = 2,
        dropout_prob: float = 0.1,
        freeze_embed: bool = False,
        freeze_layers: int = 0,
        pretrained_model_name: str = "bert-base-uncased",
    ):
        super(BERTModel, self).__init__()
        self.config = BertConfig.from_pretrained(pretrained_model_name)
        self.bert = BertModel.from_pretrained(pretrained_model_name, config=self.config)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        self.freeze_layers(freeze_embed, freeze_layers)

    def freeze_layers(self, freeze_embed: bool = False, freeze_layers: int = 0):
        """Freeze embedding layer and first N encoder layers."""
        if freeze_embed:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
            print("[INFO] Embedding layer frozen.")

        if freeze_layers > 0:
            for i, layer in enumerate(self.bert.encoder.layer):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
            print(f"[INFO] Frozen first {freeze_layers} encoder layers.")

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output  # CLS token
        logits = self.classifier(self.dropout(pooled))
        return logits

    def count_trainable_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[INFO] Trainable parameters: {trainable:,} / {total:,}")
        return trainable, total


if __name__ == "__main__":
    dummy_ids = torch.randint(0, 100, (2, 32))
    mask = torch.ones_like(dummy_ids)
    model = BERTModel(num_classes=2, freeze_embed=True, freeze_layers=4)
    model.count_trainable_parameters()
    logits = model(dummy_ids, mask)
    print("Output logits shape:", logits.shape)
