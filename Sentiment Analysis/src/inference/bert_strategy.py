import torch
from transformers import BertTokenizer
from src.models.bert import BERTModel
from src.inference.strategy import InferenceStrategy
import sys

class BertInferenceStrategy(InferenceStrategy):
    """Inference strategy for BERT models."""

    def load_model(self, config, model_path, device):
        print(f"Loading BERT tokenizer: {config['model']['pretrained_model_name']}")
        tokenizer = BertTokenizer.from_pretrained(config['model']['pretrained_model_name'])
        
        print("Initializing BERT model...")
        model = BERTModel(
            pretrained_model_name=config['model']['pretrained_model_name'],
            num_classes=config['model']['num_classes'],
            dropout_prob=config['model']['dropout_prob'],
            freeze_embed=config['model']['freeze_embed'],
            freeze_layers=config['model']['freeze_layers']
        )
        
        print(f"Loading model state from: {model_path}")
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as e:
            print(f"Error loading model state: {e}")
            sys.exit(1)
            
        model.to(device)
        model.eval()
        return model, tokenizer

    def get_collate_fn(self):
        # BERT uses default collate
        return None

    def prepare_batch(self, batch, device):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        labels = batch['label'].to(device)
        # Handle optional ratings
        if 'star_rating' in batch:
            ratings = batch['star_rating'].to(device)
        else:
            ratings = torch.zeros_like(labels)
            
        return inputs, labels, ratings