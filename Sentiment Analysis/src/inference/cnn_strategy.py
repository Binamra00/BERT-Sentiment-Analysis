import torch
from src.models.kim_cnn import KimCNN
from src.utils.cnn_utils import build_vocab_and_tokenizer, collate_fn_cnn
from src.inference.strategy import InferenceStrategy
import sys

class CnnInferenceStrategy(InferenceStrategy):
    """Inference strategy for Kim CNN models."""

    def __init__(self):
        self.vocab = None

    def load_model(self, config, model_path, device):
        print("Building Vocabulary for CNN (from train data)...")
        # Rebuild vocab to match training
        self.vocab, tokenizer = build_vocab_and_tokenizer(
            config['data']['train_path'], unk_token="<unk>", pad_token="<pad>"
        )
        
        print("Initializing KimCNN model...")
        model = KimCNN(
            vocab_size=len(self.vocab),
            embed_dim=config['model']['embed_dim'],
            num_classes=config['model']['num_classes'],
            num_filters=config['model']['num_filters'],
            filter_sizes=config['model']['filter_sizes'],
            dropout_prob=config['model']['dropout_prob']
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
        if self.vocab is None:
            raise ValueError("Vocabulary not initialized. Call load_model first.")
        # Return lambda with captured vocab
        return lambda b: collate_fn_cnn(b, self.vocab["<pad>"])

    def prepare_batch(self, batch, device):
        ids = batch['ids'].to(device)
        
        inputs = {
            'ids': ids
        }
        
        labels = batch['label'].to(device)
        
        # Handle optional ratings
        if 'star_rating' in batch:
            ratings = batch['star_rating'].to(device)
        else:
            ratings = torch.zeros_like(labels)
            
        return inputs, labels, ratings