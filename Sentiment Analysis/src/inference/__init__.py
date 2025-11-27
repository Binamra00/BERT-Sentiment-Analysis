from src.inference.bert_strategy import BertInferenceStrategy
from src.inference.cnn_strategy import CnnInferenceStrategy

def get_inference_strategy(model_type):
    """
    Factory function to return the appropriate inference strategy.
    
    Args:
        model_type (str): 'bert' or 'cnn'
        
    Returns:
        InferenceStrategy: An instance of the requested strategy.
    """
    if model_type == 'bert':
        return BertInferenceStrategy()
    elif model_type == 'cnn':
        return CnnInferenceStrategy()
    else:
        raise ValueError(f"Unknown model type: {model_type}")