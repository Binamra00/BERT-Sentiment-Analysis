from abc import ABC, abstractmethod
import torch

class InferenceStrategy(ABC):
    """
    Abstract base class for inference strategies.
    Defines the interface for loading models and preparing data processing tools.
    """
    
    @abstractmethod
    def load_model(self, config, model_path, device):
        """
        Loads the model architecture and weights.
        
        Args:
            config (dict): The configuration dictionary.
            model_path (str): Path to the saved model weights (.pt file).
            device (torch.device): Device to load the model onto.
            
        Returns:
            tuple: (model, tokenizer)
        """
        pass

    @abstractmethod
    def get_collate_fn(self):
        """
        Returns the collate function required by the DataLoader.
        
        Returns:
            callable or None: The collate function, or None if not needed.
        """
        pass
    
    @abstractmethod
    def prepare_batch(self, batch, device):
        """
        Prepares a batch for the model forward pass.
        
        Args:
            batch (dict): The batch dictionary from the DataLoader.
            device (torch.device): The device to move tensors to.
            
        Returns:
            tuple: (inputs_dict, labels, ratings)
        """
        pass