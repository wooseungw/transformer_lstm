import numpy as np
import torch

def positional_encoding(position, d_model):
    """
    Compute positional encoding for a given position and model dimension.
    
    Args:
    - position (int): Position of a token in the sequence.
    - d_model (int): Dimension of the model.
    
    Returns:
    - pos_enc (torch.Tensor): Positional encoding of size (1, d_model).
    """
    # Initialize a position array
    pos = np.arange(position)[:, np.newaxis]
    
    # Compute the div term
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    # Compute the positional encoding values
    pos_enc = np.zeros((position, d_model))
    pos_enc[:, 0::2] = np.sin(pos * div_term)
    pos_enc[:, 1::2] = np.cos(pos * div_term)
    
    return torch.tensor(pos_enc[np.newaxis, ...], dtype=torch.float32)

# Example usage:
d_model = 512
max_length = 100
pos_encoding = positional_encoding(max_length, d_model)
print(pos_encoding)  # Expected: torch.Size([1, 100, 512])
