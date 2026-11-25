"""Activation functions

Registries and helper functions for activation 
functions in our networks.
"""

import torch.nn as nn
import torch.nn.functional as F

# setup activation functions
activation_functions = {
    'gelu': nn.GELU,
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'identity': nn.Identity
}

def make_activation(name: str) -> nn.Module:
    """Make specified activation function.
    
    Parameters
    ----------
    name : {'gelu', 'relu', 'tanh', 'sigmoid', 'identity'}
        Name of desired activation function.
    
    Returns
    -------
    activation : nn.Module
        Desired activation function.
    
    Raises
    ------
    ValueError
        If `name` is not a supported activation name.
    """
    
    # check name
    if name not in activation_functions:
        raise ValueError(
            f'Unknown activation function {name}. Available ' +
            f'activation functions: {list(activation_functions.keys())}.'
        )
    
    return activation_functions[name]