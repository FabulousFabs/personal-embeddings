"""Gram matrix implementations and utilities.
"""

from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn.functional as F

from typing import Callable

"""Gram implementations.
"""

def gram_euclidean(X: torch.Tensor, norm: bool = True) -> torch.Tensor:
    """Compute the gram matrix based on euclidean distances.
    
    Parameters
    ----------
    X : torch.Tensor
        Input data of shape `(n_batch, n_features)`.
    norm : bool, default=True
        Should features be normalised? For euclidean distances, this 
        means that outcomes are a monotonic function of cosine distances. 
        Consequently, this is critical for embeddings (but should not 
        be applied to raw behavioural data).
    
    Returns
    -------
    gram : torch.Tensor
        Gram matrix of shape `(n_batch, n_batch)`.
    """
    
    # check norm
    if norm:
        X = F.normalize(X, dim = -1)
    
    # make gram matrix
    return torch.linalg.norm(X[:,None,:] - X[None,:,:], dim = -1)

def gram_cosine(X: torch.Tensor) -> torch.Tensor:
    """Compute the gram matrix based on cosine similarity.
    
    Parameters
    ----------
    X : torch.Tensor
        Input data of shape `(n_batch, n_features)`.
    
    Returns
    -------
    gram : torch.Tensor
        Gram matrix of shape `(n_batch, n_batch)`.
    """
    
    # apply norm
    X = F.normalize(X, dim = -1)
    
    return X @ X.t()

"""Registries and utility functions.
"""

# register gram functions
grams = {
    'euclidean': gram_euclidean,
    'cosine': gram_cosine
}

def get_gram_fn(cfg: DictConfig) -> Callable:
    """Obtain the desired gram matrix.
    
    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration.
    
    Returns
    -------
    fn : Callable
        The desired gram function.
    """
    
    # check function
    if cfg.metrics.gram not in grams:
        raise ValueError(
            f'Unknown gram function `{cfg.metrics.gram}`. ' + 
            f'Available options: {list(grams.keys())}.'
        )
    
    return grams[cfg.metrics.gram]