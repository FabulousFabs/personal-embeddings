"""Statistical corrections.

Contains various corrections for p-values and correlations that are 
applied for inference.
"""

import torch
import numpy as np

from typing import Tuple

"""Helper functions.
"""

def adjust_clip(p: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Clip p-values to valid range [0, 1].
    
    Parameters
    ----------
    p : np.ndarray | torch.Tensor
        Vector of p-values.
    
    Returns
    -------
    p : np.ndarray | torch.Tensor
        Vector of p-values clipped to [0, 1].
    """
    
    # check numpy
    if isinstance(p, np.ndarray):
        return np.clip(p, a_min = 0, a_max = 1)
    
    # check torch
    if isinstance(p, torch.Tensor):
        return torch.clamp(p, min = 0, max = 1)

    # otherwise, type error
    raise ValueError(
        f'Expected p to be of type np.ndarray or torch.Tensor, '
        f'but got {type(p)}.'
    )

def adjust_sort(p: np.ndarray | torch.Tensor) -> Tuple[np.ndarray | torch.Tensor]:
    """Sort p-values in ascending order.
    
    Parameters
    ----------
    p : np.ndarray | torch.Tensor
        Vector of p-values.
    
    Returns
    -------
    sorted : np.ndarray | torch.Tensor
        Sorted vector of p-values.
    indices : np.ndarray | torch.Tensor
        Indices used to create the sorted vector.
    """
    
    # check numpy
    if isinstance(p, np.ndarray):
        indices = np.argsort(p)
        p = p[indices]
        return (p, indices)

    # check torch
    if isinstance(p, torch.Tensor):
        indices = torch.argsort(p)
        p = p[indices]
        return (p, indices)

    # otherwise, type error
    raise ValueError(
        f'Expected p to be of type np.ndarray or torch.Tensor, '
        f'but got {type(p)}.'
    )

def adjust_unsort(p: np.ndarray | torch.Tensor, indices: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Make unsorted vector from sorted p-values.
    
    Parameters
    ----------
    p : np.ndarray | torch.Tensor
        Vector of sorted p-values.
    indices : np.ndarray | torch.Tensor
        Vector of indices used to sort p-values.
    
    Returns
    -------
    p : np.ndarray | torch.Tensor
        Vector of unsorted p-values in original positions.
    """
    
    # check numpy
    if isinstance(p, np.ndarray):
        p_uns = np.empty_like(p)
        p_uns[indices] = p
        return p_uns
    
    # check torch
    if isinstance(p, torch.Tensor):
        p_uns = torch.empty_like(p)
        p_uns[indices] = p
        return p_uns
    
    # otherwise, type error
    raise ValueError(
        f'Expected p to be of type np.ndarray or torch.Tensor, '
        f'but got {type(p)}.'
    )

"""Registry and implementations.
"""

# setup methods
adjust_methods = {}

# register bonferroni
def adjust_bonferroni(p: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Perform Bonferroni correction.
    
    Parameters
    ----------
    p : np.ndarray | torch.Tensor
        Vector of p-values.
    
    Returns
    -------
    p : np.ndarray | torch.Tensor
        Vector of corrected p-values.
    """
    
    # check tests
    n_tests = p.shape[0]
    
    # correct
    return adjust_clip(p * n_tests)

adjust_methods['bonferroni'] = adjust_bonferroni

# register holm
def adjust_holm(p: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Perform Bonferroni-Holm correction.
    
    Parameters
    ----------
    p : np.ndarray | torch.Tensor
        Vector of p-values.
    
    Returns
    -------
    p : np.ndarray | torch.Tensor
        Vector of corrected p-values.
    """
    
    # check tests
    n_tests = p.shape[0]
    
    # sort values (ascending)
    p_sorted, p_indices = adjust_sort(p)
    
    # correct
    if isinstance(p, np.ndarray):
        p_corr = p_sorted * np.arange(n_tests, 0, -1)
        p_corr = np.maximum.accumulate(p_corr)
    elif isinstance(p, torch.Tensor):
        p_corr = p_sorted * torch.arange(n_tests, 0, -1, device = p.device)
        p_corr = torch.cummax(p_corr, dim = 0).values
    else:
        raise ValueError(
            f'Expected p to be of type np.ndarray or torch.Tensor, '
            f'but got {type(p)}.'
        )
    
    # unsort
    p_corr = adjust_unsort(p_corr, p_indices)
    
    return adjust_clip(p_corr)

adjust_methods['holm'] = adjust_holm

# register none
def adjust_none(p: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Perform no corrections.
    
    Parameters
    ----------
    p : np.ndarray | torch.Tensor
        Vector of p-values.
    
    Returns
    -------
    p : np.ndarray | torch.Tensor
        Vector of uncorrected p-values.
    """
    
    return p

adjust_methods['none'] = adjust_none

# register handler
def adjust_p(p: np.ndarray | torch.Tensor, method: str = 'holm') -> np.ndarray | torch.Tensor:
    """Perform correction of p-values.
    
    Parameters
    ----------
    p : np.ndarray | torch.Tensor
        Vector of p-values.
    method : str, default='holm'
        Type of correction to perform. One of {'holm', 'bonferroni', 'none'}.
    
    Returns
    -------
    p : np.ndarray | torch.Tensor
        Vector of corrected p-values.
    """
    
    # check method
    if method not in adjust_methods:
        raise ValueError(
            f'Unknown method {method}. Available options: ' +
            f'{[list(adjust_methods.keys())]}.'
        )
    
    return adjust_methods[method](p)

"""Miscallaneous corrections.
"""

def spearman_brown(X: float | np.ndarray | torch.Tensor, n: int = 2) -> float | np.ndarray | torch.Tensor:
    """Apply the Spearman-Brown correction to X.
    
    Parameters
    ----------
    X : float | np.ndarray | torch.Tensor
        Correlation(s) to be corrected.
    n : int, default=2
        Number of tests combined.
    
    Returns
    -------
    X : float | np.ndarray | torch.Tensor
        Spearman-Brown corrected correlation(s).
    """
    
    return (n * X) / (1 + (n - 1) * X)