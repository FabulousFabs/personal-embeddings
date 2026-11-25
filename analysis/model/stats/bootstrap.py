"""Bootstrapping.

Contains various bootstrapping methods.
"""

import torch
import numpy as np

from typing import Tuple

def bootstrap_se(X: np.ndarray | torch.Tensor, n_samples: int = 10_000) -> np.ndarray | torch.Tensor:
    """Estimate standard error by bootstrapping.
    
    Parameters
    ----------
    X : np.ndarray | torch.Tensor
        Vector of values to compute standard error over.
    n_samples : int, default=10_000
        Number of samples to use for bootstrapping.
    
    Returns
    -------
    se : np.ndarray | torch.Tensor
        Standard error of X estimated via bootstrap.
    """
    
    # check dims
    n_dim = X.shape[0]
    
    # check numpy
    if isinstance(X, np.ndarray):
        # compute bootstrap
        return X[
            np.random.choice(
                np.arange(n_dim),
                size = (n_samples, n_dim),
                replace = True
            )
        ].mean(1).std(0)
    
    # otherwise, check torch
    if isinstance(X, torch.Tensor):
        # compute bootstrap
        return X[
            torch.randint(
                low = 0, high = n_dim,
                size = (n_samples, n_dim),
                dtype = torch.int64,
                device = X.device
            )
        ].mean(1).std(0)
    
    # otherwise, unknown datatype
    raise ValueError(
        f'X must be either np.ndarray or torch.Tensor, '
        f'but got {type(X)}.'
    )

def prepare_bootstrap_se_within(X: np.ndarray | torch.Tensor, n_samples: int = 10_000) -> np.ndarray | torch.Tensor:
    """Correct data for within-subject design in bootstrapping.
    
    Briefly, typically CIs are computed in a between-subject 
    fashion. This is unlike it would be computed in, for ex-
    ample, an ANOVA where we are interested in within-subject 
    effects. We can, however, remove subject-variance from the
    data before computing the CI to facilitate interpretation 
    of the resulting CI. For details of this procedure, please 
    consult [1].
    
    Parameters
    ----------
    X : np.ndarray | torch.Tensor
        Vector of values to prepare.
    n_samples : int, default=10_000
        Number of samples to use for bootstrapping.
    
    Returns
    -------
    Z : np.ndarray | torch.Tensor
        Within-subject corrected data for bootstrapping down the line.
    
    References
    ----------
    [1] Loftus, G.R., & Masson, M.E.J. (1994). Using confidence intervals in within-subject designs. Psychonomic Bulleting & Review, 1, 476-490. 10.3758/BF03210951
    """
    
    # compute deviation score
    d_m = X.mean(0) - X.mean()
    
    # correct data
    X_d = X - d_m[None,:]

    # return corrected data
    return X_d

def bootstrap_se_ratio(x: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor, n_samples: int = 10_000) -> np.ndarray | torch.Tensor:
    """Estimate standard error of a ratio by bootstrapping.
    
    Parameters
    ----------
    X : np.ndarray | torch.Tensor
        Vector of values where we estimate SE(X / y).
    y : np.ndarray | torch.Tensor
        Vector of values where we estimate SE(X / y).
    n_samples : int, default=10_000
        Number of samples to use for bootstrapping.
    
    Returns
    -------
    se : np.ndarray | torch.Tensor
        Standard error of X / y estimated via bootstrap, expressed as a percentage.
    """
    
    # check dims
    n_dim = x.shape[0]
    
    # check numpy
    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        # take bootstrap
        B = np.random.choice(
            np.arange(n_dim),
            size = (n_samples, n_dim),
            replace = True
        )
    elif isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
        # compute bootstrap
        B = torch.randint(
            low = 0, high = n_dim,
            size = (n_samples, n_dim),
            dtype = torch.int64,
            device = x.device
        )
    else:
        # otherwise, unknown datatype
        raise ValueError(
            f'x and y must be either both np.ndarray or both torch.Tensor, '
            f'but got {type(x)} and {type(y)}.'
        )
    
    # compute bootstrap
    R = 100.0 * (x[B].mean(1) / y[B].mean(1) - 1.0)
    
    return R.std(0)