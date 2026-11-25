"""Effects.

Contains various effect size measures.
"""

import torch
import numpy as np

def one_sample_d(x: np.ndarray | torch.Tensor, popmean: float = 0.0) -> np.ndarray | torch.Tensor:
    """Compute Cohen's d for a one-sample test.
    
    Parameters
    ----------
    x : np.ndarray | torch.Tensor
        Vector of values.
    popmean : float, default=0.0
        Assumed population mean.
    
    Returns
    -------
    d : np.ndarray | torch.Tensor
        Effect size measure Cohen's d.
    """
    
    # compute difference
    d = x - popmean
    
    return d.mean() / x.std()

def dependent_samples_d(x: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Compute Cohen's d for dependent samples.
    
    Parameters
    ----------
    x : np.ndarray | torch.Tensor
        Vector of values.
    y : np.ndarray | torch.Tensor
        Vector of baseline values.
    
    Returns
    -------
    d : np.ndarray | torch.Tensor
        Effect size measure Cohen's d.
    """
    
    # compute difference
    d = x - y
    
    return d.mean() / d.std()