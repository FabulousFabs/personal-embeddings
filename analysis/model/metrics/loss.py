"""Loss function implementations and utilities.
"""

from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn.functional as F

from mvpy.math import pearsonr_d
from .gram import get_gram_fn

from typing import Callable, Any, Dict

"""Loss implementations.
"""

def loss_mse(X: torch.Tensor, y: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Compute mean squared error.
    
    Parameters
    ----------
    X : torch.Tensor
        Input data.
    y : torch.Tensor
        Input data.
    **kwargs : Any
        Additional keyword arguments.
    
    Returns
    -------
    loss : torch.Tensor
        Loss in the form of mean-squared error.
    """
    
    return F.mse_loss(X, y)

def loss_simse(X: torch.Tensor, y: torch.Tensor, eps: float = 1e-6, **kwargs: Any) -> torch.Tensor:
    """Compute a scale-invariant mean squared error.
    
    Typically, mean squared errors are computed as:
    
    .. math::
    
        \\frac{1}{n}\\sum_{i=1}^{n}(X_i - y_i)^2
    
    However, if :math:`X` and :math:`y` are on different 
    scales, MSE may be dominated by such scale differences.
    If, however, we are interested in relative differences,
    we may want to ignore scale. To do this, here we normalise
    error terms. Effectively, this means that we assume that
    there is no information to be gained from the scale
    of individual boards in arrangement tasks.
    
    Parameters
    ----------
    X : torch.Tensor
        Input data.
    y : torch.Tensor
        Input data.
    eps : float, default=1e-6
        Error term to use in computations where zeros would lead to NaNs.
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    loss : torch.Tensor
        Loss in the form of a scale-invaraitn mean squared error.
    """
    
    # compute denominator
    denom = torch.sum(X ** 2).clamp_min(eps)
    
    # compute scaling factor
    scale = torch.sum(X * y) / denom
    
    return F.mse_loss(scale * X, y)

def loss_pearson(X: torch.Tensor, y: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Compute a pearson loss.
    
    Parameters
    ----------
    X : torch.Tensor
        Input data
    y : torch.Tensor
        Input data.
    **kwargs : Any
        Additional keyword arguments.
    
    Returns
    -------
    loss : torch.Tensor
        Loss in the form of pearson distance.
    """
    
    return pearsonr_d(X, y)

def loss_rank(X: torch.Tensor, y: torch.Tensor, samples: int = 64, tau: float = 1e-1, **kwargs: Any) -> torch.Tensor:
    """Compute a rank-based loss.
    
    Parameters
    ----------
    X : torch.Tensor
        Input data.
    y : torch.Tensor
        Input data.
    samples : int, default=64
        How many samples should be drawn per board?
    tau : float, default=1e-1
        Temperature of the softplus ranking.
    **kwargs : Any
        Additional keyword arguments.
    
    Returns
    -------
    loss : torch.Tensor
        Loss in the form of a ranked margins.
    """
    
    # if desired, compute rank loss
    n = X.numel()
    
    # check pair size
    if n < 2:
        return torch.tensor(0.0, dtype = X.dtype, device = X.device)

    # make permutation
    idx = torch.randint(0, n, (samples, 2), device = X.device)
    p = idx[:,0]
    q = idx[:,1]

    # check signs
    sign = torch.sign(y[p] - y[q]) # + if y[p] > y[q]
    keep = sign != 0
    
    # check available
    if keep.sum() == 0:
        return torch.tensor(0.0, dtype = X.dtype, device = X.device)
    
    # compute margins
    margin = sign[keep] * (X[p][keep] - X[q][keep])

    return F.softplus(-margin / tau).mean()

"""Registries and utility functions.
"""

# register losses
losses = {
    'mse': loss_mse,
    'simse': loss_simse,
    'pearson': loss_pearson,
    'rank': loss_rank
}

def get_loss_fn(cfg: DictConfig) -> Callable:
    """Obtain the desired loss function.
    
    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration.
    
    Returns
    -------
    fn : Callable
        Desired loss function.
    """
    
    # check validity
    if cfg.model.training.loss.geometrical.type not in losses:
        raise ValueError(
            f'Unknown loss type {cfg.model.training.loss.geometrical.type}. ' +
            f'Available losses: {list(losses.keys())}.'
        )
    
    return losses[cfg.model.training.loss.geometrical.type]

"""Full loss implementations.
"""

def loss_geometrical(X: torch.Tensor, y: torch.Tensor, inputs: Dict[str, torch.Tensor], cfg: DictConfig, norm_y: bool | None = None, ß: torch.Tensor | None = None) -> torch.Tensor:
    """Compute the geometrical loss between :math:`X` and :math:`y`.
    
    Parameters
    ----------
    X : torch.Tensor
        Input data of shape (n_batch, n_features).
    y : torch.Tensor
        Input data of shape (n_batch, n_features).
    inputs : Dict[str, torch.Tensor]
        Dictionary containing all model inputs.
    cfg : DictConfig
        Hydra configuration.
    norm_y : Optional[bool], default=None
        Should values in y be normalised before computing a gram matrix?
    ß : Optional[torch.Tensor], default=None
        If any, weigh the loss computed by the ß value of each subject.
    
    Returns
    -------
    loss : torch.Tensor
        The geometrical loss.
    """
    
    # grab subjects
    unq_subjects = torch.unique(inputs['subjects'])
    
    # setup loss
    loss = 0.0
    count = 0
    
    # setup functions
    gram_fn = get_gram_fn(cfg)
    loss_fn = get_loss_fn(cfg)
    
    # iterate over subjects
    for subject in unq_subjects:
        # subject mask
        mask_i = inputs['subjects'] == subject
        
        # grab boards
        unq_boards = torch.unique(inputs['boards'][mask_i])
        
        # setup loss
        loss_s = 0.0
        loss_n = 0
        
        # iterate over boards
        for board in unq_boards:
            # subject and board mask
            mask_j = mask_i & (inputs['boards'] == board)
            
            # skip if not useful
            if mask_j.sum() < 3:
                continue
            
            # grab data
            X_ij = X[mask_j]
            y_ij = y[mask_j]
            
            # compute gram matrices
            kwargs = {} if norm_y is None and cfg.metrics.gram == 'euclidean' else {'norm': False}
            D_X = gram_fn(X_ij)
            D_y = gram_fn(y_ij, **kwargs)
            
            # grab triangles from from gram
            tx, ty = torch.triu_indices(D_X.shape[0], D_X.shape[1], offset = 1)
            T_X = D_X[tx,ty]
            T_y = D_y[tx,ty]
            
            # compute loss
            kwargs = getattr(cfg.metrics.loss, cfg.model.training.loss.geometrical.type)
            kwargs = {} if kwargs is None else kwargs
            loss_sb = loss_fn(T_X, T_y, **kwargs)
            
            # board tally
            if ß is not None:
                loss_s += ß[subject] * loss_sb
            else:
                loss_s += loss_sb
            
            loss_n += 1
        
        # make loss
        loss_s = loss_s / max(loss_n, 1)
        loss += loss_s
        count += 1
    
    # normalise
    loss = loss / max(count, 1)
    
    return loss   
    
def loss(inputs: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor], cfg: DictConfig) -> torch.Tensor:
    """Compute the loss for our embedding model.
    
    Parameters
    ----------
    inputs : Dict[str, torch.Tensor]
        Dictionary of all input data.
    outputs : Dict[str, torch.Tensor]
        Dictionary of all output data.
    cfg : DictConfig
        Hydra configuration.
    
    Returns
    -------
    loss : torch.Tensor
        The full loss for model training.
    """
    
    # setup loss
    loss = 0.0
    
    # compute geometrical loss
    ß0, ß1 = (None, None) if not cfg.model.training.loss.geometrical.ceiling else (inputs['ß'], 1 - inputs['ß'])
    loss += cfg.model.training.loss.geometrical.lam * (
        loss_geometrical(outputs['final'], inputs['positions'], inputs, cfg, norm_y = False, ß = ß0) + 
        loss_geometrical(outputs['final'], inputs['embeddings'], inputs, cfg, ß = ß1)
    )
    
    # if available, compute L2 delta loss
    if 'delta' in outputs:
        loss += cfg.model.training.loss.delta.lam * (
            outputs['delta'].pow(2).mean()
        )
    
    # if available, compute L2 gate loss
    if 'weights' in outputs:
        loss += cfg.model.training.loss.attention.lam * (
            (outputs['weights'] - 1.0).pow(2).mean()
        )
    
    return loss