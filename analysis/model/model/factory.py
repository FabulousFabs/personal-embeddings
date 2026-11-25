"""Factory utilities.

Contains registries and utility functions for model creation and loading.
"""

from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from .checkpoints import make_path_identifier

from .delta import Delta
from .linear import Linear
from .nonlinear import NonLinear

# make registry of models
models = {
    'delta': Delta,
    'linear': Linear,
    'nonlinear': NonLinear
}

def create(cfg: DictConfig, name: str, repeat: int, split: int, permutation: int, dims: int, subjects: int, extension: str = '', preload: bool = False, force_model: str = None) -> nn.Module:
    """Create a new model.
    
    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration.
    name : str
        Embedding used in this model.
    repeat : int
        Cross-validation repetition number.
    split : int
        Cross-validation split number.
    permutation : int
        Cross-validation permutation number.
    dims : int
        Number of input dimensions of the embedding.
    subjects : int
        Number of subjects.
    extension : str, default=''
        Extension for the model name.
    preload : bool, default=False
        If the model already exists, should weights be preloaded?
    force_model : str | None, default=None
        Should we force a specific model type, overriding the configuration?
    
    Returns
    -------
    model : nn.Module
        The desired model that, optionally, may have preloaded weights.
    """
    
    # check model type
    model_name = cfg.model.name
    if force_model is not None:
        model_name = force_model
    
    # check registration
    if model_name not in models:
        raise ValueError(
            f'Unknown model type {model_name}. Supported ' + 
            f'types: {list(models.keys())}.'
        )
    
    # create model
    model = models[model_name](
        cfg, dims, subjects
    )
    
    # get path and identifier
    path, identifier = make_path_identifier(
        cfg, name, repeat, split, permutation, model_name + extension
    )
    
    # if required, try to preload a model
    if preload:
        # check for existing model
        f_path = path / Path(f'{identifier}_mod.pt')

        if not f_path.exists():
            raise ValueError(
                f'Model requested `{f_path}` does not exist.'
            )
        
        # load model
        model.load_state_dict(
            torch.load(
                f_path, 
                weights_only = True,
                map_location = torch.device(cfg.device)
            )
        )
    
    return model.to(
        cfg.device
    )