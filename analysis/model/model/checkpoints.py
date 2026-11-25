"""Checkpoint and utility functions.

Contains functions that help with checkpointing and saving of 
models and associated data.
"""

from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn as nn
from pathlib import Path

from typing import Tuple, Dict

def make_path_identifier(cfg: DictConfig, name: str, repeat: int, split: int, permutation: int, model: str) -> Tuple[Path, str]:
    """Make a path and identifier for a given model.
    
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
    model : str
        Name of the model used.
    
    Returns
    -------
    path : Path
        The subdirectory used for this model.
    identifier : str
        The identifier for outputs from this model.
    """
    
    # make root path
    root_path = Path(cfg.data.path) / Path('derivatives') / Path(f'run-{cfg.run:03d}')

    # make subdirectories
    sub_path = root_path / Path(f'rep-{repeat:03d}') / Path(f'spt-{split:03d}') / Path(f'prm-{permutation:03d}')
    sub_path.mkdir(parents = True, exist_ok = True)
    
    # make identifier
    identifier = f'rep-{repeat:03d}_spt-{repeat:03d}_prm-{permutation:03d}_bse-{name}_mod-{model}'
    
    return sub_path, identifier

def save(model: nn.Module, cfg: DictConfig, name: str, repeat: int, split: int, permutation: int, extension: str = '') -> None:
    """Save a trained model.
    
    Parameters
    ----------
    model : nn.Module
        The model to be saved.
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
    extension : str, default=''
        Extension for the model name.
    """
    
    # get path and identifier
    path, identifier = make_path_identifier(
        cfg, name, repeat, split, permutation, cfg.model.name + extension
    )
    
    # make path
    f_path = path / Path(f'{identifier}_mod.pt')
    
    # save data
    torch.save(
        model.state_dict(),
        f_path
    )

def save_supplement(data: Dict[str, torch.Tensor | int | float], cfg: DictConfig, name: str, repeat: int, split: int, permutation: int, supplement: str, extension: str = '') -> None:
    """Save outputs as a supplementary file.
    
    Parameters
    ----------
    data : Dict[str, torch.Tensor | int | float]
        The output dictionary to be saved.
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
    supplement : str
        Name of this supplement.
    extension : str, default=''
        Extension for the model name.
    """
    
    # get path and identifier
    path, identifier = make_path_identifier(
        cfg, name, repeat, split, permutation, cfg.model.name + extension
    )
    
    # make path
    f_path = path / Path(f'{identifier}_supp-{supplement}.pt')
    
    # save data
    torch.save(
        data,
        f_path
    )