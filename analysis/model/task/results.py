"""Results.

Contains helper functions for handling results.
"""

from omegaconf import DictConfig, OmegaConf
from loguru import logger

import torch

from pathlib import Path

from typing import Tuple, List, Dict

def make_path_identifier(cfg: DictConfig, task: str, name: List[str], model: str) -> Tuple[Path, str]:
    """Make a path and identifier for a given task.
    
    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration.
    task : str
        Name of the task.
    name : List[str]
        List of embeddings used in this task.
    model : str
        Name of the model used in this task.
    
    Returns
    -------
    path : Path
        The subdirectory used for this task.
    identifier : str
        The identifier for outputs from this task.
    """
    
    # make root path
    root_path = Path(cfg.data.path) / Path('derivatives') / Path(f'run-{cfg.run:03d}')
    
    # make subdirectories
    sub_path = root_path / Path('tasks')
    sub_path.mkdir(parents = True, exist_ok = True)
    
    # make embedding id
    emb = '_'.join(
        f'bse{i}-{base}'
        for i, base in enumerate(name)
    )
    
    # make identifier
    identifier = f'task-{task}_{emb}_mod-{model}'
    
    return sub_path, identifier

def save_supplement(data: Dict[str, torch.Tensor | int | float], task: str, cfg: DictConfig, logger: logger) -> None:
    """Save outputs as a supplementary file.
    
    Parameters
    ----------
    data : Dict[str, torch.Tensor | int | float]
        The output dictionary to be saved.
    task : str
        Name of the task.
    cfg : DictConfig
        Hydra configuration.
    logger : logger
        The logger object.
    """
    
    # get path and identifier
    path, identifier = make_path_identifier(
        cfg, task, cfg.data.embeddings.models, cfg.model.name
    )
    
    # make path
    f_path = path / Path(f'{identifier}.pt')
    
    # save data
    torch.save(
        data,
        f_path
    )

def load_supplement(task: str, cfg: DictConfig, logger: logger) -> Dict[str, torch.Tensor | int | float]:
    """Load outputs from supplementary file.
    
    Parameters
    ----------
    task : str
        Name of the task.
    cfg : DictConfig
        Hydra configuration.
    logger : logger
        The logger object.
    
    Returns
    -------
    data : Dict[str, torch.Tensor | int | float]
        The output dictionary to be saved.
    """
    
    # get path and identifier
    path, identifier = make_path_identifier(
        cfg, task, cfg.data.embeddings.models, cfg.model.name
    )
    
    # make path
    f_path = path / Path(f'{identifier}.pt')
    
    # load data
    return torch.load(f_path, weights_only = True)