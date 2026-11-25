"""Factory.

Contains registry and performance helpers for all tasks.
"""

from omegaconf import DictConfig, OmegaConf
from loguru import logger

from ..dataset import Dataset
from .training import arrangement_train, arrangement_cross
from .reliability import arrangement_reliability, arrangement_crossreliability
from .demographics import arrangement_subemb, arrangement_subsub
from .space import arrangement_space

# setup task registry
tasks = {
    'mareliability': arrangement_reliability,
    'macrossreliability': arrangement_crossreliability,
    'matrain': arrangement_train,
    'macross': arrangement_cross,
    'masubemb': arrangement_subemb,
    'masubsub': arrangement_subsub,
    'maspace': arrangement_space,
}

def perform(name: str, task: DictConfig, data: Dataset, cfg: DictConfig, logger: logger) -> None:
    """Perform one prespecified task.
    
    Parameters
    ----------
    name : str
        Name of the task to perform.
    task : DictConfig
        Hydra subconfiguration for this specific task.
    data : Dataset
        The preloaded dataset.
    cfg : DictConfig
        Hydra configuration.
    logger : logger
        The logger object.
    """
    
    # check validity
    if name not in tasks:
        raise ValueError(
            f'Task `{name}` is unknown.'
        )
    
    # hand off to dedicated function
    tasks[name](
        task, data, cfg, logger
    )