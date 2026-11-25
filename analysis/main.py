"""Entry point for all analysis.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger

import model

@hydra.main(config_path = 'conf', config_name = 'config', version_base = None)
def main(cfg: DictConfig) -> None:
    """Start analysis workflow.
    
    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration describing workflow.
    """
    
    # load dataset
    data = model.dataset.Dataset(
        cfg, logger
    ).load()
    
    # start working on tasks
    for name in cfg.task:
        # grab task
        task = cfg.task[name]
        
        if task.skip:
            continue
        
        # perform task
        model.task.perform(
            name, task, data, cfg, logger
        )

if __name__ == '__main__':
    main()