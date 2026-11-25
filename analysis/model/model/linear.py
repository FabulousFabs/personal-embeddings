"""Linear network.

Contains the implementation of the default linear network that we 
present in the paper.
"""

from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict

class Linear(nn.Module):
    """Create a simple linear network.
    
    This is to probe the nature of individual semantic changes. This 
    class effectively models
    
        Z_S = X + A X + A_S X
    
    where A is our weight matrix. Unlike the delta network, therefore, 
    here we do not employ an encoder/latent/decoder structure and do 
    not allow the model to make any non-linear changes.
    
    Note that, principally, this class also supports biases. However, 
    be aware that these should be applied with great caution, as they 
    may be exploited by the network if metrics are cosine (as well as 
    normalised euclidean distances) by using their dependence on the 
    origin to make non-linear changes. When in doubt, leave them out.
    """
    
    def __init__(self, cfg: DictConfig, n_dim: int, n_subjects: int):
        """Initialise a linear network.
        
        Parameters
        ----------
        cfg : DictConfig
            Hydra configuration.
        n_dim : int
            Number of input dimensions.
        n_subjects : int
            Number of subjects.
        """
        
        super().__init__()
        
        # setup config
        self.cfg = cfg
        self.n_dim = n_dim
        
        # shared linear deformation
        if self.cfg.model.shared:
            self.shared = nn.Parameter(
                torch.zeros(
                    n_dim,
                    n_dim
                )
            )
            
            if self.cfg.model.bias:
                self.shared_b = nn.Parameter(
                    torch.zeros(
                        n_dim,
                    )
                )

        # participant-specific linear deformation
        self.individual = nn.Parameter(
            torch.zeros(
                n_subjects,
                n_dim,
                n_dim
            )
        )
        
        if self.cfg.model.bias:
            self.individual_b = nn.Parameter(
                torch.zeros(
                    n_subjects,
                    n_dim
                )
            )
        
    def forward(self, X: torch.Tensor, subject_ids: torch.Tensor, context: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute forward pass.
        
        Parameters
        ----------
        X : torch.Tensor
            Input data of shape `(n_batch, n_features)`.
        subject_ids : torch.Tensor
            Subject numbers of shape `(n_batch,)`.
        context : torch.Tensor
            Context data of shape `(n_batch, n_context, n_features)`.
        
        Returns
        -------
        y : torch.Tensor
            Output data of shape `(n_batch, n_features)`.
        """
        
        # prepare outputs
        outputs = {}

        # shared transformation
        if self.cfg.model.shared:
            s = torch.einsum(
                'b...d,kd->b...k',
                X,
                self.shared
            )
            
            if self.cfg.model.bias:
                s = s + self.shared_b

        # participant-specific transformation
        A = self.individual[subject_ids]

        h = torch.einsum(
            'b...d,bkd->b...k',
            X,
            A
        )
        
        if self.cfg.model.bias:
            h = h + self.individual_b[subject_ids]
        
        # Useful to retain these separately
        if self.cfg.model.shared:
            outputs['shared'] = X + s
        outputs['individual'] = h
        
        if self.cfg.model.shared:
            outputs['final'] = X + s + h
        else:
            outputs['final'] = X + h

        return outputs