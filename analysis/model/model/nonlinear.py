"""Non-linear network.

Contains the implementation of the default non-linear network that we 
present in the paper.
"""

from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activation import make_activation

from typing import Dict

class NonLinear(nn.Module):
    """Create a simple non-linear network.
    
    This is to probe the nature of individual semantic changes. This 
    class effectively models
    
        Z_S = X + ϕ(A X + b) + ϕ(A_S X + b_S)
    
    where A is our weight matrix. Unlike the delta network, therefore, 
    this class does not employ an encoder/latent/decoder architecture, 
    but does allow the model to make non-linear changes.
    
    Note that, technically, this description is inaccurate given that 
    the network uses two hidden layers. The notation is, therefore, 
    more illustrative of the general approach, not the exact implemen-
    tation.
    """
    
    def __init__(self, cfg: DictConfig, n_dim: int, n_subjects: int):
        """Initialise a non-linear network.
        
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
        self.n_hidden = n_hidden = self.cfg.model.hidden
        
        # shared nonlinear transformation
        if self.cfg.model.shared:
            self.S1 = nn.Parameter(
                torch.empty(
                    n_hidden,
                    n_dim
                )
            )

            self.S2 = nn.Parameter(
                torch.zeros(
                    n_dim,
                    n_hidden
                )
            )
            
            self.B1 = nn.Parameter(
                torch.zeros(
                    n_hidden,
                )
            )
            
            self.B2 = nn.Parameter(
                torch.zeros(
                    n_dim,
                )
            )

        # participant-specific nonlinear transformation
        self.W1 = nn.Parameter(
            torch.empty(
                n_subjects,
                n_hidden,
                n_dim
            )
        )
        
        self.b1 = nn.Parameter(
            torch.zeros(
                n_subjects,
                n_hidden
            )
        )

        self.W2 = nn.Parameter(
            torch.zeros(
                n_subjects,
                n_dim,
                n_hidden
            )
        )
        
        self.b2 = nn.Parameter(
            torch.zeros(
                n_subjects,
                n_dim
            )
        )

        self.activation = make_activation(
            cfg.model.activation
        )()

        # same initialization for shared and individual input projections
        if self.cfg.model.shared:
            nn.init.normal_(
                self.S1,
                mean = 0.0,
                std = cfg.model.std
            )

        nn.init.normal_(
            self.W1,
            mean = 0.0,
            std = cfg.model.std
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

        # shared nonlinear deformation
        if self.cfg.model.shared:
            s = torch.einsum(
                'b...d,hd->b...h',
                X,
                self.S1
            )

            s = self.activation(s + self.B1)

            s = torch.einsum(
                'b...h,dh->b...d',
                s,
                self.S2
            ) + self.B2

        # participant-specific nonlinear deformation
        W1 = self.W1[subject_ids]
        W2 = self.W2[subject_ids]
        b1 = self.b1[subject_ids]
        b2 = self.b2[subject_ids]

        h = torch.einsum(
            'b...d,bhd->b...h',
            X,
            W1
        )

        h = self.activation(h + b1)

        h = torch.einsum(
            'b...h,bdh->b...d',
            h,
            W2
        ) + b2

        # residual output
        if self.cfg.model.shared:
            outputs['shared'] = X + s
        outputs['individual'] = h
        
        if self.cfg.model.shared:
            outputs['final'] = X + s + h
        else:
            outputs['final'] = X + h
        
        return outputs
