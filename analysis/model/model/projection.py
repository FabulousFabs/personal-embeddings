"""Projection layers.

Contains barebones auto-encoder style projections.
"""

from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activation import make_activation

"""Projection implementations.
"""

class Encoder(nn.Module):
    """Create an encoder for an autoencoder-style bottleneck architecture.
    
    Attributes
    ----------
    cfg : DictConfig
        Parsed hydra configuration.
    n_dim : int
        Input dimensions.
    projection : nn.Sequential
        The projection that takes (n_batch, n_dim) and returns (n_batch, n_latent).
    """
    
    def __init__(self, cfg: DictConfig, n_dim: int):
        """Initialise the encoder.
        
        Parameters
        ----------
        cfg : DictConfig
            Parsed hydra configuration.
        n_dim : int
            Input dimensions
        """
        
        super().__init__()
        
        # setup config
        self.cfg = cfg
        self.n_dim = n_dim
        
        # setup layers
        inputs = [n_dim, *self.cfg.model.encoder.layers[:-1]]
        outputs = [*self.cfg.model.encoder.layers]
        layers = []
                
        for i, (ins, outs) in enumerate(zip(inputs, outputs)):
            # add layer
            layers += [
                nn.Dropout(self.cfg.model.encoder.dropout),
                nn.Linear(ins, outs)
            ]
            
            # check activation
            if i < (len(inputs) - 1):
                # add activation
                layers += [
                    make_activation(self.cfg.model.encoder.activation)()
                ]
        
        # make sequential model
        self.projection = nn.Sequential(
            *layers
        )
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Compute an encoder forward pass.
        
        Parameters
        ----------
        X : torch.Tensor
            Input data of shape (n_batch, n_dim).
        
        Returns
        -------
        y : torch.Tensor
            Projected data of shape (n_batch, n_latent).
        """
        
        return self.projection(X)

class Decoder(nn.Module):
    """Create a decoder for an autoencoder-style bottleneck architecture.
    
    Attributes
    ----------
    cfg : DictConfig
        Parsed hydra configuration.
    n_dim : int
        Output dimensions.
    projection : nn.Sequential
        The projection that takes (n_batch, n_latent) and returns (n_batch, n_dim).
    """
    
    def __init__(self, cfg: DictConfig, n_dim: int):
        """Initialise the decoder.
        
        Parameters
        ----------
        cfg : DictConfig
            Parsed hydra configuration.
        n_dim : int
            Input dimensions
        """
        
        super().__init__()
        
        # setup config
        self.cfg = cfg
        self.n_dim = n_dim
        
        # setup layers
        inputs = [*self.cfg.model.decoder.layers, self.cfg.model.decoder.layers[-1]]
        outputs = [*self.cfg.model.decoder.layers[1:], n_dim]
        layers = []
        
        for i, (ins, outs) in enumerate(zip(inputs, outputs)):
            # add layer
            layers += [
                nn.Dropout(self.cfg.model.decoder.dropout),
                nn.Linear(ins, outs)
            ]
            
            # check activation
            if i < (len(inputs) - 1):
                # add activation
                layers += [
                    make_activation(self.cfg.model.decoder.activation)()
                ]
        
        # make sequential model
        self.projection = nn.Sequential(
            *layers
        )
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Compute a decoder forward pass.
        
        Parameters
        ----------
        X : torch.Tensor
            Input data of shape (n_batch, n_latent).
        
        Returns
        -------
        y : torch.Tensor
            Projected data of shape (n_batch, n_dim).
        """
        
        return self.projection(X)