"""Delta network.

Contains the implementation of the default delta network that we 
present in the paper.
"""

from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activation import make_activation
from .latent import make_latent
from .attention import make_attention
from .projection import Encoder, Decoder

from typing import Dict

class Delta(nn.Module):
    """Create a delta network.
    
    Given some behevioural data that is some function of the true latent
    semantic representations of subjects B = f(Z), corresponding word 
    embeddings E, context embeddings C and subject S, we want to find 
    some function φ(E|C,S) = Z_S. Because we have relatively few con-
    straints in our data (i.e., little training data), we here use a 
    trick in modelling this such that:

        Z_S = E + Δ(E|S)
    
    and, if desired, \hat{Z}_{S|C} (i.e., the individual representa-
    tion given the board context) as:

        \hat{Z}_{S|C} = f(C|S) Z_S
    
    where f(C|S) is some attention function.
    
    Attributes
    ----------
    cfg : DictConfig
        Hydra configuration.
    n_dim : int
        Number of input dimensions.
    n_subjects : int
        Number of subjects.
    encoder : nn.Module | None
        If applied, the encoder submodule.
    latent : nn.ModuleList | None
        If applied, the latent submodules.
    decoder : nn.Module | None
        If applied, the decoder submodule.
    attention : nn.Module | None
        If applied, the attention submodule.
    """
    
    def __init__(self, cfg: DictConfig, n_dim: int, n_subjects: int):
        """Initialise a delta network.
        
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
        
        # track latent dims
        n_latent = n_dim
        
        # if required, setup encoder
        if self.cfg.model.encoder.enable:
            self.encoder = Encoder(self.cfg, n_latent)
            n_latent = self.cfg.model.encoder.layers[-1]
        
        # if required, setup latent layers
        if self.cfg.model.latent.enable:
            self.latent = nn.ModuleList([
                make_latent(self.cfg.model.latent.layers[layer].type)(
                    self.cfg.model.latent.layers[layer], n_latent, n_subjects
                )
                for layer in self.cfg.model.latent.layers
            ])
        
        # if required, setup decoder
        if self.cfg.model.decoder.enable:
            self.decoder = Decoder(self.cfg, n_dim)
        
        # if required, setup attention
        if self.cfg.model.attention.enable:
            self.attention = make_attention(self.cfg.model.attention.type)(
                self.cfg, n_dim, n_subjects
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
        
        # setup outputs
        outputs = {}
        
        # set current
        current = X
        
        # if required, encode inputs
        if self.cfg.model.encoder.enable:
            outputs['latent'] = self.encoder(X)
            current = outputs['latent'].clone()
        
        # if required, encode contexts
        if self.cfg.model.encoder.enable and self.cfg.model.attention.enable and self.cfg.model.attention.latent:
            # encode contexts
            outputs['latent_context'] = self.encoder(context)
            current_context = outputs['latent_context'].clone()
        
            # reshape contexts
            shape = current_context.shape
            context_vec = current_context.clone().reshape((-1, current_context.shape[-1]))
            subject_vec = subject_ids.repeat(current_context.shape[1], 1).t().reshape((-1,))
        
        # if required, run latent
        if self.cfg.model.latent.enable:
            for layer in self.latent:
                # for inputs
                current = layer(current, subject_ids)
                
                # if required, for contexts
                if self.cfg.model.encoder.enable and self.cfg.model.attention.enable and self.cfg.model.attention.latent:
                    context_vec = layer(context_vec, subject_vec)
        
        # if required, reshape contexts
        if self.cfg.model.attention.latent:
            current_context = context_vec.reshape(shape)
        
        # if required, decode outputs
        if self.cfg.model.decoder.enable:
            outputs['delta'] = self.decoder(current)
            current = X + outputs['delta']
        else:
            current = X + current
        
        # if required, decode contexts
        if self.cfg.model.decoder.enable and self.cfg.model.attention.enable and self.cfg.model.attention.latent:
            outputs['delta_context'] = self.decoder(current_context)
            context = context + outputs['delta_context']

        # if required, apply attention
        if self.cfg.model.attention.enable:
            current = self.attention(current, context, subject_ids)
        
        # make final output
        outputs['final'] = current
        
        return outputs