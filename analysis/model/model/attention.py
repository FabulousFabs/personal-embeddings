"""Attention layers.

Contains attention layer specifications, registries and helper functions 
to create attention layers in the delta network.
"""

from omegaconf import DictConfig, OmegaConf

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .activation import make_activation
from .latent import make_latent

"""Attention implementations.
"""

class PooledAttention(nn.Module):
    """Create a basic subject-specific attention layer.
    
    Here, attention is simply formalised as pooling the
    context into one average vector before applying one
    of the latent layers as a feature gating mechanism.
    
    Attributes
    ----------
    cfg : DictConfig
        Hydra configuration.
    n_dim : int
        Number of input dimensions.
    n_subjects : int
        Number of subjects.
    gate : nn.Module
        The latent gating mechanism.
    """
    
    def __init__(self, cfg: DictConfig, n_dim: int, n_subjects: int):
        """Initialise an attention layer.
        
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
        self.n_subjects = n_subjects
        
        # setup feature gate
        self.gate = make_latent(self.cfg.model.attention.gate.type)(
            self.cfg.model.attention.gate, self.cfg.model.attention.hidden, n_subjects,
            n_hidden = n_dim
        )
    
    def forward(self, X: torch.Tensor, Z: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        """Compute forward pass.
        
        Parameters
        ----------
        X : torch.Tensor
            Input data of shape `(n_batch, n_features)`.
        Z : torch.Tensor
            Context data of shape `(n_batch, n_context, n_features)`.
        subject_ids : torch.Tensor
            Subject numbers of shape `(n_batch,)`.
        
        Returns
        -------
        y : torch.Tensor
            Output data of shape `(n_batch, n_features)`.
        """
        
        # pool context
        h = Z.mean(1) # (n_batch, n_features)
        
        # normalise context vector
        h = F.normalize(h, dim = -1) # (n_batch, n_features)
        
        # compute gated feature weights
        weight = self.gate(h, subject_ids) # (n_batch, n_features)
        weight = torch.sigmoid(weight)
        
        # if training, perform a final dropout over items
        # this is to encourage the (harder to learn) latent
        # states to receive more airtime
        if self.training:
            p = torch.rand((weight.shape[0],), device = X.device)
            p = (p <= self.cfg.model.attention.deactivate).to(X.dtype).unsqueeze(1)
            weight = (1.0 - p) * weight + p
        
        return X * weight

class SelfAttention(nn.Module):
    """Create a subject-specific attention layer.
    
    Here, attention is formalised as a small QKV self-attention layer,
    where Q is instantiated from a latent layer, followed by a latent
    feature gating layer. For more information on latent layers, see 
    :py:meth:`~make_latent`.
    
    Attributes
    ----------
    cfg : DictConfig
        Hydra configuration.
    n_dim : int
        Number of input dimensions.
    n_subjects : int
        Number of subjects.
    Q : nn.Module
        Query layer (instantiated as a latent layer).
    K : nn.Module
        Key layer (instantiated as linear).
    V : nn.Module
        Value layer (instantiated as linear).
    dropout : nn.Module
        Feature dropout for attention.
    gate : nn.Module
        Gating layer (instantiated as latent layer).
    """
    
    def __init__(self, cfg: DictConfig, n_dim: int, n_subjects: int):
        """Initialise an attention layer.
        
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
        self.n_subjects = n_subjects
        
        # setup context attention
        self.Q = make_latent(self.cfg.model.attention.Q.type)(
            self.cfg.model.attention.Q, n_dim, n_subjects, 
            n_hidden = self.cfg.model.attention.hidden
        )
        
        self.K = nn.Linear(n_dim, self.cfg.model.attention.hidden)
        self.V = nn.Linear(n_dim, self.cfg.model.attention.hidden)
        self.dropout = nn.Dropout(self.cfg.model.attention.dropout)
        
        # setup feature gate
        self.gate = make_latent(self.cfg.model.attention.gate.type)(
            self.cfg.model.attention.gate, self.cfg.model.attention.hidden, n_subjects,
            n_hidden = n_dim
        )
    
    def forward(self, X: torch.Tensor, Z: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        """Compute forward pass.
        
        Parameters
        ----------
        X : torch.Tensor
            Input data of shape `(n_batch, n_features)`.
        Z : torch.Tensor
            Context data of shape `(n_batch, n_context, n_features)`.
        subject_ids : torch.Tensor
            Subject numbers of shape `(n_batch,)`.
        
        Returns
        -------
        y : torch.Tensor
            Output data of shape `(n_batch, n_features)`.
        """
        
        # set d
        d = Z.shape[-1]
        
        # compute QKV
        Q = self.Q(X, subject_ids) # (n_batch, n_hidden)
        K = self.K(Z) # (n_batch, n_words, n_hidden)
        V = self.V(Z) # (n_batch, n_words, n_hidden)
        
        # compute scores
        scores = (K * Q[:,None,:]).sum(-1) / math.sqrt(d) # (n_batch, n_words)
        
        # compute weights
        w = F.softmax(scores, dim = 1) # (n_batch, n_words)
        w = self.dropout(w)

        # compute attended context
        h = (w[:,:,None] * V).sum(dim = 1) # (n_batch, n_hidden)
        
        # compute gated feature weights
        weight = self.gate(h, subject_ids) # (n_batch, n_features)
        weight = torch.sigmoid(weight)
        
        # if training, perform a final dropout over items
        # this is to encourage the (harder to learn) latent
        # states to receive more airtime
        if self.training:
            p = torch.rand((weight.shape[0],), device = X.device)
            p = (p <= self.cfg.model.attention.deactivate).to(X.dtype).unsqueeze(1)
            weight = (1.0 - p) * weight + p
        
        return X * weight

"""Registry and helper functions.
"""

# make attention registry
attention_layers = {
    'pooled' : PooledAttention,
    'self': SelfAttention
}

def make_attention(name: str) -> nn.Module:
    """Make a latent layer.
    
    Parameters
    ----------
    name : str
        Type of latent layer, as specified in :py:attr:`~attention_layers`.
    
    Returns
    -------
    layer : nn.Module
        The requested layer type.
    """
    
    # check entry
    if name.lower() not in attention_layers:
        raise ValueError(
            f'`{name}` unknown for attention layers.'
        )
    
    return attention_layers[name.lower()]