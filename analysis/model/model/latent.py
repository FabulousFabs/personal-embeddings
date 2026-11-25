"""Latent layers

Contains latent layer specifications, registries and helper functions 
to create latent layers in the delta network.

Notes
-----
Layer implementations require manual registration in :py:attr:`~latent_layers` 
to be accessible for delta networks, which use :py:meth:`~make_latent` when 
instantiating latents.
"""

from omegaconf import DictConfig, OmegaConf

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .activation import make_activation

"""Layer implementations.
"""

class LatentDiagonal(nn.Module):
    """Create a subject-specific diagonal adapter.
    
    The diagonal adapter is a simple vector of weights 
    per subject that is applied to the latent represen-
    tations, constituting a strongly constrained trans-
    formation of the latent inputs.
    
    Attributes
    ----------
    cfg : DictConfig
        Hydra configuration
    n_dim : int
        Number of input dimensions.
    n_subjects : int
        Number of subjects.
    n_hidden : int | None, default=None
        If any, number of hidden dimensions to use. If `None`, py:attr:`~n_dim` will be used.
    dropout : nn.Module
        Feature dropout module.
    base : nn.Module
        Linear base layer.
    activation : nn.Module
        Activation function.
    diag : nn.Module
        Diagonal layer.
    """
    
    def __init__(self, cfg: DictConfig, n_dim: int, n_subjects: int, n_hidden: int | None = None):
        """Initialise a subject-specific diagonal layer.
        
        Parameters
        ----------
        cfg : DictConfig
            Hydra configuration.
        n_dim : int
            Number of input dimensions.
        n_subjects : int
            Number of subjects.
        n_hidden : int | None, default=None
            If any, number of hidden dimensions to use. If `None`, :py:attr:`~n_dim` will be used.
        """
        
        super().__init__()
        
        # setup config
        self.cfg = cfg
        self.n_dim = n_dim
        self.n_subjects = n_subjects
        self.n_hidden = n_hidden if n_hidden is not None else n_dim
        
        # setup shared projection
        self.dropout = nn.Dropout(cfg.linear.dropout)
        self.base = nn.Linear(n_dim, self.n_hidden)
        self.activation = make_activation(cfg.linear.activation)()
        
        # setup diagonals
        if cfg.linear.per_subject:
            self.diag = nn.Parameter(torch.empty(n_subjects, self.n_hidden))
        else:
            self.diag = nn.Parameter(torch.empty(self.n_hidden,))
        nn.init.kaiming_uniform_(self.diag, a = math.sqrt(5))
    
    def forward(self, X: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass.
        
        Parameters
        ----------
        X : torch.Tensor
            Input data of shape `(n_batch, n_features)`.
        subject_ids : torch.Tensor
            Subject numbers of shape `(n_batch,)`.
        
        Returns
        -------
        y : torch.Tensor
            Outut data of shape `(n_batch, n_features)`.
        """
        
        # linear layer (n_batch, n_features)
        y = self.base(
            self.dropout(X)
        )
        
        # compute diagonals
        if self.cfg.linear.per_subject:
            W = self.diag.index_select(0, subject_ids)
        else:
            W = self.diag.unsqueeze(0).expand(X.shape[0], *self.diag.shape)
        
        # add diagonals
        y = self.activation(W * y)
        
        return y

class LatentLoRA(nn.Module):
    """Create a subject-specific low-rank adapter.
    
    This includes both a shared linear layer as well as, if desired,
    all subject specific adapters. Contrary to LoRA elsewhere, here
    we train both the shared and LoRA parts concurrently.
    
    Attributes
    ----------
    cfg : DictConfig
        Hydra configuration.
    n_dim : int
        Number of input dimensions.
    n_subjects : int
        Number of subjects.
    n_hidden: int | None, default=None
        If any, number of hidden dimensions to use. If `None`, :py:attr:`~n_dim` will be used.
    dropout : nn.Module
        Feature dropout module.
    linear : nn.Module
        Shared linear projection.
    activation : nn.Module
        Shared activation function.
    rank : int
        Rank of adapters.
    alpha : float
        Coefficient determining scaling magnitude.
    scaling : float
        Precomputed scaling factor.
    A_dims : Tuple[int]
        Dimensions of A matrix for LoRA.
    B_dims : Tuple[int]
        Dimensions of B matrix for LoRA
    dropout_lora : nn.Module
        Feature dropout for LoRA.
    A : nn.Parameter
        A matrices of shape `(n_subjects, n_rank, n_dim)` or `(n_rank, n_dim)`.
    B : nn.Parameter
        B matrices of shape `(n_subjects, n_hidden, n_rank)` or `(n_hidden, n_rank)`.
    """
    
    def __init__(self, cfg: DictConfig, n_dim: int, n_subjects: int, n_hidden: int | None = None):
        """Initialise a subject-specific LoRA layer.
        
        Parameters
        ----------
        cfg : DictConfig
            Hydra configuration.
        n_dim : int
            Number of input dimensions.
        n_subjects : int
            Number of subjects.
        n_hidden : int | None, default=None
            If any, number of hidden dimensions to use. If `None`, :py:attr:`~n_dim` will be used.
        """
        
        super().__init__()
        
        # setup config
        self.cfg = cfg
        self.n_dim = n_dim
        self.n_subjects = n_subjects
        self.n_hidden = n_hidden if n_hidden is not None else n_dim

        # setup shared layer
        self.dropout = nn.Dropout(cfg.linear.dropout)
        self.linear = nn.Linear(self.n_dim, self.n_hidden)
        self.activation = make_activation(cfg.linear.activation)()
        
        # setup LoRA parameters
        self.rank = self.cfg.lora.rank
        self.alpha = self.cfg.lora.alpha
        self.scaling = self.alpha / max(1, self.rank)
        
        # setup LoRA dimensions
        if self.cfg.lora.per_subject:
            self.A_dims = (n_subjects, self.rank, self.n_dim)
            self.B_dims = (n_subjects, self.n_hidden, self.rank)
        else:
            self.A_dims = (self.rank, self.n_dim)
            self.B_dims = (self.n_hidden, self.rank)
        
        # setup LoRA
        self.dropout_lora = nn.Dropout(cfg.lora.dropout)
        self.A = nn.Parameter(torch.empty(*self.A_dims))
        self.B = nn.Parameter(torch.empty(*self.B_dims))
        
        # initialise LoRA
        nn.init.kaiming_uniform_(self.A, a = math.sqrt(5))
        nn.init.zeros_(self.B)
    
    def forward(self, X: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass.
        
        Parameters
        ----------
        X : torch.Tensor
            Input data of shape `(n_batch, n_features)`.
        subject_ids : torch.Tensor
            Subject numbers of shape `(n_batch,)`.
        
        Returns
        -------
        y : torch.Tensor
            Outut data of shape `(n_batch, n_features)`.
        
        Notes
        -----
        It would be computationally equivalent, but more efficient to use
        the weight formulation of LoRA here. For speed optimisations, this 
        should probably be implemented at some point.
        """
        
        # linear layer (n_batch, n_features)
        y = self.linear(
            self.dropout(X)
        )
        
        # prepare LoRA
        x_d = self.dropout_lora(X)
        
        if self.cfg.lora.per_subject:
            # make individual adapters
            A = self.A.index_select(0, subject_ids) # (n_batch, rank, n_dim)
            B = self.B.index_select(0, subject_ids) # (n_batch, n_hidden, rank)
        else:
            # make shared adapter
            A = self.A.unsqueeze(0).expand(X.shape[0], *self.A.shape) # (n_batch, rank, n_dim)
            B = self.B.unsqueeze(0).expand(X.shape[0], *self.B.shape) # (n_batch, n_hidden, rank)
        
        # compute LoRA
        y_A = torch.bmm(A, x_d.unsqueeze(-1)).squeeze(-1) # (n_batch, rank)
        y_B = torch.bmm(B, y_A.unsqueeze(-1)).squeeze(-1) # (n_batch, n_hidden)
        
        # make outputs (n_batch, n_features)
        y = self.activation(
            y + y_B * self.scaling
        )
        
        return y

class LatentMoLoRA(nn.Module):
    """Create a subject-specific mixture of low-rank adapters.
    
    This includes both a shared linear layer as well as, if desired,
    all subject specific adapters and mixtures. Contrary to LoRA 
    elsewhere, here we train both the shared and LoRA parts con-
    currently.
    
    Attributes
    ----------
    cfg : DictConfig
        Hydra configuration.
    n_dim : int
        Number of input dimensions.
    n_subjects : int
        Number of subjects.
    n_hidden : int | None, default=None
        If any, number of hidden dimensions to use. If `None`, :py:attr:`~n_dim` will be used.
    n_experts : int
        Number of experts.
    rank : int
        Rank of adapters.
    alpha : float
        Coefficient determining scaling magnitude.
    scaling : float
        Precomputed scaling factor.
    temperature : float | None
        If any, the temperature to use for gating.
    topk : int | None
        If any, the number of top-k experts to use per subject.
    base : nn.Module
        Shared linear projection.
    dropout : nn.Module
        Feature dropout module.
    activation : nn.Module
        Shared activation function.
    lora_dropout : nn.Module
        Feature dropout for LoRA.
    A_dims : Tuple[int]
        Dimensions of A matrix for LoRA.
    B_dims : Tuple[int]
        Dimensions of B matrix for LoRA
    A : nn.Parameter
        A matrices of shape `(n_experts, n_rank, n_dim)`.
    B : nn.Parameter
        B matrices of shape `(n_experts, n_hidden, n_rank)`.
    gate : nn.Module
        The shared mixture layer.
    gate_A_dims : Tuple[int]
        Dimensions of gate A matrix for LoRA.
    gate_B_dims : Tuple[int]
        Dimensions of gate B matrix for LoRA.
    gate_dropout : nn.Module
        Feature dropout for gate LoRA.
    gate_A : nn.Parameter
        A matrices for gates of shape `(n_subjects, n_rank, n_dim)` or `(n_rank, n_dim)`.
    gate_B : nn.Parameter
        B matrices for gates of shape `(n_subjects, n_experts, n_rank)` or `(n_experts, n_rank)`.
    """
    
    def __init__(self, cfg: DictConfig, n_dim: int, n_subjects: int, n_hidden: int | None = None):
        """Initialise a subject-specific MoLoRA layer.
        
        Parameters
        ----------
        cfg : DictConfig
            Hydra configuration.
        n_dim : int
            Number of input dimensions.
        n_subjects : int
            Number of subjects.
        n_hidden : int | None, default=None
            If any, number of hidden dimensions to use. If `None`, :py:attr:`~n_dim` will be used.
        """
        
        super().__init__()
        
        # setup config
        self.cfg = cfg
        self.n_dim = n_dim
        self.n_subjects = n_subjects
        self.n_hidden = n_hidden if n_hidden is not None else n_dim
        self.n_experts = self.cfg.experts
        
        # setup LoRA parameters
        self.rank = self.cfg.lora.rank
        self.alpha = self.cfg.lora.alpha
        self.scaling = self.alpha / max(1, self.rank)
        
        # setup mixture parameters
        self.temperature = self.cfg.gate.temperature
        self.topk = self.cfg.gate.topk
        
        # setup base
        self.base = nn.Linear(self.n_dim, self.n_hidden)
        self.dropout = nn.Dropout(cfg.linear.dropout)
        self.activation = make_activation(cfg.linear.activation)()
        
        # setup LoRA experts
        self.lora_dropout = nn.Dropout(cfg.lora.dropout)
        self.A_dims = (self.n_experts, self.rank, self.n_dim)
        self.B_dims = (self.n_experts, self.n_hidden, self.rank)
        self.A = nn.Parameter(torch.empty(*self.A_dims))
        self.B = nn.Parameter(torch.empty(*self.B_dims))
        nn.init.kaiming_uniform_(self.A, a = math.sqrt(5))
        nn.init.zeros_(self.B)
        
        # setup shared mixtures
        self.gate = nn.Linear(self.n_dim, self.n_experts)
        
        # setup LoRA mixtures
        if self.cfg.lora.per_subject:
            self.gate_A_dims = (n_subjects, self.rank, self.n_dim)
            self.gate_B_dims = (n_subjects, self.n_experts, self.rank)
        else:
            self.gate_A_dims = (self.rank, self.n_dim)
            self.gate_B_dims = (self.n_experts, self.rank)
        
        self.gate_dropout = nn.Dropout(cfg.gate.dropout)
        self.gate_A = nn.Parameter(torch.empty(*self.gate_A_dims))
        self.gate_B = nn.Parameter(torch.empty(*self.gate_B_dims))
        nn.init.kaiming_uniform_(self.gate_A, a = math.sqrt(5))
        nn.init.zeros_(self.gate_B)
    
    def forward(self, X: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass.
        
        Parameters
        ----------
        X : torch.Tensor
            Input data of shape `(n_batch, n_features)`.
        subject_ids : torch.Tensor
            Subject numbers of shape `(n_batch,)`.
        
        Returns
        -------
        y : torch.Tensor
            Outut data of shape `(n_batch, n_features)`.
        
        Notes
        -----
        For computationaly efficiency, this should technically use the weight 
        formulation of LoRA. As is, this is not optimal.
        """
        
        # base forward pass (n_batch, n_features)
        y_b = self.base(
            self.dropout(X)
        )
        
        # compute shared mixture weights
        x_g = self.gate_dropout(X)
        logits = self.gate(x_g) # (n_batch, n_experts)
        
        # compute LoRA mixture weights
        if self.cfg.lora.per_subject:
            gate_A = self.gate_A.index_select(0, subject_ids) # (n_batch, n_rank, n_features)
            gate_B = self.gate_B.index_select(0, subject_ids) # (n_batch, n_experts, n_rank)
        else:
            gate_A = self.gate_A.unsqueeze(0).expand(subject_ids.shape[0], -1, -1) # (n_batch, n_rank, n_features)
            gate_B = self.gate_B.unsqueeze(0).expand(subject_ids.shape[0], -1, -1) # (n_batch, n_experts, n_rank)
        
        gate_A_x = torch.bmm(gate_A, x_g.unsqueeze(-1)).squeeze(-1) # (n_batch, n_rank)
        gate_BA_x = torch.bmm(gate_B, gate_A_x.unsqueeze(-1)).squeeze(-1) # (n_batch, n_experts)
        logits = logits + self.scaling * gate_BA_x
        
        # check temperature
        if self.temperature != 1.0:
            logits = logits / max(self.temperature, 1e-6)
        
        # check sparsity
        if self.topk and (self.topk < self.n_experts):
            # for sparsity, keep only topk logits per row 
            # and set others to -inf
            topk_vals, topk_idx = torch.topk(logits, k = self.topk, dim = -1)
            masked = logits.new_full(logits.shape, float("-inf"))
            masked.scatter_(dim = -1, index = topk_idx, src = topk_vals)

            # make weights
            pi = F.softmax(masked, dim = -1)
        else:
            # otherwise, make full weights
            pi = F.softmax(logits, dim = -1)
        
        # compute expert updates
        x_d = self.lora_dropout(X)
        U = torch.einsum("krd,bd->bkr", self.A, x_d) # (n_batch, n_experts, n_rank)
        V = torch.einsum("khr,bkr->bkh", self.B, U) # (n_batch, n_experts, n_hidden)
        
        # make outputs
        y = self.activation(
            y_b + self.scaling * torch.einsum("bk,bkh->bh", pi, V)
        )
                
        return y

"""Registry and helper functions.
"""

# register latent layers
latent_layers = {
    'diagonal': LatentDiagonal,
    'lora': LatentLoRA,
    'molora': LatentMoLoRA
}

def make_latent(name: str) -> nn.Module:
    """Make a latent layer.
    
    Parameters
    ----------
    name : str
        Type of latent layer, as specified in :py:attr:`~latent_layers`.
    
    Returns
    -------
    layer : nn.Module
        The requested layer type.
    """
    
    # check entry
    if name.lower() not in latent_layers:
        raise ValueError(
            f'`{name}` unknown for latent layers.'
        )
    
    return latent_layers[name.lower()]