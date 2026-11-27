"""Dataset split classes.

Contains classes to help with dataset splits.
"""

import torch
import pandas as pd
import numpy as np

from dataclasses import dataclass

from typing import Generic, TypeVar

# define generic typevar
T = TypeVar("T")

@dataclass(slots = True)
class Split(Generic[T]):
    """Create a generic dataset split.
    
    Attributes
    ----------
    train : T
        Training dataset.
    valid : T
        Validation dataset.
    test : T
        Test dataset.
    repeat : int
        Repetition in cross-validation.
    split : int
        Split in cross-validation.
    train_id : torch.Tensor
        Identifiers of training data.
    valid_id : torch.Tensor
        Identifiers of validation data.
    test_id : torch.Tensor
        Identifiers of test data.
    """
    
    train: T
    valid: T
    test: T
    repeat: int
    split: int
    train_id: torch.Tensor
    valid_id: torch.Tensor
    test_id: torch.Tensor

@dataclass(slots = True)
class ArrangementsSplit(Split):
    """Create a datasplit for arrangements.
    
    Attributes
    ----------
    train : T
        Training dataset.
    valid : T
        Validation dataset.
    test : T
        Test dataset.
    repeat : int
        Repetition in cross-validation.
    split : int
        Split in cross-validation.
    train_id : torch.Tensor
        Identifiers of training data.
    valid_id : torch.Tensor
        Identifiers of validation data.
    test_id : torch.Tensor
        Identifiers of test data.
    """
    
    def _boards_demean(self, positions: torch.Tensor, board_ids: torch.Tensor) -> torch.Tensor:
        """Apply demeaning to embedding or behavioural data.
        
        Parameters
        ----------
        positions : torch.Tensor
            Behavioural or embedding coordinates.
        board_ids : torch.Tensor
            Tags for each board.
        
        Returns
        -------
        positions : torch.Tensor
            Behavioural or embedding coordinates, demeaned per board.
        """
        
        # find unique boards
        unq_boards = torch.unique(board_ids)
        
        # iterate over boards
        for board in unq_boards:
            # make mask
            mask = board_ids == board
            
            # demean
            positions[mask] = positions[mask] - positions[mask].mean(dim = 0, keepdim = True)
        
        return positions

    def _boards_scale(self, positions: torch.Tensor, board_ids: torch.Tensor) -> torch.Tensor:
        """Apply scaling to embedding or behavioural data.
        
        Parameters
        ----------
        positions : torch.Tensor
            Behavioural or embedding coordinates.
        board_ids : torch.Tensor
            Tags for each board.
        
        Returns
        -------
        positions : torch.Tensor
            Behavioural or embedding coordinates, scaled per board.
        """
        
        # find unique boards
        unq_boards = torch.unique(board_ids)
        
        # iteratore over boards
        for board in unq_boards:
            # make mask
            mask = board_ids == board
            
            # compute radius
            r = torch.linalg.vector_norm(positions[mask], dim = -1).max()
            
            # scale
            positions[mask] = positions[mask] / r
        
        return positions
    
    def _embeddings_normalise(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply normalisation to embeddding or behavioural data.
        
        Parameters
        ----------
        embeddings : torch.Tensor
            Behavioural or embeding coordinates.
        
        Returns
        -------
        embeddings : torch.Tensor
            Behavioural or embedding coordinates that are normalised.
        """
        
        return embeddings / torch.linalg.norm(embeddings, dim = -1, keepdim = True)
    
    def preprocess(self) -> "ArrangementsSplit":
        """Apply preprocessing to boards and embeddings.
        
        Returns
        -------
        split : ArrangementsSplit
            The preprocessed class.
        """
        
        # check board demeaning
        if self.train.cfg.data.preprocess.boards.demean:
            self.train.positions = self._boards_demean(self.train.positions, self.train.board_nos)
            self.valid.positions = self._boards_demean(self.valid.positions, self.valid.board_nos)
            self.test.positions = self._boards_demean(self.test.positions, self.test.board_nos)
        
        # check board scaling
        if self.train.cfg.data.preprocess.boards.scale:
            self.train.positions = self._boards_scale(self.train.positions, self.train.board_nos)
            self.valid.positions = self._boards_scale(self.valid.positions, self.valid.board_nos)
            self.test.positions = self._boards_scale(self.test.positions, self.test.board_nos)
        
        # check embedding SVD
        if self.train.cfg.data.preprocess.embeddings.svd:
            # iterate over embeddings
            for name in self.train.cfg.data.embeddings.models:
                # compute mean
                mu = self.train.embeddings[name].mean(0, keepdim = True)
                
                # demean and SVD
                self.train.embeddings[name] -= mu
                _, _, Vh = torch.linalg.svd(self.train.embeddings[name], full_matrices = False)
                V = Vh[:self.train.cfg.data.preprocess.embeddings.rank].T
                
                # apply decomposition
                self.train.embeddings[name] = self.train.embeddings[name] @ V
                self.valid.embeddings[name] = (self.valid.embeddings[name] - mu) @ V
                self.test.embeddings[name] = (self.test.embeddings[name] - mu) @ V
                
                # apply decomposition to contexts
                self.train.contexts[name] = (self.train.contexts[name] - mu[None,:,:]) @ V
                self.valid.contexts[name] = (self.valid.contexts[name] - mu[None,:,:]) @ V
                self.test.contexts[name] = (self.test.contexts[name] - mu[None,:,:]) @ V
        
        # check embedding normalisation
        if self.train.cfg.data.preprocess.embeddings.normalise:
            # iterate over embeddings
            for name in self.train.cfg.data.embeddings.models:
                # normalise embeddings
                self.train.embeddings[name] = self._embeddings_normalise(self.train.embeddings[name])
                self.valid.embeddings[name] = self._embeddings_normalise(self.valid.embeddings[name])
                self.test.embeddings[name] = self._embeddings_normalise(self.test.embeddings[name])
                
                # normalise contexts
                self.train.contexts[name] = self._embeddings_normalise(self.train.contexts[name])
                self.valid.contexts[name] = self._embeddings_normalise(self.valid.contexts[name])
                self.test.contexts[name] = self._embeddings_normalise(self.test.contexts[name])
        
        return self