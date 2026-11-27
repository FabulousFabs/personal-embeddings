"""Arrangement datasets.

Contains classes to help with arrangement datasets.
"""

from omegaconf import DictConfig, OmegaConf
from loguru import logger

import torch
import pandas as pd
import numpy as np

from dataclasses import dataclass
from mvpy.math import rank, spearmanr
from torch.utils.data import TensorDataset, DataLoader

from .split import ArrangementsSplit

from typing import Optional, Union, Dict, List, Tuple, Iterator

@dataclass(slots = True)
class Arrangements:
    """Create an arrangement class.
    
    Attributes
    ----------
    cfg : DictConfig
        Hydra configuration.
    logger : logger
        Loguru logger object.
    subject_ids : torch.Tensor
        Subject identifiers.
    subject_nos : torch.Tensor
        Subject numbers (incremental).
    board_ids : torch.Tensor
        Board identifiers.
    board_nos : torch.Tensor
        Board numbers (incremental).
    word_nos : torch.Tensor
        Word numbers (incremental, uniquely identifying words).
    embeddings : Dict[str, torch.Tensor]
        Embeddings for all embedding types and words.
    contexts : Dict[str, torch.Tensor]
        Context embeddings for all embedding types and words.
    positions : torch.Tensor
        Behavioural coordinates for all words.
    metadata : pd.DataFrame
        Metadata for this dataset.
    reliability : Optional[torch.Tensor], default=None
        Reliability of behavioural data per subject.
    ceiling : Optional[torch.Tensor], default=None
        Noise ceiling of behavioural data per subject.
    """
    
    cfg: DictConfig
    logger: logger
    subject_ids: torch.Tensor
    subject_nos: torch.Tensor
    board_ids: torch.Tensor
    board_nos: torch.Tensor
    word_nos: torch.Tensor
    embeddings: Dict[str, torch.Tensor]
    contexts: Dict[str, torch.Tensor]
    positions: torch.Tensor
    metadata: pd.DataFrame
    reliability: torch.Tensor | None = None
    ceiling: torch.Tensor | None = None

    def subset(self, selection: torch.Tensor) -> "Arrangements":
        """Make a subset of this dataclass.
        
        Parameters
        ----------
        selection : torch.Tensor
            Mask for data to be selected.
        
        Returns
        -------
        subset : Arrangements
            Subset arrangement class.
        """
        
        # return selected arrangements
        return Arrangements(
            self.cfg, self.logger,
            self.subject_ids[selection], self.subject_nos[selection],
            self.board_ids[selection], self.board_nos[selection],
            self.word_nos[selection],
            {
                name: self.embeddings[name][selection]
                for name in self.embeddings
            },
            {
                name: self.contexts[name][selection]
                for name in self.contexts
            },
            self.positions[selection],
            self.metadata
        )
    
    def split(self, n_splits: int, n_repeats: int, seed: int | None = None) -> Iterator[ArrangementsSplit]:
        """Make repeated k-folds.
        
        Parameters
        ----------
        n_splits : int
            Number of splits.
        n_repeats : int
            Number of repeats.
        seed : Optional[int], default=None
            Seed to apply for splitting.
        
        Returns
        -------
        splits : Iterator[ArrangementsSplit]
            Iterable dataset splits.
        """
        
        # check splits
        if n_splits < 3:
            raise ValueError(
                f'n_splits must be greater than 3 to produce ' +
                f'train, valid, test splits.'
            )
        
        # check repeats
        if n_repeats < 1:
            raise ValueError(
                f'n_repeats must be greater or equal to 1.'
            )
        
        # find row to board mappings
        boards_unq, boards_inv = torch.unique(
            self.board_ids, return_inverse = True
        )
        
        # check boards
        n_boards = int(boards_unq.numel())
        
        if n_splits > n_boards:
            raise ValueError(
                f'Number of splits ({n_splits}) cannot be greater ' + 
                f'than number of available boards ({n_boards}).'
            )
        
        # make balanced fold sizes
        base = n_boards // n_splits
        rem = n_boards % n_splits
        fold_sizes = [base + 1] * rem + [base] * (n_splits - rem)
        
        # make generator
        gen = torch.Generator(device = self.cfg.device)
        if seed is not None:
            gen.manual_seed(seed)
        
        # iterate over repeats
        for i in range(n_repeats):
            # permute board indices
            perm_board_idx = torch.randperm(n_boards, device = self.cfg.device, generator = gen)
            
            # assign chunks to folds
            fold_id_of_board = torch.empty(n_boards, device = self.cfg.device, dtype = torch.long)
            
            start = 0
            folds = []
            for f, sz in enumerate(fold_sizes):
                idx = perm_board_idx[start:start + sz]
                fold_id_of_board[idx] = f
                folds.append(idx)
                start += sz
            
            # for each row, which fold is its board in?
            fold_id_of_row = fold_id_of_board[boards_inv]
            
            # iterate over splits
            for f_test in range(n_splits):
                # select validation
                f_valid = (f_test + 1) % n_splits
                
                # make masks
                test_mask = fold_id_of_row == f_test
                valid_mask = fold_id_of_row == f_valid
                train_mask = (test_mask | valid_mask) == False
                
                # grab identifiers
                boards_test = boards_unq[folds[f_test]]
                boards_valid = boards_unq[folds[f_valid]]
                boards_train = boards_unq[
                    torch.isin(
                        torch.arange(n_boards, device = self.cfg.device), 
                        torch.cat([folds[f_test], folds[f_valid]])
                    ) == False
                ]
                
                # check patience mode
                if self.cfg.model.training.patience.enable == False:
                    # in case we don't employ patience, we'd like to 
                    # include the validation data as part of the test
                    # set because validation isn't besides the early
                    # stopping algo
                    test_mask = valid_mask | test_mask
                    boards_test = torch.cat([boards_test, boards_valid])
                
                yield ArrangementsSplit(
                    self.subset(train_mask),
                    self.subset(valid_mask),
                    self.subset(test_mask),
                    i, f_test,
                    boards_train,
                    boards_valid,
                    boards_test
                )
        
    def split_like(self, split: ArrangementsSplit) -> ArrangementsSplit:
        """Recreate a the same split from another dataset.
        
        Note that this will also change :py:attr:`~subject_nos` to make sure
        that they match those in the prespecified dataset. This is critical
        to make sure that pretrained models (that operate on subject nos, not
        subject identities, due to the potential of missing participants) are
        fed the correct subject information when computing individualised
        embeddings.
        
        Parameters
        ----------
        split : ArrangementsSplit
            The split to recreate.
        
        Returns
        -------
        resplit : ArrangementsSplit
            The current dataset, split as the previous one.
        """
        
        # create masks
        train_mask = torch.isin(self.board_ids, split.train_id)
        valid_mask = torch.isin(self.board_ids, split.valid_id)
        test_mask = torch.isin(self.board_ids, split.test_id)
        
        # translate subject numbers
        # this is necessary because there may be missing
        # subjects in one sessions; however, subject nos
        # must be continuous integers to work with the
        # model, so IDs cannot be used and we need to 
        # rely on NOs instead -- therefore, we now need
        # to make sure that the numbers in our new split
        # will match those in the old split such that
        # each subject still gets assigned their model
        subject_nos = torch.zeros_like(self.subject_nos)
        
        for i, subject in enumerate(torch.unique(self.subject_ids)):
            # find subject number in prespecified split
            subject_no_j = split.train.subject_nos[split.train.subject_ids == subject][0].item()
            
            # create mask for this split and record previous subject number
            mask = self.subject_ids == subject
            subject_nos[mask] = subject_no_j
        
        # make a trivial mask to clone the dataset
        mask = torch.ones((self.board_ids.shape[0],), dtype = torch.bool, device = self.board_ids.device)
        clone = self.subset(mask)
        clone.subject_nos = subject_nos
        
        return ArrangementsSplit(
            clone.subset(train_mask),
            clone.subset(valid_mask),
            clone.subset(test_mask),
            split.repeat, split.split,
            split.train_id,
            split.valid_id,
            split.test_id
        )
        
    def make_loader(self, device: str, batch_size: int, name: str, shuffle: bool = False) -> DataLoader:
        """Make a torch dataloader from this dataset.
        
        Parameters
        ----------
        device : str
            The device to put the data on.
        batch_size : int
            Number of items per batch.
        name : str
            The embedding to use for this loader.
        shuffle : bool, default=False
            Should this dataloader be shuffled?
        
        Returns
        -------
        dataloader : DataLoader
            Iterable torch dataloader.
        """
                
        # make tensor dataset
        dataset = TensorDataset(
            self.subject_nos.to(device), self.board_nos.to(device), self.word_nos.to(device), 
            self.embeddings[name].to(device), self.contexts[name].to(device),
            self.positions.to(device)
        )
        
        # make tensor loader
        dataloader = DataLoader(
            dataset,
            batch_size = batch_size,
            shuffle = shuffle
        )
        
        return dataloader

    def compute_split_half_reliability(self, min_length: int = 3, repeats: int = 1) -> None:
        """Compute split half reliability over repeat items in the dataset.
        
        Parameters
        ----------
        min_length : int, default=3
            Minimum number of items per board that can be considered.
        repeats : int, default=1
            Number of repetitions to compute (currently unused given non-determinism).
        """
        
        # order by subject nos
        unq_subject_nos = torch.unique(self.subject_nos)
        unq_subjects = torch.tensor(
            [
                self.subject_ids[
                    torch.where(self.subject_nos == subject_no)[0][0]
                ].item()
                for subject_no in unq_subject_nos
            ],
            dtype = torch.long, device = self.cfg.device
        )
        
        # setup container
        rho = torch.zeros((
            unq_subjects.shape[0], repeats
        ), dtype = torch.float, device = self.cfg.device)
        
        # iterate over repeats
        for r_i in range(repeats):
            # iterate over subjects
            for s_i, subject in enumerate(unq_subjects):
                # subject mask
                mask_i = self.subject_ids == subject
                
                # grab boards
                unq_boards = torch.unique(self.board_nos[mask_i])
                
                # setup container
                rho_i = []
                count_i = []
                
                # iterate over boards
                for b_j, board1 in enumerate(unq_boards):
                    # subject and board mask
                    mask_j = mask_i & (self.board_nos == board1)
                    
                    # find ids
                    ids_j = self.word_nos[mask_j]
                    
                    # iterate over boards
                    for b_k, board2 in enumerate(unq_boards):
                        # check triangle
                        if b_k >= b_j:
                            continue
                        
                        # subject and board mask
                        mask_k = mask_i & (self.board_nos == board2)
                        
                        # find ids
                        ids_k = self.word_nos[mask_k]
                        
                        # find repeated words
                        anchors = torch.isin(ids_j, ids_k)
                        
                        # check length
                        if anchors.sum() < min_length:
                            continue
                        
                        # obtain ordered anchors and indices
                        anchors_j = torch.where(anchors)[0]
                        anchors_k = torch.tensor(
                            [
                                torch.where(ids_k == anchor_j)[0][0]
                                for anchor_j in ids_j[anchors_j]
                            ],
                            dtype = torch.long, device = self.cfg.device
                        )
                        
                        # compute gram matrices
                        gram = lambda x: torch.linalg.norm(x[:,None,:] - x[None,:,:], dim = -1)
                        JJ = gram(self.positions[mask_j][anchors_j])
                        KK = gram(self.positions[mask_k][anchors_k])
                        
                        # compute spearman rho
                        tx, ty = torch.triu_indices(*JJ.shape, offset = 1)
                        rho_i.append(
                            spearmanr(
                                JJ[tx,ty], KK[tx,ty]
                            ).item()
                        )
                        count_i.append(tx.shape[0])
                
                # tally container
                rho_i = torch.tensor(rho_i, dtype = torch.float, device = self.cfg.device)
                count_i = torch.tensor(count_i, dtype = torch.float, device = self.cfg.device)
                rho[s_i,r_i] = (count_i * rho_i).sum() / count_i.sum()
        
        # compute mean
        rho = rho.mean(1)
        
        # apply spearman brown correction
        rho = (2 * rho) / (1 + rho)
        
        # make reliability
        self.reliability = torch.clamp(
            rho, min = 0.0, max = 1.0
        )
        
        # make noise ceiling
        self.ceiling = torch.sqrt(self.reliability)
    
    def make_demographics(self, demographics: DictConfig, cfg: DictConfig, permute: Optional[Union[List, str]] = None) -> Tuple[torch.Tensor, Dict]:
        """Prepare demographics for the current dataset.
        
        Parameters
        ----------
        demographics : DictConfig
            The demographics structure to prepare.
        cfg : DictConfig
            The hydra configuration for this run.
        permute : List | str | None, default=None
            Which demographic (if any) to permute.
        
        Returns
        -------
        X : torch.Tensor
            The demographics of shape `(n_samples, n_features)`.
        L : Dict[str, Dict[str, Any]]
            Label structure containing each demographic name bound to a dictionary of 
            `ids` denoting the position in `X` and `labels` denoting category labels 
            for all positions.
        """
        
        # check permutation argument
        if permute is None:
            permute = []
        
        if isinstance(permute, str):
            permute = [permute,]
        
        # check permutation axes
        for parameter in permute:
            if parameter not in demographics:
                raise ValueError(
                    f'Permutation parameter {parameter} not found ' +
                    f'in demographic specification.'
                )
        
        # check participants
        n_subjects = torch.unique(self.subject_ids).shape[0]
        
        # setup container
        X = []
        L = dict()
        
        # setup permutation
        permutation = np.arange(n_subjects)
        permutation = np.random.choice(
            permutation,
            size = (permutation.shape[0],),
            replace = False
        )
        
        # setup tracker
        is_dyad = None
        
        # iterate over predictors
        for name in demographics:
            # grab descriptor
            predictor = demographics[name]
            
            # track dyad status
            if (is_dyad is not None) and (is_dyad != predictor['dyad']):
                raise ValueError(
                    f'Predictor {name} violates prior dyad status. ' +
                    f'Predictors must either be all dyad or none.'
                )
            is_dyad = predictor['dyad']
            
            # check typing
            if predictor['type'] not in ['continuous', 'categorical']:
                raise ValueError(
                    f'Predictor {name} has unknown type {predictor["type"]}.'
                )
            
            # require order for categoricals
            if (predictor['type'] == 'categorical') and ('order' not in predictor):
                raise ValueError(
                    f'Predictor {name} is categorical, but does not specify order.'
                )
            
            # disallow multisource for categoricals
            if (predictor['type'] == 'categorical') and isinstance(predictor['sources'], list):
                raise ValueError(
                    f'Predictor {name} is categorical, which does not support multiple sources.'
                )
            
            # prepare sources
            if isinstance(predictor['sources'], str):
                predictor['sources'] = [predictor['sources']]
            
            # setup dummy
            X_n = []
            L_n = []
            
            # iterate over sources
            for s_i, source in enumerate(predictor['sources']):
                # grab data
                X_s = np.array(self.metadata[source].tolist())[:,None]
                
                # if desired, permute data
                if name in permute:
                    X_s = X_s[permutation]
                
                # check for categorical types that need processing
                if predictor['type'] == 'categorical':
                    # check for ordering
                    X_p = np.zeros(
                        (n_subjects, len(predictor['order']))
                    )
                    
                    # make binary labels
                    for o_i, label in enumerate(predictor['order']):
                        X_p[
                            (X_s[:,0] == label),o_i
                        ] = 1
                    
                    # check solution
                    if (X_p.sum(1) == 0).any():
                        raise ValueError(
                            f'Predictor {name} source {source} has zero-sum subjects.'
                        )
                    
                    # check ranking (and, if not dyad, collapse)
                    if predictor['rank'] or predictor['dyad']:
                        X_p = np.argmax(X_p, axis = 1)[:,None]
                        L_n.append(source)
                    else:
                        for o_i, label in enumerate(predictor['order']):
                            L_n.append(label)
                    
                    # overwrite solution
                    X_s = X_p
                else:
                    # check continuous ranking
                    if predictor['rank']:
                        X_s = rank(X_s.T).T
                    
                    # add label
                    L_n.append(source)
                
                # add to dummy
                for i in range(X_s.shape[1]):
                    X_n += [list(X_s[:,i])]
            
            # make numpy
            X_n = np.array(X_n).T
            
            # check output type
            if predictor['dyad']:
                # convert to dyads
                if(predictor['type'] == 'continuous') or (predictor['rank']):
                    # compute distances for continuous or ranked dyads
                    X_n = np.linalg.norm(X_n[:,None,:] - X_n[None,:,:], axis = -1)[...,None]
                else:
                    # compute regular dyad for categorical unranked
                    X_n = (X_n[:,None] == X_n[None,:])
            
            # add to L
            L[name] = {
                'ids': torch.arange(len(X), len(X) + X_n.shape[-1]),
                'labels': L_n
            }
            
            # add to X
            for i in range(X_n.shape[-1]):
                X += [list(X_n[...,i])]
        
        # make numpy
        X = np.array(X)
        X = np.moveaxis(X, 0, -1)
        
        # convert to torch
        X = torch.from_numpy(X).float().to(cfg.device)
        
        return X, L