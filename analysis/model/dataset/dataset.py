"""Dataset classes.

Contains classes to help with datasets.
"""

from omegaconf import DictConfig, OmegaConf
from loguru import logger

import torch
import pandas as pd
import numpy as np
import csv

from pathlib import Path
from bids import BIDSLayout

from .filter import make_filter
from .embedding import read_embedding, write_embedding
from .arrangements import Arrangements

class Dataset:
    """Dataset class.
    
    Attributes
    ----------
    cfg : DictConfig
        Hydra configuration.
    logger : logger
        Loguru logger object.
    bids_root : Path
        Path for the BIDS root.
    bids : BIDSLayout
        BIDS structure for navigating the path.
    subject_df : Optional[pd.DataFrame], default=None
        Dataframe containing all subject information.
    subject_ids : Optional[torch.Tensor], default=None
        Subject identifiers.
    embeddings : Dict[str, torch.Tensor]
        All available embeddings for this dataset.
    """
    
    def __init__(self, cfg: DictConfig, logger: logger) -> "Dataset":
        """Create a dataset class.
        
        Parameters
        ----------
        cfg : DictConfig
            Hydra configuration.
        logger : logger
            Loguru logger object.
        """
        
        # copy config and logger
        self.cfg = cfg
        self.logger = logger
        
        # setup bids
        self.bids_root = Path(cfg.data.path)
        self.bids = BIDSLayout(
            self.bids_root, 
            validate = cfg.data.bids.validate,
            database_path = self.bids_root / Path(cfg.data.bids.cache_file),
            reset_database = cfg.data.bids.cache_reset,
            derivatives = True
        )
    
    def load(self) -> "Dataset":
        """Load subjects and embeddings for this dataset.
        
        Returns
        -------
        dataset : Dataset
            Prepared and filtered dataset.
        """
        
        # load subject data
        if self.cfg.verbose:
            self.logger.info('Loading BIDS...')
        
        self.subject_df = pd.read_csv(
            self.bids_root / 'participants.tsv',
            sep = '\t', quoting = csv.QUOTE_NONE
        )
        
        # filter subject data
        if self.cfg.verbose:
            self.logger.info('Filtering subjects...')
        
        subject_mask = np.ones((len(self.subject_df),), dtype = bool)
        
        for name in self.cfg.data.filter:
            # grab filter
            filter = getattr(self.cfg.data.filter, name)
            subject_mask = subject_mask & make_filter(filter, self.subject_df)
        
        # apply filter
        valid = np.where(subject_mask)[0]
        self.subject_df = self.subject_df.loc[
            valid
        ].reset_index(drop = True)
        
        # make unique subjects
        self.subject_ids = np.unique(
            self.subject_df.participant_id.tolist()
        )
        
        # load embeddings
        self.embeddings = {
            name: read_embedding(
                Path(self.cfg.data.resources) / Path(name) / Path(self.cfg.data.embeddings.version + '.tsv')
            )
            for name in self.cfg.data.embeddings.models
        }
        
        return self

    def load_arrangements(self, n_session: int | None = None) -> Arrangements:
        """Load arrangement task data.
        
        Returns
        -------
        data : Arrangements
            Data from arrangement task.
        n_session : Optional[int], default=None
            Which session to look for.
        """
        
        # stacks
        subject_ids = []
        subject_nos = []
        board_ids = []
        board_nos = []
        word_nos = []
        embeddings = {name: [] for name in self.embeddings}
        contexts = {name: [] for name in self.embeddings}
        positions = []
        
        # loop over subjects
        for i, subject_id in enumerate(self.subject_ids):
            # grab identifier
            sid = subject_id.split('-')[-1]
            
            # setup extra args
            kwargs = {'session': n_session} if n_session is not None else {}
            
            # load files
            files = self.bids.get(
                subject = [sid],
                datatype = 'beh',
                task = 'ma',
                extension = 'tsv',
                **kwargs
            )
            
            # check availability
            if len(files) < 1:
                continue
                        
            # load data
            df = pd.read_csv(
                files[0], 
                sep = '\t', quoting = csv.QUOTE_NONE
            ).sort_values(
                ['board_id', 'word']
            )
            
            # compute local word identifiers
            words = np.unique(df.word.tolist())
            word_to_no = {word: i for i, word in enumerate(words)}
            word_no = np.array([
                word_to_no[word]
                for word in df.word.tolist()
            ])
            
            # compute embeddings
            embedding = {
                name: torch.stack(
                    [
                        self.embeddings[name][word.lower()]
                        for word in df.word.tolist()
                    ],
                    dim = 0
                ).float()
                for name in self.embeddings
            }
            
            # compute contexts
            context = {name: [] for name in self.embeddings}
            
            for name in self.embeddings:
                numbers = torch.tensor(df.board_no.tolist()).long()
                mask = numbers[:,None] == numbers[None,:]
                
                for j in range(mask.shape[0]):
                    context[name].append(embedding[name][mask[j,:]].clone())

                context[name] = torch.stack(context[name], dim = 0)
            
            # prepare data
            subject_id = torch.tensor([int(sid)] * len(df)).long()
            subject_no = torch.tensor([i] * len(df)).long()
            board_id = torch.tensor(df.board_id.tolist()).long()
            board_no = torch.tensor(df.board_no.tolist()).long()
            word_no = torch.tensor(word_no).long()
            position = torch.tensor([
                np.array(df.x.tolist()).astype(float).tolist(),
                np.array(df.y.tolist()).astype(float).tolist()
            ]).float().t()
            
            # add data
            subject_ids.append(subject_id)
            subject_nos.append(subject_no)
            board_ids.append(board_id)
            board_nos.append(board_no)
            word_nos.append(word_no)
            positions.append(position)
            
            for name in self.embeddings:
                embeddings[name].append(embedding[name])
                contexts[name].append(context[name])
        
        # stack data
        subject_ids = torch.stack(subject_ids, dim = 0).reshape((-1,)).long()
        subject_nos = torch.stack(subject_nos, dim = 0).reshape((-1,)).long()
        board_ids = torch.stack(board_ids, dim = 0).reshape((-1,)).long()
        board_nos = torch.stack(board_nos, dim = 0).reshape((-1,)).long()
        word_nos = torch.stack(word_nos, dim = 0).reshape((-1,)).long()
        positions = torch.stack(positions, dim = 0)
        positions = positions.reshape((-1, positions.shape[-1])).float()
        
        for name in self.embeddings:
            embeddings[name] = torch.stack(embeddings[name], dim = 0)
            contexts[name] = torch.stack(contexts[name], dim = 0)
            
            embeddings[name] = embeddings[name].reshape((-1, embeddings[name].shape[-1])).float()
            contexts[name] = contexts[name].reshape((-1, contexts[name].shape[-2], contexts[name].shape[-1])).float()
        
        return Arrangements(
            self.cfg, self.logger,
            subject_ids.to(self.cfg.device), subject_nos.to(self.cfg.device),
            board_ids.to(self.cfg.device), board_nos.to(self.cfg.device),
            word_nos.to(self.cfg.device),
            {name: embeddings[name].to(self.cfg.device) for name in embeddings}, 
            {name: contexts[name].to(self.cfg.device) for name in embeddings},
            positions.to(self.cfg.device),
            self.subject_df
        )