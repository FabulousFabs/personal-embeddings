"""Reliability.

Contains the arrangement (cross)-reliability tasks.
"""

from omegaconf import DictConfig, OmegaConf
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from mvpy.math import spearmanr
from tqdm import tqdm

from ..dataset import Dataset, ArrangementsSplit
from .. import metrics
from .. import model
from .results import save_supplement

from typing import Tuple, Dict

def arrangement_reliability(task: DictConfig, data: Dataset, cfg: DictConfig, logger: logger) -> None:
    """Perform a reliability check of subject answers.
    
    Parameters
    ----------
    task : DictConfig
        Hydra subconfiguration for this specific task.
    data : Dataset
        The preloaded dataset.
    cfg : DictConfig
        Hydra configuration.
    logger : logger
        The logging object.
    """
    
    # log progress
    if cfg.verbose:
        logger.info('Performing arrangement reliability...')
    
    # load multiple arrangement data
    arrangements = data.load_arrangements(n_session = 1)
    
    # check subjects
    n_subjects = torch.unique(arrangements.subject_nos).shape[0]
    
    # order by subject nos
    unq_subject_nos = torch.unique(arrangements.subject_nos)
    unq_subjects = torch.tensor(
        [
            arrangements.subject_ids[
                torch.where(arrangements.subject_nos == subject_no)[0][0]
            ].item()
            for subject_no in unq_subject_nos
        ],
        dtype = torch.long, device = cfg.device
    )
    
    # setup container
    rho = torch.zeros(
        (n_subjects, n_subjects, task.repeats), 
        dtype = torch.float,
        device = cfg.device
    )
    
    rho_emb = torch.zeros(
        (n_subjects, n_subjects, len(arrangements.embeddings), task.repeats), 
        dtype = torch.float,
        device = cfg.device
    )
    
    # setup progress
    pbar = tqdm(
        total = task.repeats * unq_subjects.shape[0],
        leave = True,
        disable = (cfg.verbose == False)
    )
    
    # iterate over repeats
    for r_i in range(task.repeats):
        # iterate over subjects
        for s_i, subject_i in enumerate(unq_subjects):
            # mask subject i
            mask_i = arrangements.subject_ids == subject_i
            
            # grab boards
            board_nos_i = torch.unique(arrangements.board_nos[mask_i])
            board_ids_i = torch.tensor(
                [
                    arrangements.board_ids[mask_i][
                        torch.where(arrangements.board_nos[mask_i] == board_no_i)[0][0]
                    ].item()
                    for board_no_i in board_nos_i
                ],
                dtype = torch.long, device = cfg.device
            )
            
            # iterate over subjects
            for s_j, subject_j in enumerate(unq_subjects):
                # check triangle
                if s_i > s_j:
                    continue
                
                # mask subject j
                mask_j = arrangements.subject_ids == subject_j
                
                # grab boards
                board_nos_j = torch.unique(arrangements.board_nos[mask_j])
                board_ids_j = torch.tensor(
                    [
                        arrangements.board_ids[mask_j][
                            torch.where(arrangements.board_nos[mask_j] == board_no_j)[0][0]
                        ].item()
                        for board_no_j in board_nos_j
                    ],
                    dtype = torch.long, device = cfg.device
                )
                
                # setup dummy
                rho_j = []
                emb_j = [[] for name in arrangements.embeddings]
                count_j = []
                
                # iterate over boards
                for b_i, (board_no_i, board_id_i) in enumerate(zip(board_nos_i, board_nos_j)):
                    # subject and board masks
                    mask_ii = mask_i & (arrangements.board_nos == board_no_i)
                    
                    # find ids
                    ids_i = arrangements.word_nos[mask_ii]
                    
                    # iterate over boards
                    for b_k, (board_no_k, board_id_k) in enumerate(zip(board_nos_i, board_nos_i)):
                        # check upper triangle
                        if b_i >= b_k:
                            continue
                        
                        # subject and board masks
                        mask_jk = mask_j & (arrangements.board_nos == board_no_k)
                        
                        # find ids
                        ids_k = arrangements.word_nos[mask_jk]
                        
                        # find repeated words
                        anchors = torch.isin(ids_i, ids_k)
                        
                        # check length
                        if anchors.sum() < task.min_length:
                            continue
                        
                        # obtain ordered anchors and indices
                        anchors_i = torch.where(anchors)[0]
                        anchors_k = torch.tensor(
                            [
                                torch.where(ids_k == anchor_i)[0][0]
                                for anchor_i in ids_i[anchors_i]
                            ]
                        )
                        
                        # compute gram matrices
                        gram = lambda x: torch.linalg.norm(x[:,None,:] - x[None,:,:], dim = -1)
                        II = gram(arrangements.positions[mask_ii][anchors_i])
                        JJ = gram(arrangements.positions[mask_jk][anchors_k])
                        
                        # compute spearman rho
                        tx, ty = torch.triu_indices(*II.shape, offset = 1)
                        rho_j.append(
                            spearmanr(
                                II[tx,ty], JJ[tx,ty]
                            ).item()
                        )
                        count_j.append(tx.shape[0])
                        
                        # compute same for embeddings
                        for e_j, embedding in enumerate(arrangements.embeddings):
                            # grab embeddings
                            X = arrangements.embeddings[embedding][mask_ii][anchors_i]
                            
                            # normalise embeddings
                            X = F.normalize(X, dim = -1)
                            
                            # compute gram matrix
                            KK = gram(X)
                            
                            # compute spearman rho
                            emb_j[e_j].append(
                                spearmanr(
                                    II[tx,ty], KK[tx,ty]
                                ).item()
                            )
                
                # make tensors
                rho_j = torch.tensor(rho_j, dtype = torch.float, device = cfg.device)
                count_j = torch.tensor(count_j, dtype = torch.float, device = cfg.device)
                rho[s_j,s_i,r_i] = rho[s_i,s_j,r_i] = (count_j * rho_j).sum() / count_j.sum()
                
                for e_j, embedding in enumerate(arrangements.embeddings):
                    E_J = torch.tensor(emb_j[e_j], dtype = torch.float, device = cfg.device)
                    rho_emb[s_j,s_i,e_j,r_i] = rho_emb[s_i,s_j,e_j,r_i] = (count_j * E_J).sum() / count_j.sum()

            # update tqdm
            pbar.update(1)
    
    # close tqdm
    pbar.close()
    
    # compute mean
    rho = rho.mean(-1)
    rho_emb = rho_emb.mean(-1)
    
    # if required, apply spearman brown correction
    if task.correct:
        rho = (2 * rho) / (1 + rho)
        rho_emb = (2 * rho_emb) / (1 + rho_emb)
    
    # if required, make noise ceiling
    if task.ceiling:
        rho = torch.sqrt(
            torch.clamp(rho, min = 0.0, max = 1.0)
        )
        
        rho_emb = torch.sqrt(
            torch.clamp(rho_emb, min = 0.0, max = 1.0)
        )
    
    # prepare outputs
    outputs = {
        'subject_nos': unq_subject_nos,
        'subject_ids': unq_subjects,
        'embeddings': list(arrangements.embeddings.keys()),
        'rho': rho,
        'rho_emb': rho_emb
    }
    
    # save data
    save_supplement(outputs, 'mareliability', cfg, logger)

def arrangement_crossreliability(task: DictConfig, data: Dataset, cfg: DictConfig, logger: logger) -> None:
    """Perform a reliability check of subject answers between repeated sessions.
    
    Parameters
    ----------
    task : DictConfig
        Hydra subconfiguration for this specific task.
    data : Dataset
        The preloaded dataset.
    cfg : DictConfig
        Hydra configuration.
    logger : logger
        The logging object.
    """
    
    # log progress
    if cfg.verbose:
        logger.info('Performing arrangement cross-reliability...')
    
    # load multiple arrangement data for both sessions
    arrangements = data.load_arrangements(n_session = 1)
    repeated = data.load_arrangements(n_session = 2)
    
    # check subjects
    n_subjects = torch.unique(arrangements.subject_nos).shape[0]
    
    # order by subject nos
    unq_subject_nos = torch.unique(arrangements.subject_nos)
    unq_subjects = torch.tensor(
        [
            arrangements.subject_ids[
                torch.where(arrangements.subject_nos == subject_no)[0][0]
            ].item()
            for subject_no in unq_subject_nos
        ],
        dtype = torch.long, device = cfg.device
    )
    
    # setup container
    rho = torch.zeros(
        (n_subjects, n_subjects, task.repeats), 
        dtype = torch.float,
        device = cfg.device
    )
    
    rho_emb = torch.zeros(
        (n_subjects, n_subjects, len(arrangements.embeddings), task.repeats), 
        dtype = torch.float,
        device = cfg.device
    )
    
    rho_bme = torch.zeros(
        (n_subjects, n_subjects, len(arrangements.embeddings), task.repeats), 
        dtype = torch.float,
        device = cfg.device
    )
    
    # setup progress
    pbar = tqdm(
        total = task.repeats * unq_subjects.shape[0] ** 2,
        leave = True,
        disable = (cfg.verbose == False)
    )
    
    # iterate over repeats
    for r_i in range(task.repeats):
        # iterate over subjects
        for s_i, subject_i in enumerate(unq_subjects):
            # mask subject i
            mask_i = arrangements.subject_ids == subject_i
            
            # grab boards
            board_nos_i = torch.unique(arrangements.board_nos[mask_i])
            board_ids_i = torch.tensor(
                [
                    arrangements.board_ids[mask_i][
                        torch.where(arrangements.board_nos[mask_i] == board_no_i)[0][0]
                    ].item()
                    for board_no_i in board_nos_i
                ],
                dtype = torch.long, device = cfg.device
            )
            
            # iterate over subjects
            for s_j, subject_j in enumerate(unq_subjects):
                # mask subject j
                mask_j = repeated.subject_ids == subject_j
                
                # grab boards
                board_nos_j = torch.unique(repeated.board_nos[mask_j])
                board_ids_j = torch.tensor(
                    [
                        repeated.board_ids[mask_j][
                            torch.where(repeated.board_nos[mask_j] == board_no_j)[0][0]
                        ].item()
                        for board_no_j in board_nos_j
                    ],
                    dtype = torch.long, device = cfg.device
                )
                
                # setup dummy
                rho_j = []
                emb_j = [[] for _ in arrangements.embeddings]
                emb_k = [[] for _ in repeated.embeddings]
                count_j = []
                
                # iterate over boards
                for b_i, (board_no_i, board_id_i) in enumerate(zip(board_nos_i, board_nos_j)):
                    # subject and board masks
                    mask_ii = mask_i & (arrangements.board_nos == board_no_i)
                    
                    # find ids
                    ids_i = arrangements.word_nos[mask_ii]
                    
                    # iterate over boards
                    for b_k, (board_no_k, board_id_k) in enumerate(zip(board_nos_i, board_nos_j)):
                        # subject and board masks
                        mask_jk = mask_j & (repeated.board_nos == board_no_k)
                                                
                        # find ids
                        ids_k = repeated.word_nos[mask_jk]
                        
                        # find repeated words
                        anchors = torch.isin(ids_i, ids_k)
                        
                        # check length
                        if anchors.sum() < task.min_length:
                            continue
                        
                        # obtain ordered anchors and indices
                        anchors_i = torch.where(anchors)[0]
                        anchors_k = torch.tensor(
                            [
                                torch.where(ids_k == anchor_i)[0][0]
                                for anchor_i in ids_i[anchors_i]
                            ]
                        )
                        
                        # compute gram matrices
                        gram = lambda x: torch.linalg.norm(x[:,None,:] - x[None,:,:], dim = -1)
                        II = gram(arrangements.positions[mask_ii][anchors_i])
                        JJ = gram(repeated.positions[mask_jk][anchors_k])
                        
                        # compute spearman rho
                        tx, ty = torch.triu_indices(*II.shape, offset = 1)
                        rho_j.append(
                            spearmanr(
                                II[tx,ty], JJ[tx,ty]
                            ).item()
                        )
                        count_j.append(tx.shape[0])
                        
                        # compute same for embeddings
                        for e_j, embedding in enumerate(arrangements.embeddings):
                            # grab embeddings
                            X = arrangements.embeddings[embedding][mask_ii][anchors_i]
                            
                            # normalise embeddings
                            X = F.normalize(X, dim = -1)
                            
                            # compute gram matrix
                            KK = gram(X)
                            
                            # compute spearman rho
                            emb_j[e_j].append(
                                spearmanr(
                                    II[tx,ty], KK[tx,ty]
                                ).item()
                            )
                            
                            # grab embeddings
                            X = repeated.embeddings[embedding][mask_jk][anchors_k]
                            
                            # normalise embeddings
                            X = F.normalize(X, dim = -1)
                            
                            # compute gram matrix
                            KK = gram(X)
                            
                            # compute spearman rho
                            emb_k[e_j].append(
                                spearmanr(
                                    JJ[tx,ty], KK[tx,ty]
                                )
                            )
                
                # make tensors
                rho_j = torch.tensor(rho_j, dtype = torch.float, device = cfg.device)
                count_j = torch.tensor(count_j, dtype = torch.float, device = cfg.device)
                rho[s_i,s_j,r_i] = (count_j * rho_j).sum() / count_j.sum()
                
                for e_j, embedding in enumerate(arrangements.embeddings):
                    E_J = torch.tensor(emb_j[e_j], dtype = torch.float, device = cfg.device)
                    rho_emb[s_i,s_j,e_j,r_i] = (count_j * E_J).sum() / count_j.sum()
                    
                    E_J = torch.tensor(emb_k[e_j], dtype = torch.float, device = cfg.device)
                    rho_bme[s_i,s_j,e_j,r_i] = (count_j * E_J).sum() / count_j.sum()
                
                # update tqdm
                pbar.update(1)
    
    # close tqdm
    pbar.close()
    
    # compute mean
    rho = rho.mean(-1)
    rho_emb = rho_emb.mean(-1)
    rho_bme = rho_bme.mean(-1)
    
    # if required, apply spearman brown correction
    if task.correct:
        rho = (2 * rho) / (1 + rho)
        rho_emb = (2 * rho_emb) / (1 + rho_emb)
        rho_bme = (2 * rho_bme) / (1 + rho_bme)
    
    # if required, make noise ceiling
    if task.ceiling:
        rho = torch.sqrt(
            torch.clamp(rho, min = 0.0, max = 1.0)
        )
        
        rho_emb = torch.sqrt(
            torch.clamp(rho_emb, min = 0.0, max = 1.0)
        )
        
        rho_bme = torch.sqrt(
            torch.clamp(rho_bme, min = 0.0, max = 1.0)
        )
    
    # prepare outputs
    outputs = {
        'subject_nos': unq_subject_nos,
        'subject_ids': unq_subjects,
        'embeddings': list(arrangements.embeddings.keys()),
        'rho': rho,
        'rho_emb': rho_emb,
        'rho_bme': rho_bme
    }
    
    # save data
    save_supplement(outputs, 'macrossreliability', cfg, logger)