"""Demographics.

Contains the arrangement demographics tasks.
"""

from omegaconf import DictConfig, OmegaConf
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from mvpy.math import pearsonr, spearmanr
from mvpy.estimators import RidgeEncoder, RidgeCV
from mvpy.preprocessing import Scaler
from mvpy.crossvalidation import RepeatedKFold, LeaveGroupsOut, cross_val_score
from mvpy import metrics as mv_metrics
from sklearn.pipeline import make_pipeline
from tqdm import tqdm

from ..dataset import Dataset, ArrangementsSplit
from .. import metrics
from .. import model
from .results import save_supplement

from typing import Tuple, Dict

def arrangement_subemb(task: DictConfig, data: Dataset, cfg: DictConfig, logger: logger) -> None:
    """Perform an embedding-human correspondence analysis.
    
    Parameters
    ----------
    task : DictConfig
        Hydra subconfiguration for this specific training run.
    data : Dataset
        The preloaded dataset.
    cfg : DictConfig
        Hydra configuration.
    logger : logger
        Logger object.
    """
    
    # log progress
    if cfg.verbose:
        logger.info('Performing arrangement subject-embedding...')
    
    # get functions
    gram_fn = metrics.get_gram_fn(cfg)
    
    # load multiple arrangement data
    arrangements = data.load_arrangements(n_session = 1)
    
    # check subjects
    n_subjects = torch.unique(arrangements.subject_nos).shape[0]
    
    # check boards
    n_boards = torch.unique(arrangements.board_nos).shape[0]
    
    # check items
    n_items = (
        (arrangements.subject_nos == arrangements.subject_nos[0]) & \
        (arrangements.board_nos == arrangements.board_nos[0])
    ).sum().cpu().item()
    
    # setup board-wise RDMs
    rdm_f = torch.zeros(
        (n_subjects, n_boards, n_items, n_items),
        device = cfg.device
    )
    
    rdm_e = {
        key: torch.zeros_like(rdm_f)
        for key in arrangements.embeddings
    }
    
    # setup progressbar
    pbar = tqdm(
        total = n_subjects * n_boards,
        leave = True,
        disable = (cfg.verbose == False)
    )
    
    # iterate over subjects
    for i, subject_i in enumerate(torch.unique(arrangements.subject_ids)):
        # make board id placeholder
        is_first = True
        
        # iterate over boards
        for j, board_i in enumerate(torch.unique(arrangements.board_nos)):
            # create mask
            mask_ij = (arrangements.subject_ids == subject_i) & \
                      (arrangements.board_nos == board_i)
            
            # grab board identity
            board_idx = arrangements.board_ids[mask_ij][0].cpu().item()
            board_idx = board_idx + 1 if ((board_idx == 13) and (not is_first)) else board_idx
            is_first = False if board_idx == 13 else is_first
            
            # prepare behavioural data
            order = arrangements.word_nos[mask_ij].argsort()
            pos_ij = arrangements.positions[mask_ij][order,:]
            
            # make gram matrices
            kwargs = {} if cfg.metrics.gram != 'euclidean' else {'norm': False}
            rdm_f[i,board_idx] = gram_fn(pos_ij, **kwargs)
            
            # iterate over embeddings
            for k, key in enumerate(arrangements.embeddings):
                # prepare embedding data
                emb_ij = F.normalize(arrangements.embeddings[key][mask_ij], dim = -1)[order,:]
                
                # make gram matrices
                rdm_e[key][i,board_idx] = gram_fn(emb_ij, **kwargs)
            
            # update tally
            pbar.update(1)
    
    # make sure tally is closed
    pbar.close()
    
    # compute subject-embedding similarities
    tri_x, tri_y = torch.triu_indices(
        n_items, n_items, 
        offset = 1,
        device = cfg.device
    )
    
    cross = {
        key: torch.stack([
            spearmanr(
                rdm_f[i,:,tri_x,tri_y],
                rdm_e[key][i,:,tri_x,tri_y]
            )
            for i in tqdm(
                range(n_subjects),
                leave = True,
                disable = (cfg.verbose == False)
            )
        ], dim = 0)
        for key in arrangements.embeddings
    } # (dict(key: (n_subjects, n_boards)))
    
    # obtain regressor set
    n_demo = len(list(task.demographics.keys()))
    X, L = arrangements.make_demographics(task.demographics, cfg)
    
    # setup model pipeline
    model = make_pipeline(
        Scaler().to_torch(),
        RidgeEncoder(
            alphas = torch.logspace(-5, 5, 10, device = cfg.device)
        )
    )
    
    # setup cross-validation scheme
    cv = RepeatedKFold(
        n_splits = task.splits,
        n_repeats = task.repeats
    )
    
    # allocate resources
    r_full = {name: torch.zeros((task.permutations,), device = cfg.device) for name in arrangements.embeddings}
    r_base = {name: torch.zeros((task.permutations,), device = cfg.device) for name in arrangements.embeddings}
    r_perm = {name: torch.zeros((task.permutations, n_demo), device = cfg.device) for name in arrangements.embeddings}
    r_ceil = {name: torch.zeros((task.permutations,), device = cfg.device) for name in arrangements.embeddings}
    
    ß_full = {name: torch.zeros((task.permutations, X.shape[1]), device = cfg.device) for name in arrangements.embeddings}
    ß_perm = {name: torch.zeros((task.permutations, X.shape[1], n_demo), device = cfg.device) for name in arrangements.embeddings}
    
    # count iterations
    n_total = len(rdm_e) * (
        task.permutations * n_demo + 2 * task.permutations
    )
    
    # setup progressbar
    pbar = tqdm(
        total = n_total,
        leave = True,
        disable = (cfg.verbose == False)
    )
    
    metric = (mv_metrics.pearsonr,)
    
    # iterate over embeddings
    for e_i, embedding in enumerate(rdm_e):
        # get outcomes
        y_e = cross[embedding].mean(1)[:,None]
        
        # iterate over permutations
        for p_i in range(task.permutations):
            # iterate over predictors
            for p_j, name in enumerate(task.demographics):
                # make permuted data
                X_p, _ = arrangements.make_demographics(task.demographics, cfg, permute = [name])
                
                # perform model scoring with permuted predictor
                validator, score_eij = cross_val_score(
                    model, X_p, y_e,
                    cv = cv, metric = metric,
                    verbose = False, n_jobs = task.jobs
                )
                
                # write data
                r_perm[embedding][p_i,p_j] = score_eij.mean()
                ß_perm[embedding][p_i,:,p_j] = validator.collect(
                    'coef_', from_step = -1
                ).mean(0)
                
                # tally
                pbar.update(1)
            
            # perform model scoring for full model
            validator, score_ei = cross_val_score(
                model, X, y_e,
                cv = cv, metric = metric,
                verbose = False, n_jobs = task.jobs
            )
            
            # write data
            r_full[embedding][p_i] = score_ei.mean()
            ß_full[embedding][p_i,:] = validator.collect(
                'coef_', from_step = -1
            ).mean(0)
            
            pbar.update(1)
            
            # perform model scoring for fully permuted model
            permutation = torch.randperm(y_e.shape[0], device = cfg.device)
            
            y_p = y_e[permutation]
            _, score_ei = cross_val_score(
                model, X, y_p,
                cv = cv, metric = metric,
                verbose = False, n_jobs = task.jobs
            )
            
            # write data
            r_base[embedding][p_i] = score_ei.mean()
            
            pbar.update(1)
            
            # compute noise ceiling
            boards = torch.randperm(n_boards, device = cfg.device)
            
            split_A = cross[embedding][:,boards[0:n_boards//2]].mean(1)
            split_B = cross[embedding][:,boards[n_boards//2:]].mean(1)
            r_ceil[embedding][p_i] = pearsonr(
                split_A,
                split_B
            )
    
    # make sure progressbar is closed
    pbar.close()
    
    # correct noise ceiling
    for embedding in arrangements.embeddings:
        r_ceil[embedding] = (2 * r_ceil[embedding]) / (1 + r_ceil[embedding])
        r_ceil[embedding] = torch.sqrt(r_ceil[embedding])
    
    # make outputs
    outputs = {
        'labels': L,
        'r_ceil': r_ceil,
        'r_full': r_full,
        'r_base': r_base,
        'r_perm': r_perm,
        'ß_full': ß_full,
        'ß_perm': ß_perm
    }
    
    # save data
    save_supplement(outputs, 'masubemb', cfg, logger)

def arrangement_subsub(task: DictConfig, data: Dataset, cfg: DictConfig, logger: logger) -> None:
    """Perform a human-human deviation correspondence analysis.
    
    Parameters
    ----------
    task : DictConfig
        Hydra subconfiguration for this specific training run.
    data : Dataset
        The preloaded dataset.
    cfg : DictConfig
        Hydra configuration.
    logger : logger
        Logger object.
    """
    
    # log progress
    if cfg.verbose:
        logger.info('Performing arrangement subject-subject...')
    
    # get functions
    gram_fn = metrics.get_gram_fn(cfg)
    
    # load multiple arrangement data
    arrangements = data.load_arrangements(n_session = 1)
    
    # check subjects
    n_subjects = torch.unique(arrangements.subject_nos).shape[0]
    
    # check boards
    n_boards = torch.unique(arrangements.board_nos).shape[0]
    
    # check items
    n_items = (
        (arrangements.subject_nos == arrangements.subject_nos[0]) & \
        (arrangements.board_nos == arrangements.board_nos[0])
    ).sum().cpu().item()
    
    # setup board-wise RDMs
    rdm_f = torch.zeros(
        (n_subjects, n_boards, n_items, n_items),
        device = cfg.device
    )
    
    # setup progressbar
    pbar = tqdm(
        total = n_subjects * n_boards,
        leave = True,
        disable = (cfg.verbose == False)
    )
    
    # iterate over subjects
    for i, subject_i in enumerate(torch.unique(arrangements.subject_ids)):
        # make board id placeholder
        is_first = True
        
        # iterate over boards
        for j, board_i in enumerate(torch.unique(arrangements.board_nos)):
            # create mask
            mask_ij = (arrangements.subject_ids == subject_i) & \
                      (arrangements.board_nos == board_i)
            
            # grab board identity
            board_idx = arrangements.board_ids[mask_ij][0].cpu().item()
            board_idx = board_idx + 1 if ((board_idx == 13) and (not is_first)) else board_idx
            is_first = False if board_idx == 13 else is_first
            
            # prepare behavioural data
            order = arrangements.word_nos[mask_ij].argsort()
            pos_ij = arrangements.positions[mask_ij][order,:]
            
            # make gram matrices
            kwargs = {} if cfg.metrics.gram != 'euclidean' else {'norm': False}
            rdm_f[i,board_idx] = gram_fn(pos_ij, **kwargs)
            
            # update tally
            pbar.update(1)
    
    # make sure tally is closed
    pbar.close()
    
    # setup preprocessing
    from mvpy.math import rank
    preprocess = Scaler().to_torch()
    
    tri_x, tri_y = torch.triu_indices(
        n_items, n_items,
        offset = 1,
        device = cfg.device
    )
    
    pre_f = torch.zeros(
        (n_boards, tri_x.shape[0], n_subjects),
        device = cfg.device
    )
    
    for i, subject_i in enumerate(torch.unique(arrangements.subject_ids)):
        pre_f[:,:,i] = preprocess.fit_transform(
            rank(
                rdm_f[i,:,tri_x,tri_y]
            ).T
        ).T
    
    # setup residualised procedure
    residualise = make_pipeline(
        RidgeEncoder(
            alphas = torch.tensor([1e-5], device = cfg.device)
        )
    )
    
    # setup data structure
    res_f = torch.zeros(
        (n_boards, tri_x.shape[0], n_subjects, n_subjects),
        device = cfg.device
    )
    
    res_ß = torch.zeros(
        (n_subjects, n_subjects, n_boards),
        device = cfg.device
    )
    
    out_f = torch.zeros(
        (n_boards, n_subjects, n_subjects),
        device = cfg.device
    )
    
    pbar = tqdm(
        total = n_subjects * n_subjects,
        leave = True,
        disable = (cfg.verbose == False)
    )
    
    # residualise rdms
    for i, subject_i in enumerate(torch.unique(arrangements.subject_ids)):
        for j, subject_j in enumerate(torch.unique(arrangements.subject_ids)):
            # setup mask
            not_ij = torch.arange(n_subjects, device = cfg.device)
            not_ij = (not_ij != i) & (not_ij != j)
            
            # make template
            X_ij = pre_f[:,:,not_ij].mean(-1)
            
            # setup data
            i_h, j_h = [], []
            i_ß, j_ß = [], []
            
            # iterate over boards
            for k in range(n_boards):
                # prepare data
                X_ijk = X_ij[k,:,None]
                y_ijk = torch.cat([
                    pre_f[k,:,i,None],
                    pre_f[k,:,j,None]
                ], dim = -1)
                
                # fit models
                residualise.fit(X_ijk, y_ijk)
                
                # residualise
                r_ijk = y_ijk
                if residualise:
                    r_ijk = r_ijk - residualise.predict(X_ijk)
                
                # save data
                i_h.append(r_ijk[:,0])
                j_h.append(r_ijk[:,1])
                i_ß.append(residualise[-1].coef_.clone()[0,0].item())
                j_ß.append(residualise[-1].coef_.clone()[1,0].item())
            
            # stack
            res_f[:,:,i,j] = torch.stack(i_h, dim = 0)
            res_f[:,:,j,i] = torch.stack(j_h, dim = 0)
            res_ß[i,j,:] = torch.tensor(i_ß, device = cfg.device)
            res_ß[j,i,:] = torch.tensor(j_ß, device = cfg.device)
            
            # compare rdms
            out_f[:,j,i] = out_f[:,i,j] = spearmanr(res_f[:,:,i,j], res_f[:,:,j,i])
            
            # tally
            pbar.update(1)
    
    # close tally
    pbar.close()
    
    # prepare data
    X, L = arrangements.make_demographics(task.demographics, cfg)
    X = X
    y = out_f.mean(0)
    tri_x, tri_y = torch.triu_indices(n_subjects, n_subjects, offset = 1, device = cfg.device)
    
    # allocate resources
    r_full = torch.zeros((task.permutations,), device = cfg.device)
    r_base = torch.zeros((task.permutations,), device = cfg.device)
    r_perm = torch.zeros((task.permutations, X.shape[-1]), device = cfg.device)
    r_ceil = torch.zeros((task.permutations,), device = cfg.device)
    
    ß_full = torch.zeros((task.permutations, X.shape[-1]), device = cfg.device)
    ß_perm = torch.zeros((task.permutations, X.shape[-1], len(L)), device = cfg.device)
    
    # setup progressbar
    pbar = tqdm(
        total = task.permutations,
        leave = True,
        disable = (cfg.verbose == False)
    )
    
    # setup model pipeline
    model = make_pipeline(
        Scaler().to_torch(),
        RidgeEncoder(
            alphas = torch.logspace(-5, 5, 10, device = cfg.device)
        )
    )
    
    # setup cross-validation scheme
    cv = LeaveGroupsOut(
        n_splits = task.splits,
        n_repeats = task.repeats
    )
    
    groups = torch.zeros((n_subjects, n_subjects, 2), device = X.device)
    
    for i in range(n_subjects):
        for j in range(n_subjects):
            groups[i,j,0] = i
            groups[i,j,1] = j
    
    # iterate over permutations
    for p_i in range(task.permutations):
        # iterate over predictors
        for p_j, name in enumerate(task.demographics):
            # make permuted data
            X_p, _ = arrangements.make_demographics(task.demographics, cfg, permute = [name])
            
            # perform model scoring with permuted predictor
            validator, score_ij = cross_val_score(
                model, X_p[tri_x,tri_y,:], y[tri_x,tri_y,None], groups = groups[tri_x,tri_y],
                cv = cv, metric = mv_metrics.pearsonr,
                verbose = False, n_jobs = task.jobs
            )
            
            # write data
            r_perm[p_i,p_j] = score_ij.mean()
            ß_perm[p_i,:,p_j] = validator.collect(
                'coef_', from_step = -1
            ).mean(0)
        
        # perform model scoring for full model
        validator, score_i = cross_val_score(
            model, X[tri_x,tri_y,:], y[tri_x,tri_y,None], groups = groups[tri_x,tri_y],
            cv = cv, metric = mv_metrics.pearsonr,
            verbose = False, n_jobs = task.jobs
        )
        
        # write data
        r_full[p_i] = score_i.mean()
        ß_full[p_i] = validator.collect(
            'coef_', from_step = -1
        ).mean(0)
        
        # perform model scoring for fully permuted model
        X_p, _ = arrangements.make_demographics(task.demographics, cfg, permute = list(task.demographics.keys()))
        
        validator, score_i = cross_val_score(
            model, X_p[tri_x,tri_y,:], y[tri_x,tri_y,None], groups = groups[tri_x,tri_y],
            cv = cv, metric = mv_metrics.pearsonr,
            verbose = False, n_jobs = task.jobs
        )
        
        # write data
        r_base[p_i] = score_i.mean()
        
        # compute noise ceiling
        permutation = torch.randperm(n_boards, device = cfg.device)
        
        r_ceil[p_i] = pearsonr(
            out_f[permutation[0:n_boards//2]].mean(0)[tri_x,tri_y],
            out_f[permutation[n_boards//2:]].mean(0)[tri_x,tri_y]
        )
        
        # tally
        pbar.update(1)
    
    # make sure progressbar is closed
    pbar.close()
    
    # correct ceiling
    r_ceil = (2 * r_ceil) / (1 + r_ceil)
    r_ceil = torch.sqrt(r_ceil)
    
    # make outputs
    outputs = {
        'labels': L,
        'r_ceil': r_ceil,
        'r_full': r_full,
        'r_base': r_base,
        'r_perm': r_perm,
        'ß_full': ß_full,
        'ß_perm': ß_perm
    }
    
    # save data
    save_supplement(outputs, 'masubsub', cfg, logger)