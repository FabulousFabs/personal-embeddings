"""Space.

Contains the arrangement spatial tasks.
"""

from omegaconf import DictConfig, OmegaConf
from loguru import logger

from pathlib import Path

import numpy as np
import pandas as pd
import csv
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

def arrangement_space(task: DictConfig, data: Dataset, cfg: DictConfig, logger: logger) -> None:
    """Perform an analysis of the kinds of changes in semantic space.
    
    .. warning::
        Because this function likely relies on SVD (as per the setup in our 
        paper at least), it is imperative that this function is run on the 
        exact same device as matrain. Otherwise, the estimated SVD will 
        likely differ between devices, making it impossible to apply the 
        personalisation.
    
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
        logger.info('Performing arrangement space exploration...')
    
    # check configuration for features
    n_features = [
        data.embeddings[emb][
            list(data.embeddings[emb].keys())[0]
        ].shape[-1]
        for emb in data.embeddings
    ]
    
    if cfg.data.preprocess.embeddings.svd:
        n_features = [cfg.data.preprocess.embeddings.rank] * len(data.embeddings)
    
    # load multiple arrangement data
    arrangements = data.load_arrangements(n_session = 1)
    
    # get functions
    gram_fn = metrics.get_gram_fn(cfg)
    
    # check subjects
    n_subjects = torch.unique(arrangements.subject_nos).shape[0]
    
    # setup tqdm
    pbar = tqdm(
        total = cfg.task.matrain.repeats * cfg.task.matrain.splits * len(data.embeddings) * cfg.task.matrain.permutations,
        leave = True,
        disable = (cfg.verbose == False)
    )
    
    # setup data containers
    RX = {
        emb: torch.zeros((70, n_features[i]), dtype = torch.float, device = cfg.device)
        for i, emb in enumerate(data.embeddings)
    }
    
    R1 = {
        emb: torch.zeros((n_subjects, 70), dtype = torch.float, device = cfg.device)
        for emb in data.embeddings
    }
    
    R2 = {
        emb: torch.zeros((n_subjects, 70), dtype = torch.float, device = cfg.device)
        for emb in data.embeddings
    }
    
    ER = {
        emb: torch.zeros((n_subjects, 70), dtype = torch.float, device = cfg.device)
        for emb in data.embeddings
    }
    
    N = {
        emb: torch.zeros((n_subjects, 70), dtype = torch.float, device = cfg.device)
        for emb in data.embeddings
    }
    
    # create configurations
    from hydra import compose
    from hydra.core.hydra_config import HydraConfig
    runtime = HydraConfig.get()
    
    overrides = [
        override
        for override in runtime.overrides.task
        if override.split("=", 1)[0] != "model"
    ]
    
    cfg_m1 = compose(
        config_name = runtime.job.config_name,
        overrides = [*overrides, f"model={task.compare.baseline}"]
    )
    
    cfg_m2 = compose(
        config_name = runtime.job.config_name,
        overrides = [*overrides, f"model={task.compare.contender}"]
    )
    
    # iterate over data splits
    for f_i, split in enumerate(arrangements.split(
        cfg.task.matrain.splits, cfg.task.matrain.repeats,
        seed = cfg.seed
    )):
        # preprocess data
        split.preprocess()
        
        # compute reliabilities
        split.train.compute_split_half_reliability()

        # iterate over embeddings
        for e_i, name in enumerate(data.embeddings):
            # iterate over permutations
            for p_i in range(cfg.task.matrain.permutations):
                # create model
                delta_m1 = model.create(
                    cfg_m1, name, split.repeat, split.split, p_i,
                    split.train.embeddings[name].shape[-1],
                    n_subjects, 
                    preload = True
                )
                
                delta_m1.eval()
                
                delta_m2 = model.create(
                    cfg_m2, name, split.repeat, split.split, p_i,
                    split.train.embeddings[name].shape[-1],
                    n_subjects, 
                    preload = True
                )
                
                delta_m2.eval()
                
                # obtain data
                for s_i, subject_i in enumerate(torch.unique(split.test.subject_nos)):
                    # select data splits
                    splits = []
                    
                    if task.data.train:
                        splits += [split.train]
                    if task.data.test:
                        splits += [split.test]
                    
                    for split_i in splits:
                        # mask subject data
                        mask_i = (split_i.subject_nos == subject_i)

                        # find boards
                        boards_i = torch.unique(split_i.board_nos[mask_i])
                        ids_i = split_i.board_ids[mask_i][
                            torch.tensor([
                                torch.where(split_i.board_nos[mask_i] == board_i)[0][0].item()
                                for board_i in boards_i
                            ], dtype = torch.long, device = cfg.device)
                        ]

                        # sort boards
                        order = ids_i.argsort()
                        boards_i = boards_i[order]
                        ids_i = ids_i[order]

                        # iterate over boards
                        for b_j, board_j in enumerate(boards_i):
                            # mask board data
                            mask_j = mask_i & (split_i.board_nos == board_j)

                            # grab data
                            embeddings = split_i.embeddings[name][mask_j]
                            contexts = split_i.contexts[name][mask_j]
                            positions = split_i.positions[mask_j]

                            # make order
                            word_order = split_i.word_nos[mask_j].argsort()
                            word_nos = split_i.word_nos[mask_j][word_order]
                            X_ij = embeddings[word_order]
                            C_ij = contexts[word_order,...][:,word_order,:]
                            B_ij = positions[word_order,:]
                            S_ij = torch.full((X_ij.shape[0],), subject_i, dtype = torch.long, device = cfg.device)

                            with torch.no_grad():
                                Z1_ij = delta_m1(X_ij, S_ij, C_ij)['final'].detach()
                                Z2_ij = delta_m2(X_ij, S_ij, C_ij)['final'].detach()

                            # compute gram
                            kwargs = {} if cfg.metrics.gram != 'euclidean' else {'norm': False}
                            DX = gram_fn(X_ij)
                            DB = gram_fn(B_ij, **kwargs)
                            D1 = gram_fn(Z1_ij)
                            D2 = gram_fn(Z2_ij)
                            
                            # remove diagonals
                            nd = DX.shape[0]
                            offd = ~torch.eye(nd, dtype = torch.bool, device = DX.device)
                            DX = DX[offd].reshape(nd, nd - 1)
                            DB = DB[offd].reshape(nd, nd - 1)
                            D1 = D1[offd].reshape(nd, nd - 1)
                            D2 = D2[offd].reshape(nd, nd - 1)
                            
                            # compute correlations with behaviour
                            R1[name][s_i,word_nos] += spearmanr(DB, D1)
                            R2[name][s_i,word_nos] += spearmanr(DB, D2)
                            ER[name][s_i,word_nos] += spearmanr(DB, DX)
                            
                            # compute counts
                            N[name][s_i,word_nos] += torch.ones((20,), dtype = torch.float, device = cfg.device)
                            
                            # save embeddings for later
                            RX[name][word_nos] = X_ij
                
                pbar.update(1)
    
    pbar.close()
    
    # compute aggregate statistics
    ER = {name: ER[name] / torch.clamp(N[name], min = 1) for name in data.embeddings}
    R1 = {name: R1[name] / torch.clamp(N[name], min = 1) for name in data.embeddings}
    R2 = {name: R2[name] / torch.clamp(N[name], min = 1) for name in data.embeddings}
    
    # read norm data
    df = pd.read_csv(
        Path(cfg.data.resources) / Path('vocab') / Path('norms.tsv'),
        sep = '\t', quoting = csv.QUOTE_NONE
    )
    
    # compute anchors
    anchors = [
        'nature', 'relationship', 'politics', 'cuisine', 'technology'
    ]
    
    indices = torch.tensor([
        np.where(df.word.to_numpy() == word)[0][0].item()
        for word in anchors
    ], device = cfg.device)
    
    # obtain interest in contexts
    I, L = arrangements.make_demographics(task.demographics, cfg)
    
    # save data
    r_full = {name: None for name in data.embeddings}
    r_reduced = {name: None for name in data.embeddings}
    ß_full = {name: None for name in data.embeddings}
    D_full = {name: {} for name in data.embeddings}
    
    r_full2 = {name: None for name in data.embeddings}
    r_perm2 = {name: None for name in data.embeddings}
    ß_full2 = {name: None for name in data.embeddings}
    ß_perm2 = {name: None for name in data.embeddings}
    I_full2 = {name: None for name in data.embeddings}
    I_perm2 = {name: None for name in data.embeddings}
    D_full2 = {name: {} for name in data.embeddings}
    
    r_full3 = {name: None for name in data.embeddings}
    r_redu3 = {name: None for name in data.embeddings}
    ß_full3 = {name: None for name in data.embeddings}
    I_full3 = {name: None for name in data.embeddings}
    D_full3 = {name: {} for name in data.embeddings}
    
    for i, name in enumerate(data.embeddings):
        # score words by context
        V = F.normalize(RX[name], dim = -1)
        A = F.normalize(V[indices], dim = -1)
        
        C = V @ A.mT
        C = (C - C.mean(0, keepdim = True)) / C.std(0, keepdim = True)
        S = C.clone()
        
        # make data
        X = S[None,:,:].expand(R1[name].shape[0], -1, -1)
        y = (R2[name] - R1[name])

        # prepare pipeline
        pipeline = make_pipeline(
            Scaler().to_torch(),
            RidgeCV(
                alphas = torch.logspace(-5, 5, 10, device = cfg.device)
            )
        )
        
        # prepare CV
        rkf = RepeatedKFold(
            n_splits = task.splits,
            n_repeats = task.repeats
        )
        
        # prepare arguments
        kwargs = dict(
            cv = rkf,
            metric = (mv_metrics.pearsonr,),
            n_jobs = task.jobs,
            verbose = False
        )
        
        # prepare data
        r_full[name] = torch.zeros((X.shape[0], task.splits * task.repeats), device = cfg.device)
        r_reduced[name] = torch.zeros((X.shape[0], X.shape[-1], task.splits * task.repeats), device = cfg.device)
        ß_full[name] = torch.zeros((X.shape[0], task.splits * task.repeats, X.shape[-1]), device = cfg.device)
        
        # fit full model
        validator, score = cross_val_score(
            pipeline, X[0], y.mT, **kwargs
        )
        
        r_full[name] = score.mT.clone()
        ß_full[name] = validator.collect('coef_', from_step = -1).permute(1, 0, 2).clone()
        
        # save data for later
        D_full[name]['X'] = X.clone()
        D_full[name]['y'] = y.mT
        D_full[name]['z'] = torch.zeros_like(y.mT)
        n = torch.zeros_like(y.mT)
        z = validator.predict(X[0])
        
        for j, test in enumerate(validator.test_):
            D_full[name]['z'][test,:] += z[j,test,:]
            n[test,:] += torch.ones_like(n[test,:])
        
        D_full[name]['z'] = D_full[name]['z'] / torch.clamp(n, min = 1)
        
        # fit reduced models
        for j in range(X.shape[-1]):
            not_j = torch.arange(X.shape[-1], device = cfg.device) != j
            
            validator_j, score_j = cross_val_score(
                pipeline, X[0][:,not_j], y.mT, **kwargs
            )
            
            r_reduced[name][:,j,:] = score_j.mT.clone()

        # prepare affinity computations
        X = I.clone()
        X = (X - X.mean(0, keepdim = True))
        
        y = ß_full[name].mean(1)
        y = (y - y.mean(0, keepdim = True))
        y_c = y.clone()
        
        H = torch.arange(I.shape[0], device = cfg.device)[:,None].expand(-1, X.shape[-1])
        
        # reshape the data
        X = X.reshape(-1)[:,None]
        y = y.reshape(-1)[:,None]
        G = H.reshape(-1)
        
        # prepare CV
        lgo = LeaveGroupsOut(
            n_splits = task.splits,
            n_repeats = task.repeats
        )
        
        # prepare arguments
        kwargs = dict(
            cv = lgo,
            metric = (mv_metrics.pearsonr,),
            n_jobs = task.jobs,
            verbose = False
        )
        
        # prepare tqdm
        pbar = tqdm(
            total = task.permutations,
            leave = True,
            disable = (cfg.verbose == False)
        )
        
        # prepare outputs
        r_full2[name] = torch.zeros((task.permutations, task.splits * task.repeats, 1), device = cfg.device)
        r_perm2[name] = torch.zeros((task.permutations, task.splits * task.repeats, 1), device = cfg.device)
        ß_full2[name] = torch.zeros((task.permutations, task.splits * task.repeats, 1, 1), device = cfg.device)
        ß_perm2[name] = torch.zeros((task.permutations, task.splits * task.repeats, 1, 1), device = cfg.device)
        I_full2[name] = torch.zeros((task.permutations, task.splits * task.repeats, 1, 1), device = cfg.device)
        I_perm2[name] = torch.zeros((task.permutations, task.splits * task.repeats, 1, 1), device = cfg.device)
        
        D_full2[name]['X'] = torch.zeros((task.permutations, *I.shape), device = cfg.device)
        D_full2[name]['y'] = torch.zeros((task.permutations, *y_c.shape), device = cfg.device)
        D_full2[name]['z'] = torch.zeros((task.permutations, *y_c.shape), device = cfg.device)
        
        # iterate
        for j in range(task.permutations):
            # fit full model
            validator, r_full2[name][j] = cross_val_score(
                pipeline, X, y, groups = G, **kwargs
            )
            
            ß_full2[name][j] = validator.collect('coef_', from_step = -1).clone()
            I_full2[name][j] = validator.collect('intercept_', from_step = -1).clone()
            
            # save data for later
            D_full2[name]['X'][j] = X.reshape(I.shape).clone()
            D_full2[name]['y'][j] = y.reshape(y_c.shape).clone()
            z = validator.predict(X)
            p = torch.zeros((z.shape[1], z.shape[2]), device = cfg.device)
            n = torch.zeros((z.shape[1], z.shape[2]), device = cfg.device)
            
            for k, test in enumerate(validator.test_):
                p[test,:] += z[k,test,:]
                n[test,:] += torch.ones_like(n[test,:])
            
            p = p / torch.clamp(n, min = 1)
            D_full2[name]['z'][j] = p.reshape(y_c.shape)

            # fit permuted model
            permutation = torch.randperm(y_c.shape[0], device = cfg.device)
            H_p = permutation[:,None].expand(-1, H.shape[-1])
            G_p = H_p.reshape(-1)
            y_p = y_c.clone()[permutation].reshape(-1)[:,None]
            
            validator, r_perm2[name][j] = cross_val_score(
                pipeline, X, y_p, groups = G_p, **kwargs
            )
            
            ß_perm2[name][j] = validator.collect('coef_', from_step = -1).clone()
            I_perm2[name][j] = validator.collect('intercept_', from_step = -1).clone()
            
            pbar.update(1)
        
        pbar.close()
        
        # make word norm data
        for source in task.norms.sources:
            if source not in list(df.keys()):
                raise AttributeError(
                    f'Unknown source {source} in word norms.'
                )
        
        X = torch.tensor([
            df[key].to_list()
            for key in task.norms.sources
        ], device = cfg.device).mT
        
        X_G = torch.arange(X.shape[-1], device = cfg.device)
        
        if task.norms.groups:
            if len(task.norms.groups) != len(task.norms.sources):
                raise ValueError(
                    f'For word norms, {len(task.norm.groups)} were given but ' + 
                    f'{len(task.norm.sources)} sources were found. Unclear ' +
                    f'grouping.'
                )
            
            X_G = torch.tensor(task.norms.groups, device = cfg.device)
        
        X_unq = torch.unique(X_G)
        X_n = X_unq.shape[0]
        
        y = (R2[name] - R1[name]).mT
        
        if task.norms.norm_y:
            y = (y / y.abs().sum(0, keepdim = True))
        
        # prepare pipeline
        pipeline = make_pipeline(
            Scaler().to_torch(),
            RidgeCV(alphas = torch.logspace(-5, 5, 10, device = cfg.device))
        )
        
        # prepare cv
        rkf = RepeatedKFold(
            n_splits = task.splits,
            n_repeats = task.repeats
        )
        
        # prepare keyword args
        kwargs = dict(
            cv = rkf,
            metric = (mv_metrics.pearsonr,),
            n_jobs = task.jobs,
            verbose = False
        )
        
        # prepare outputs
        r_full3[name] = torch.zeros((task.splits * task.repeats, y.shape[1]), device = cfg.device)
        r_redu3[name] = torch.zeros((X_n, task.splits * task.repeats, y.shape[1]), device = cfg.device)
        ß_full3[name] = torch.zeros((task.splits * task.repeats, y.shape[1], X.shape[-1]), device = cfg.device)    
        I_full3[name] = torch.zeros((task.splits * task.repeats, 1, y.shape[1]), device = cfg.device)    
        
        # fit full model for all participants
        validator, r_full3[name] = cross_val_score(
            pipeline, X, y, **kwargs
        )
        
        ß_full3[name] = validator.collect('coef_', from_step = -1).clone()
        I_full3[name] = validator.collect('intercept_', from_step = -1).clone()
        D_full3[name]['X'] = X.clone()
        D_full3[name]['y'] = y.clone()
        z = validator.predict(X)
        h = torch.zeros_like(y)
        n = torch.zeros_like(y)
        
        for i, test in enumerate(validator.test_):
            h[test,:] += z[i,test,:]
            n[test,:] += torch.ones_like(h[test,:], device = cfg.device)
        
        h = h / torch.clamp(n, min = 1)
        D_full3[name]['z'] = h.clone()
        
        # fit reduced models
        for i, unq in enumerate(X_unq):
            not_i = X_G != unq
            
            validator_i, r_redu3[name][i] = cross_val_score(
                pipeline, X[:,not_i], y, **kwargs
            )
    
    # prepare outputs
    outputs = {
        'S1_r_full': r_full,
        'S1_r_redu': r_reduced,
        'S1_ß_full' : ß_full,
        'S1_D': D_full,
        
        'S2_r_full': r_full2,
        'S2_r_perm': r_perm2,
        'S2_ß_full': ß_full2,
        'S2_ß_perm': ß_perm2,
        'S2_I_full': I_full2,
        'S2_I_perm': I_perm2,
        'S2_D': D_full2,
        
        'S3_r_full': r_full3,
        'S3_r_redu': r_redu3,
        'S3_ß_full': ß_full3,
        'S3_I_full': I_full3,
        'S3_D': D_full3,
        'S3_L': task.norms.sources,
        'S3_G': task.norms.groups,
        
        'r_eb': ER,
        'r_1b': R1,
        'r_2b': R2,
        
        'X': RX
    }
    
    # save data
    save_supplement(outputs, 'maspace', cfg, logger)