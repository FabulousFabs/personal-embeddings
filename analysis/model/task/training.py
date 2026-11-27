"""Training.

Contains the arrangement training and (cross-)evaluation tasks.
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

def arrangement_train_single(delta: nn.Module, split: ArrangementsSplit, name: str, cfg: DictConfig) -> Tuple[nn.Module, Dict[str, torch.Tensor | int | float]]:
    """Perform a single training run for one model over arrangement data.
    
    Parameters
    ----------
    delta : nn.Module
        The network to train.
    split : ArrangementsSplit
        The arrangement data split into train, valid and test sets.
    name : str
        The name of the embedding to use for this training run.
    cfg : DictConfig
        Hydra configuration.
    
    Returns
    -------
    delta : nn.Module
        The network that was trained.
    outputs : Dict[str, torch.Tensor | int | float]
        The output dictionary that includes all relevant training metrics.
    """
    
    # setup optimiser
    optim = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, delta.parameters()),
        lr = cfg.model.training.optimiser.lr,
        weight_decay = cfg.model.training.optimiser.decay
    )
    
    # setup dataloaders
    opts = [cfg.device, cfg.model.training.batch_size, name]
    train = split.train.make_loader(*opts, shuffle = True)
    valid = split.valid.make_loader(*opts, shuffle = False)
    
    # setup tracking
    loss_best = torch.inf
    epoch_best = 0
    params = None
    waiting = 0
    history_train, history_valid = [], []
    
    # if required, setup noise parameters
    if cfg.model.training.denoising.enable:
        sigma_x = cfg.model.training.denoising.sigma_X * split.train.embeddings[name].std(0, keepdim = True)
        sigma_c = sigma_x.clone().unsqueeze(0)
        noise_x = None
        noise_c = None
    
    # iterate over epochs
    for ep in range(cfg.model.training.epochs):
        # set to train
        delta.train()
        
        # reset loss tracking
        loss_total, loss_n = 0.0, 0
        
        # iterate over training
        for (subjects, boards, words, embeddings, contexts, positions) in train:
            # if required, facilitate denoising in forward pass
            if cfg.model.training.denoising.enable:
                # check noise buffers
                if noise_x is None or noise_x.shape != embeddings.shape:
                    noise_x = torch.empty_like(embeddings)
                
                if noise_c is None or noise_c.shape != contexts.shape:
                    noise_c = torch.empty_like(contexts)
                    noise_c.normal_()
                
                # generate noise
                # NOTE: Here, noise is generated on the fly only for
                # embeddings - this is because the context noise is
                # extremely expensive to compute compared to the 
                # context noise.
                noise_x.normal_()
                
                # forward pass
                outputs = delta(
                    embeddings + sigma_x * noise_x,
                    subjects,
                    contexts + sigma_c * noise_c
                )
            else:
                # otherwise, just compute regular forward pass
                outputs = delta(embeddings, subjects, contexts)
            
            # backward pass
            loss = metrics.loss(
                {
                    'subjects': subjects, 
                    'boards': boards, 
                    'words': words, 
                    'embeddings': embeddings, 
                    'contexts': contexts, 
                    'positions': positions,
                    'ß': split.train.ceiling
                },
                outputs,
                cfg
            )
            
            # NOTE: In extremely rare cases, it is technically possible 
            # for loss to be undefined/float-0 (if not enough items were
            # sampled in a batch). We guard for this possibility here.
            if isinstance(loss, float):
                continue
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            # track loss
            loss_total += loss.item() * subjects.shape[0]
            loss_n += subjects.shape[0]
                
        # take average loss
        loss_train = loss_total / loss_n
        
        # check patience procedure
        if cfg.model.training.patience.enable == False:
            # in this case, we can fully skip the validation
            # step and resume training
            if cfg.verbose:
                print(f'ep{ep:03d}: {loss_train}')
            
            history_train.append(loss_train)
            epoch_best = ep
            params = {
                k: v.detach().cpu().clone()
                for k, v in delta.state_dict().items()
            }
            continue
        
        # set to eval
        delta.eval()
        
        # reset loss tracking
        loss_total, loss_n = 0.0, 0
        
        # iterate over validation
        with torch.no_grad():
            for (subjects, boards, words, embeddings, contexts, positions) in valid:
                # forward pass
                outputs = delta(embeddings, subjects, contexts)
                
                # backward pass
                loss = metrics.loss(
                    {
                        'subjects': subjects, 
                        'boards': boards, 
                        'words': words, 
                        'embeddings': embeddings, 
                        'contexts': contexts, 
                        'positions': positions,
                        'ß': split.train.ceiling
                    },
                    outputs,
                    cfg
                )
                
                # NOTE: In extremely rare cases, it is technically possible 
                # for loss to be undefined/float-0 (if not enough items were
                # sampled in a batch). We guard for this possibility here.
                if isinstance(loss, float):
                    continue

                # track loss
                loss_total += loss.item() * subjects.shape[0]
                loss_n += subjects.shape[0]
        
        # take average loss
        loss_valid = loss_total / loss_n
        
        # track losses
        history_train.append(loss_train)
        history_valid.append(loss_valid)
        
        if cfg.verbose:
            print(f'ep{ep:03d}: {loss_train}\t{loss_valid}')
        
        # check patience
        burnt_in = ep >= cfg.model.training.patience.burn_in

        if (loss_valid < loss_best):
            loss_best = loss_valid
            epoch_best = ep
            waiting = 0
            params = {
                k: v.detach().cpu().clone()
                for k, v in delta.state_dict().items()
            }
        elif burnt_in:
            waiting += 1

            if waiting > cfg.model.training.patience.wait:
                break
        
    # make histories
    history_train = torch.tensor(history_train).float()
    if cfg.model.training.patience.enable == False:
        history_valid = torch.zeros_like(history_train).float()
    else:
        history_valid = torch.tensor(history_valid).float()
    
    # load best state
    delta.load_state_dict(params, strict = True)
    delta = delta.to(cfg.device)
    
    # setup outputs
    outputs = {
        'loss_train': history_train,
        'loss_valid': history_valid,
        'epoch_best': epoch_best,
        'epoch_exit': ep,
        'boards_train': torch.unique(split.train.board_ids),
        'boards_valid': torch.unique(split.valid.board_ids),
        'noise_ceiling': split.train.ceiling.detach().cpu().clone()
    }
    
    return delta, outputs

def arrangement_eval_single(delta: nn.Module, split: ArrangementsSplit, name: str, cfg: DictConfig) -> Dict[str, torch.Tensor | int | float]:
    """Perform a single evaluation run for a given model over arrangement data.
    
    Parameters
    ----------
    delta : nn.Module
        The trained network.
    split : ArrangementsSplit
        The arrangement data split into train, valid and test sets.
    name : str
        The name of the embedding to use.
    cfg : DictConfig
        Hydra configuration.
    
    Returns
    -------
    outputs : Dict[str, torch.Tensor | int | float]
        The output dictionary that includes all relevant evaluation metrics.
    """
    
    # get functions
    gram_fn = metrics.get_gram_fn(cfg)
    
    # get descriptives
    n_subjects = split.test.subject_nos.max() + 1
    n_boards = torch.unique(split.test.board_nos[split.test.subject_nos == split.test.subject_nos[0]]).shape[0]

    # setup correlations
    rho_beh_ind = torch.zeros((n_subjects, n_subjects, n_boards), dtype = torch.float, device = cfg.device)
    rho_beh_emb = torch.zeros((n_subjects, n_subjects, n_boards), dtype = torch.float, device = cfg.device)
    rho_ind_emb = torch.zeros((n_subjects, n_subjects, n_boards), dtype = torch.float, device = cfg.device)
    
    # setup unseen pair correlations
    pns_rho_beh_ind = torch.zeros((n_subjects, n_subjects, n_boards), dtype = torch.float, device = cfg.device) * torch.nan
    pns_rho_beh_emb = torch.zeros((n_subjects, n_subjects, n_boards), dtype = torch.float, device = cfg.device) * torch.nan
    pns_rho_ind_emb = torch.zeros((n_subjects, n_subjects, n_boards), dtype = torch.float, device = cfg.device) * torch.nan
    pns_count = torch.zeros((n_subjects, n_subjects, n_boards), dtype = torch.float, device = cfg.device)
    
    # setup unseen word correlations
    uns_rho_beh_ind = torch.zeros((n_subjects, n_subjects, n_boards), dtype = torch.float, device = cfg.device) * torch.nan
    uns_rho_beh_emb = torch.zeros((n_subjects, n_subjects, n_boards), dtype = torch.float, device = cfg.device) * torch.nan
    uns_rho_ind_emb = torch.zeros((n_subjects, n_subjects, n_boards), dtype = torch.float, device = cfg.device) * torch.nan
    uns_count = torch.zeros((n_subjects, n_subjects, n_boards), dtype = torch.float, device = cfg.device)
    
    # find pairs in training set
    pairing = lambda x, y: 0.5 * (min(x, y) + max(x, y)) * (min(x, y) + max(x, y) + 1) + max(x, y)
    pairs_train = torch.tensor([
        [
            [
                pairing(no_i.item(), no_j.item())
                for no_j in split.train.word_nos[
                    (split.train.subject_nos == split.train.subject_nos[0]) & 
                    (split.train.board_nos == board_no)
                ]
            ]
            for no_i in split.train.word_nos[
                (split.train.subject_nos == split.train.subject_nos[0]) & 
                (split.train.board_nos == board_no)
            ]
        ]
        for board_no in torch.unique(split.train.board_nos[
            split.train.subject_nos == split.train.subject_nos[0]
        ])
    ], dtype = torch.long, device = cfg.device).view(-1)
    pairs_train = torch.unique(pairs_train)
    
    # setup eval mode
    delta.eval()
    
    n_words = max(
        split.train.word_nos.max().item(),
        split.valid.word_nos.max().item(),
        split.test.word_nos.max().item()
    ) + 1
    
    with torch.no_grad():
        # iterate over subjects
        for i, subject_i in enumerate(torch.unique(split.test.subject_nos)):
            # mask subject data
            mask_i = (split.test.subject_nos == subject_i)
            
            # find boards
            boards_i = torch.unique(split.test.board_nos[mask_i])
            ids_i = split.test.board_ids[mask_i][
                torch.tensor([
                    torch.where(split.test.board_nos[mask_i] == board_i)[0][0].item()
                    for board_i in boards_i
                ], dtype = torch.long, device = cfg.device)
            ]
            
            # sort boards
            order = ids_i.argsort()
            boards_i = boards_i[order]
            ids_i = ids_i[order]
            
            # iterate over boards
            for j, board_j in enumerate(boards_i):
                # mask board data
                mask_j = mask_i & (split.test.board_nos == board_j)
                
                # grab data
                embeddings = split.test.embeddings[name][mask_j]
                contexts = split.test.contexts[name][mask_j]
                positions = split.test.positions[mask_j]
                
                # find words that are unique in test set
                words_test = torch.unique(split.test.word_nos[mask_j])
                words_valid = torch.unique(split.valid.word_nos)
                words_train = torch.unique(split.train.word_nos)
                if cfg.model.training.patience.enable == False:
                    b_unseen = torch.isin(words_test, words_train) == False
                else:
                    b_unseen = (torch.isin(words_test, words_train) | torch.isin(words_test, words_valid)) == False
                words_unseen = words_test[b_unseen]
                
                # make order
                word_order = split.test.word_nos[mask_j].argsort()
                
                # make pair structure
                pairs = torch.tensor([
                    [
                        pairing(word_i.item(), word_j.item())
                        for word_j in 
                        split.test.word_nos[mask_j][word_order]
                    ]
                    for word_i in 
                    split.test.word_nos[mask_j][word_order]
                ], dtype = torch.long, device = cfg.device)
                
                # check novel pairs
                is_novel_pair = torch.isin(pairs, pairs_train) == False
                
                # iterate over subjects
                for k, subject_k in enumerate(torch.unique(split.test.subject_nos)):
                    # make subject
                    subjects = torch.full((embeddings.shape[0],), subject_k, dtype = torch.long, device = cfg.device)
                    
                    # forward pass
                    outputs = delta(embeddings, subjects, contexts)
                    
                    # make gram matrices
                    kwargs = {} if cfg.metrics.gram != 'euclidean' else {'norm': False}
                    observed_ijk = gram_fn(positions.clone()[word_order,:], **kwargs)
                    predicted_ijk = gram_fn(outputs['final'].detach().clone()[word_order,:])
                    hypothetical_ijk = gram_fn(embeddings.clone()[word_order,:])
                    
                    # compute correlations
                    tx, ty = torch.triu_indices(observed_ijk.shape[0], observed_ijk.shape[1], offset = 1, device = cfg.device)
                    rho_beh_ind[i,k,j] = spearmanr(observed_ijk[tx,ty], predicted_ijk[tx,ty])
                    rho_beh_emb[i,k,j] = spearmanr(observed_ijk[tx,ty], hypothetical_ijk[tx,ty])
                    rho_ind_emb[i,k,j] = spearmanr(predicted_ijk[tx,ty], hypothetical_ijk[tx,ty])
                    
                    # if possible, select only novel pairs
                    if is_novel_pair[tx,ty].sum() >= 3:
                        # make subselection
                        mask_k = is_novel_pair[tx,ty]
                        
                        # compute correlations
                        pns_rho_beh_ind[i,k,j] = spearmanr(observed_ijk[tx,ty][mask_k], predicted_ijk[tx,ty][mask_k])
                        pns_rho_beh_emb[i,k,j] = spearmanr(observed_ijk[tx,ty][mask_k], hypothetical_ijk[tx,ty][mask_k])
                        pns_rho_ind_emb[i,k,j] = spearmanr(predicted_ijk[tx,ty][mask_k], hypothetical_ijk[tx,ty][mask_k])

                        # add counts
                        pns_count[i,k,j] = mask_k.sum()
                    
                    # if possible, select only unseen items
                    if b_unseen.sum() >= 3:
                        # make mask
                        mask_k = torch.isin(split.test.word_nos[mask_j][word_order], words_unseen)
                        
                        # make gram matrices
                        kwargs = {} if cfg.metrics.gram != 'euclidean' else {'norm': False}
                        observed_ijk = gram_fn(positions.clone()[word_order,:][mask_k], **kwargs)
                        predicted_ijk = gram_fn(outputs['final'].detach().clone()[word_order,:][mask_k])
                        hypothetical_ijk = gram_fn(embeddings.clone()[word_order,:][mask_k])
                        
                        # compute correlations
                        tx, ty = torch.triu_indices(observed_ijk.shape[0], observed_ijk.shape[1], offset = 1, device = cfg.device)
                        uns_rho_beh_ind[i,k,j] = spearmanr(observed_ijk[tx,ty], predicted_ijk[tx,ty])
                        uns_rho_beh_emb[i,k,j] = spearmanr(observed_ijk[tx,ty], hypothetical_ijk[tx,ty])
                        uns_rho_ind_emb[i,k,j] = spearmanr(predicted_ijk[tx,ty], hypothetical_ijk[tx,ty])
                        
                        # add counts
                        uns_count[i,k,j] = b_unseen.sum()
        
        # prepare outputs
        outputs = {
            'boards_test': ids_i,
            'rho_BZ': rho_beh_ind,
            'rho_BE': rho_beh_emb,
            'rho_ZE': rho_ind_emb,
            'p_unseen_rho_BZ': pns_rho_beh_ind,
            'p_unseen_rho_BE': pns_rho_beh_emb,
            'p_unseen_rho_ZE': pns_rho_ind_emb,
            'p_unseen_count': pns_count,
            'unseen_rho_BZ': uns_rho_beh_ind,
            'unseen_rho_BE': uns_rho_beh_emb,
            'unseen_rho_ZE': uns_rho_ind_emb,
            'unseen_count': uns_count
        }
        
        # compute correlations
        diff = rho_beh_ind.mean(-1)
        eye = torch.eye(diff.shape[0], dtype = torch.bool, device = cfg.device)
        within = diff[eye]
        between = diff[eye == False].view(diff.shape[0], diff.shape[0] - 1).mean(1)
        baseline = rho_beh_emb.mean(-1)[eye]
        
        if cfg.verbose:
            tqdm.write(f'[{split.repeat}-{split.split}-all] base: {baseline.mean():.4f}\twthn: {within.mean():.4f}\tbtwn: {between.mean():.4f}\t|\tΔw: {(within - baseline).mean():.4f}\tΔb: {(between - baseline).mean():.4f} in boards {ids_i.tolist()}')
        
        # compute correlations
        diff = torch.nanmean(pns_rho_beh_ind, dim = -1)
        eye = torch.eye(diff.shape[0], dtype = torch.bool, device = cfg.device)
        within = diff[eye]
        between = torch.nanmean(diff[eye == False].view(diff.shape[0], diff.shape[0] - 1), dim = 1)
        baseline = torch.nanmean(pns_rho_beh_emb, dim = -1)[eye]
        
        if cfg.verbose:
            tqdm.write(f'[{split.repeat}-{split.split}-pns] base: {torch.nanmean(baseline):.4f}\twthn: {torch.nanmean(within):.4f}\tbtwn: {torch.nanmean(between):.4f}\t|\tΔw: {torch.nanmean(within - baseline):.4f}\tΔb: {torch.nanmean(between - baseline):.4f} in boards {ids_i.tolist()}')
        
        # compute correlations
        diff = torch.nanmean(uns_rho_beh_ind, dim = -1)
        eye = torch.eye(diff.shape[0], dtype = torch.bool, device = cfg.device)
        within = diff[eye]
        between = torch.nanmean(diff[eye == False].view(diff.shape[0], diff.shape[0] - 1), dim = 1)
        baseline = torch.nanmean(uns_rho_beh_emb, dim = -1)[eye]
        
        if cfg.verbose:
            tqdm.write(f'[{split.repeat}-{split.split}-uns] base: {torch.nanmean(baseline):.4f}\twthn: {torch.nanmean(within):.4f}\tbtwn: {torch.nanmean(between):.4f}\t|\tΔw: {torch.nanmean(within - baseline):.4f}\tΔb: {torch.nanmean(between - baseline):.4f} in boards {ids_i.tolist()}')
        
        return outputs
        
def arrangement_train(task: DictConfig, data: Dataset, cfg: DictConfig, logger: logger) -> None:
    """Perform a cross-validated training run.
    
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
        logger.info('Performing arrangement training...')
    
    # load multiple arrangement data
    arrangements = data.load_arrangements(n_session = 1)
    
    # check subjects
    n_subjects = torch.unique(arrangements.subject_nos).shape[0]
    
    # make data containers
    rho_NC = torch.zeros((n_subjects, len(data.embeddings), task.splits * task.repeats, task.permutations), dtype = torch.float, device = cfg.device)
    n_train = torch.zeros((n_subjects, len(data.embeddings), task.splits * task.repeats, task.permutations), dtype = torch.float, device = cfg.device)
    
    rho_BZ = torch.zeros((n_subjects, n_subjects, len(data.embeddings), task.splits * task.repeats, task.permutations), dtype = torch.float, device = cfg.device)
    rho_BE = torch.zeros((n_subjects, n_subjects, len(data.embeddings), task.splits * task.repeats, task.permutations), dtype = torch.float, device = cfg.device)
    rho_ZE = torch.zeros((n_subjects, n_subjects, len(data.embeddings), task.splits * task.repeats, task.permutations), dtype = torch.float, device = cfg.device)
    n_boards = torch.zeros((n_subjects, n_subjects, len(data.embeddings), task.splits * task.repeats, task.permutations), dtype = torch.float, device = cfg.device)
    n_pairs = torch.zeros((n_subjects, n_subjects, len(data.embeddings), task.splits * task.repeats, task.permutations), dtype = torch.float, device = cfg.device)
    n_words = torch.zeros((n_subjects, n_subjects, len(data.embeddings), task.splits * task.repeats, task.permutations), dtype = torch.float, device = cfg.device)
    
    weighted_sums = {}
    weight_sums = {}
    
    # setup progress
    pbar = tqdm(
        total = task.repeats * task.splits * len(data.embeddings) * task.permutations,
        leave = True,
        disable = (cfg.verbose == False)
    )
    
    # iterate over data splits
    for f_i, split in enumerate(arrangements.split(
        task.splits, task.repeats,
        seed = cfg.seed
    )):
        # preprocess data
        split.preprocess()
        
        # compute reliabilities
        split.train.compute_split_half_reliability()
        
        # iterate over embeddings
        for e_i, name in enumerate(data.embeddings):
            # iterate over permutations
            for p_i in range(task.permutations):
                # create model
                delta = model.create(
                    cfg, name, split.repeat, split.split, p_i,
                    split.train.embeddings[name].shape[-1],
                    n_subjects, 
                    preload = False
                )
                
                # train model
                delta, outputs_train = arrangement_train_single(delta, split, name, cfg)
                
                # evaluate model
                outputs_test = arrangement_eval_single(delta, split, name, cfg)
                
                # save ceilings
                rho_NC[:,e_i,f_i,p_i] = outputs_train['noise_ceiling']
                n_train[:,e_i,f_i,p_i] = outputs_train['boards_train'].shape[0]
                
                # make summaries for all boards
                rho_BZ[:,:,e_i,f_i,p_i] = outputs_test['rho_BZ'].mean(-1)
                rho_BE[:,:,e_i,f_i,p_i] = outputs_test['rho_BE'].mean(-1)
                rho_ZE[:,:,e_i,f_i,p_i] = outputs_test['rho_ZE'].mean(-1)
                n_boards[:,:,e_i,f_i,p_i] = outputs_test['boards_test'].shape[0]
                
                pair_counts = outputs_test['p_unseen_count']
                word_counts = outputs_test['unseen_count']

                # A board with m unseen words contributes m * (m - 1) / 2 distances.
                word_pair_counts = word_counts * (word_counts - 1) / 2

                for source, target, weights in (
                    ('p_unseen_', 'pns_', pair_counts),
                    ('unseen_', 'uns_', word_pair_counts),
                ):
                    for metric in ('rho_BZ', 'rho_BE', 'rho_ZE'):
                        key = target + metric
                        scores = outputs_test[source + metric]  # [participant, adapter, board]
                        valid = torch.isfinite(scores) & (weights > 0)

                        if key not in weighted_sums:
                            shape = (*scores.shape[:2], len(data.embeddings))
                            weighted_sums[key] = scores.new_zeros(shape)
                            weight_sums[key] = scores.new_zeros(shape)

                        weighted_sums[key][:, :, e_i] += torch.where(
                            valid, scores * weights, 0.0
                        ).sum(dim=-1)
                        weight_sums[key][:, :, e_i] += torch.where(
                            valid, weights, 0.0
                        ).sum(dim=-1)

                # Keep the existing descriptive outputs in their original units.
                n_pairs[:, :, e_i, f_i, p_i] = pair_counts.sum(dim=-1)
                n_words[:, :, e_i, f_i, p_i] = word_counts.sum(dim=-1)
                
                # save model and data
                model.save(delta, cfg, name, split.repeat, split.split, p_i)
                model.save_supplement(outputs_train, cfg, name, split.repeat, split.split, p_i, 'train')
                model.save_supplement(outputs_test, cfg, name, split.repeat, split.split, p_i, 'test')
                
                # update tqdm
                pbar.update(1)
    
    # close tqdm
    pbar.close()
    
    # summarise data
    rho_NC = (n_train * rho_NC).sum(dim = (-1, -2)) / n_train.sum(dim = (-1, -2))
    
    rho_BZ = (n_boards * rho_BZ).sum(dim = (-1, -2)) / n_boards.sum(dim = (-1, -2))
    rho_BE = (n_boards * rho_BE).sum(dim = (-1, -2)) / n_boards.sum(dim = (-1, -2))
    rho_ZE = (n_boards * rho_ZE).sum(dim = (-1, -2)) / n_boards.sum(dim = (-1, -2))
    
    # prepare outputs
    outputs = {
        'rho_nc': rho_NC,
        'rho_BZ': rho_BZ,
        'rho_BE': rho_BE,
        'rho_ZE': rho_ZE,
        **{
            key: weighted_sums[key] / weights.masked_fill(weights == 0, torch.nan)
            for key, weights in weight_sums.items()
        },
        'n_train': n_train,
        'n_boards': n_boards,
        'n_pairs': n_pairs,
        'n_words': n_words
    }
    
    # save outputs
    save_supplement(outputs, 'matrain', cfg, logger)

def arrangement_cross(task: DictConfig, data: Dataset, cfg: DictConfig, logger: logger) -> None:
    """Perform an evaluation run across sessions.
    
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
        logger.info('Performing arrangement cross-session eval...')
    
    # load multiple arrangement data
    arrangements = data.load_arrangements(n_session = 1)
    repeated = data.load_arrangements(n_session = 2)
    
    # check subjects
    n_subjects = torch.unique(arrangements.subject_nos).shape[0]
    
    # make data containers
    rho_BZ = torch.zeros((n_subjects, n_subjects, len(data.embeddings), cfg.task.matrain.splits * cfg.task.matrain.repeats, cfg.task.matrain.permutations), dtype = torch.float, device = cfg.device)
    rho_BE = torch.zeros((n_subjects, n_subjects, len(data.embeddings), cfg.task.matrain.splits * cfg.task.matrain.repeats, cfg.task.matrain.permutations), dtype = torch.float, device = cfg.device)
    rho_ZE = torch.zeros((n_subjects, n_subjects, len(data.embeddings), cfg.task.matrain.splits * cfg.task.matrain.repeats, cfg.task.matrain.permutations), dtype = torch.float, device = cfg.device)
    n_boards = torch.zeros((n_subjects, n_subjects, len(data.embeddings), cfg.task.matrain.splits * cfg.task.matrain.repeats, cfg.task.matrain.permutations), dtype = torch.float, device = cfg.device)
    n_pairs = torch.zeros((n_subjects, n_subjects, len(data.embeddings), cfg.task.matrain.splits * cfg.task.matrain.repeats, cfg.task.matrain.permutations), dtype = torch.float, device = cfg.device)
    n_words = torch.zeros((n_subjects, n_subjects, len(data.embeddings), cfg.task.matrain.splits * cfg.task.matrain.repeats, cfg.task.matrain.permutations), dtype = torch.float, device = cfg.device)
    
    weighted_sums = {}
    weight_sums = {}
    
    # setup tqdm
    pbar = tqdm(
        total = cfg.task.matrain.repeats * cfg.task.matrain.splits * len(data.embeddings) * cfg.task.matrain.permutations,
        leave = True,
        disable = (cfg.verbose == False)
    )
    
    # iterate over data splits
    for f_i, split in enumerate(arrangements.split(
        cfg.task.matrain.splits, cfg.task.matrain.repeats,
        seed = cfg.seed
    )):
        # recreate data split
        resplit = repeated.split_like(split)
        
        # preprocess data
        split.preprocess()
        resplit.preprocess()
        
        # compute reliabilities
        split.train.compute_split_half_reliability()
        resplit.train.compute_split_half_reliability()
        
        # iterate over embeddings
        for e_i, name in enumerate(data.embeddings):
            # iterate over permutations
            for p_i in range(cfg.task.matrain.permutations):
                # create model
                delta = model.create(
                    cfg, name, split.repeat, split.split, p_i,
                    split.train.embeddings[name].shape[-1],
                    n_subjects, 
                    preload = True
                )
                
                # evaluate model
                outputs_test = arrangement_eval_single(delta, resplit, name, cfg)
                
                # make summaries for all boards
                rho_BZ[:,:,e_i,f_i,p_i] = outputs_test['rho_BZ'].mean(-1)
                rho_BE[:,:,e_i,f_i,p_i] = outputs_test['rho_BE'].mean(-1)
                rho_ZE[:,:,e_i,f_i,p_i] = outputs_test['rho_ZE'].mean(-1)
                n_boards[:,:,e_i,f_i,p_i] = outputs_test['boards_test'].shape[0]
                
                pair_counts = outputs_test['p_unseen_count']
                word_counts = outputs_test['unseen_count']

                # A board with m unseen words contributes m * (m - 1) / 2 distances.
                word_pair_counts = word_counts * (word_counts - 1) / 2

                for source, target, weights in (
                    ('p_unseen_', 'pns_', pair_counts),
                    ('unseen_', 'uns_', word_pair_counts),
                ):
                    for metric in ('rho_BZ', 'rho_BE', 'rho_ZE'):
                        key = target + metric
                        scores = outputs_test[source + metric]  # [participant, adapter, board]
                        valid = torch.isfinite(scores) & (weights > 0)

                        if key not in weighted_sums:
                            shape = (*scores.shape[:2], len(data.embeddings))
                            weighted_sums[key] = scores.new_zeros(shape)
                            weight_sums[key] = scores.new_zeros(shape)

                        weighted_sums[key][:, :, e_i] += torch.where(
                            valid, scores * weights, 0.0
                        ).sum(dim=-1)
                        weight_sums[key][:, :, e_i] += torch.where(
                            valid, weights, 0.0
                        ).sum(dim=-1)

                # Keep the existing descriptive outputs in their original units.
                n_pairs[:, :, e_i, f_i, p_i] = pair_counts.sum(dim=-1)
                n_words[:, :, e_i, f_i, p_i] = word_counts.sum(dim=-1)
                
                # save model and data
                model.save_supplement(outputs_test, cfg, name, split.repeat, split.split, p_i, 'retest')
                
                # update tqdm
                pbar.update(1)
    
    # close tqdm
    pbar.close()
    
    # summarise data
    rho_BZ = (n_boards * rho_BZ).sum(dim = (-1, -2)) / n_boards.sum(dim = (-1, -2))
    rho_BE = (n_boards * rho_BE).sum(dim = (-1, -2)) / n_boards.sum(dim = (-1, -2))
    rho_ZE = (n_boards * rho_ZE).sum(dim = (-1, -2)) / n_boards.sum(dim = (-1, -2))
    
    # prepare outputs
    outputs = {
        'rho_BZ': rho_BZ,
        'rho_BE': rho_BE,
        'rho_ZE': rho_ZE,
        **{
            key: weighted_sums[key] / weights.masked_fill(weights == 0, torch.nan)
            for key, weights in weight_sums.items()
        }
    }
    
    # save outputs
    save_supplement(outputs, 'macross', cfg, logger)