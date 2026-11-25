"""Reporting.

Contains various reporting methods.
"""

import torch
import numpy as np
import pandas as pd
import scipy

from pathlib import Path
from copy import deepcopy
from .bootstrap import prepare_bootstrap_se_within, bootstrap_se, bootstrap_se_ratio
from .corrections import adjust_clip, adjust_p
from .inference import hypothesis_test
from .effects import one_sample_d, dependent_samples_d

from typing import Optional, Any, List, Dict, Tuple

def sample_tests(X: np.ndarray | torch.Tensor, popmean = 0.0, test: str = 'ttest', alternative: str = 'two-sided', method: str = 'holm', se: str = 'between') -> List[Dict[str, float]]:
    """Run sample tests against null en-masse.
    
    Parameters
    ----------
    X : np.ndarray | torch.Tensor
        Matrix of values (n_groups, n_samples).
    popmean: float, default=0.0
        Assumed population mean.
    test : str = 'ttest'
        Test family to use. One of {'ttest', 'wilcoxon', 'sign'}.
    alternative : str, default='two-sided'
        Alternative hypothesis. One of {'two-sided', 'greater', 'lower'}.
    method : str, default='holm'
        Method to use when correcting for multiple comparisons (see model.stats.adjust_p).
    se : str, default='between'
        Method to use for bootstrapping standard errors. One of {'between', 'within'}.
    
    Returns
    -------
    tests : List[Dict[str, float]]
        List of all tests and associated descriptives. Each item is a dictionary of:
        ```
            index : int
                Position of this group in original matrix.
            mu : float
                Group mean.
            se : float
                Estimated standard error of group mean (bootstrapped).
            se_method : str
                Method used for bootstrapping standard errors.
            test : str
                Test family and variant used.
            df : float
                Degrees of freedom.
            statistic : float
                Test statistic.
            d : float
                Effect size measure Cohen's d.
            p_cor : float
                Corrected p-value.
            p_unc : float
                Uncorrected p-value.
            p_method : str
                Method used for correcting for multiple comparisons.
        ```
    """
    
    # check data
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    
    n_dims = X.shape[0]
    
    # determine bootstrap
    if se not in ['between', 'within']:
        raise ValueError(
            f'Unknown bootstrap method {se}. ' + 
            f'Available options: {["between", "within"]}.'
        )
    
    # if required, prepare within SE
    if se == 'within':
        Z = prepare_bootstrap_se_within(X)
    else:
        Z = X
    
    # compute group descriptors
    groups = [
        dict(
            index = i,
            mu = X[i].mean(0).item(), 
            se = bootstrap_se(Z[i]).item(),
            se_method = se
        )
        for i in range(n_dims)
    ]
    
    # compute group statistics
    tests = [
        hypothesis_test(
            X[i], 
            test = test,
            variant = '1samp',
            popmean = popmean, 
            alternative = alternative
        )
        for i in range(n_dims)
    ]
    
    # collect and correct group
    p = adjust_p(
        np.array([
            test.pvalue
            for test in tests
        ]),
        method = method
    )
    
    # add to group data
    for i in range(n_dims):
        groups[i]['test'] = f'{test}_1samp'
        groups[i]['df'] = X.shape[1] - 1
        groups[i]['statistic'] = tests[i].statistic.item()
        groups[i]['d'] = one_sample_d(X[i], popmean = popmean).item()
        groups[i]['p_cor'] = p[i].item()
        groups[i]['p_unc'] = tests[i].pvalue.item()
        groups[i]['p_method'] = method
    
    return groups

def pairwise_tests(X: np.ndarray | torch.Tensor, tests: List[Tuple[int, int]], test: str = 'ttest', alternative: str = 'two-sided', method: str = 'holm', se: str = 'between') -> List[Dict[str, float]]:
    """Run pairwise tests en-masse.
    
    Parameters
    ----------
    X : np.ndarray | torch.Tensor
        Matrix of values (n_groups, n_samples).
    tests : List[Tuple[int, int]]
        Design specification where each item contains a tuple (x, y) specifying a contrast.
    test : str = 'ttest'
        Test family to use. One of {'ttest', 'wilcoxon', 'sign'}.
    alternative : str, default='two-sided'
        Alternative hypothesis. One of {'two-sided', 'greater', 'lower'}.
    method : str, default='holm'
        Method to use when correcting for multiple comparisons (see model.stats.adjust_p).
    se : str, default='between'
        Method to use for bootstrapping standard errors. One of {'between', 'within'}.
    
    Returns
    -------
    tests : List[Dict[str, float]]
        List of all tests and associated descriptives. Each item is a dictionary of:
        ```
            index : tuple[int, int]
                Position of groups in this contrast in original matrix.
            mu : float
                Mean difference.
            se : float
                Estimated standard error of mean difference (bootstrapped).
            se_method : str
                Method used for bootstrapping standard errors.
            pct_mu : float
                Percentage mean difference.
            pct_se : float
                Estimated standard error of percentage mean difference (bootstrapped).
            test : str
                Test family and variant used.
            df : float
                Degrees of freedom.
            statistic : float
                Test statistic.
            d : float
                Effect size measure Cohen's d.
            p_cor : float
                Corrected p-value.
            p_unc : float
                Uncorrected p-value.
            p_method : str
                Method used for correcting for multiple comparisons.
        ```
    """
    
    # check data
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    
    n_dims = X.shape[0]
    
    # check tests
    for x, y in tests:
        if (x >= n_dims) or (y >= n_dims):
            raise ValueError(
                f'Trying to test ({x}, {y}) but ' +
                f'X has shape ({n_dims}, n).'
            )
    
    # determine bootstrap
    if se not in ['between', 'within']:
        raise ValueError(
            f'Unknown bootstrap method {se}. ' + 
            f'Available options: {["between", "within"]}.'
        )
    
    # if required, prepare within SE
    if se == 'within':
        Z = prepare_bootstrap_se_within(X)
    else:
        Z = X
    
    # comput tests
    ttests = [
        hypothesis_test(
            X[i], X[j], 
            test = test,
            variant = 'rel',
            alternative = alternative
        )
        for (i, j) in tests
    ]
    
    # collect and correct tests
    p = adjust_p(
        np.array([
            test.pvalue
            for test in ttests
        ]),
        method = method
    )
    
    # collect comparisons
    comp = [
        dict(
            index = (ii, jj),
            mu = (X[ii] - X[jj]).mean().item(),
            se = bootstrap_se(Z[ii] - Z[jj]).item(),
            se_method = se,
            pct_mu = (100 * (X[ii].mean() / X[jj].mean() - 1)).item(),
            pct_se = bootstrap_se_ratio(Z[ii], Z[jj]).item(),
            test = f'{test}_rel',
            df = X.shape[1] - 1,
            statistic = ttests[i].statistic.item(),
            d = dependent_samples_d(X[ii], X[jj]).item(),
            p_cor = p[i].item(),
            p_unc = ttests[i].pvalue.item(),
            p_method = method
        )
        for i, (ii, jj) in enumerate(tests)
    ]
    
    return comp

"""Descriptive helpers.
"""

def sample_descriptives(X: pd.DataFrame | np.ndarray | torch.Tensor, keys: Optional[List[str]] = None, se: str = 'between') -> List[Dict[str, Any]]:
    """
    """
    
    # check dataframe
    if isinstance(X, pd.DataFrame):
        # if required, select keys
        if keys is not None:
            X = X[keys]
        
        keys = np.array(list(X.keys()))
        X = X.to_numpy()
    
    # check torch
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    
    # check aberrant type
    if not isinstance(X, np.ndarray):
        raise ValueError(
            f'X must be of type pd.DataFrame, np.ndarray or torch.Tensor, '
            f'but got unknown type {type(X)} instead.'
        )
    
    # check named keys
    if keys is None:
        keys = np.arange(X.shape[1])
    
    # setup fields
    fields = [
        'key', 
        'mean', 'median', 'se', 'sd', 'lb', 'ub', 'min', 'max',
        'value', 'count', 'proportion'
    ]

    # compute descriptives
    desc = []
    
    for i, key in enumerate(keys):
        # check array-like
        if hasattr(key, 'item'):
            key = key.item()
        
        # setup dummy
        desc_i = {field: None for field in fields}
        
        # check numeric
        try:
            X_i = X[:,i].astype(float)
            is_numeric = True
        except ValueError:
            X_i = X[:,i].astype(str)
            is_numeric = False
        
        # enter key
        desc_i['key'] = key
        
        # handle numeric/string
        if is_numeric:
            # enter average descriptives
            desc_i['mean'] = X_i.mean().item()
            desc_i['median'] = np.median(X_i).item()
            
            # enter CI descriptives
            desc_i['se'] = bootstrap_se(X_i).item()
            desc_i['sd'] = X_i.std().item()
            desc_i['lb'] = desc_i['mean'] - 1.96 * desc_i['se']
            desc_i['ub'] = desc_i['mean'] + 1.96 * desc_i['se']
            
            # enter range descriptives
            desc_i['min'] = X_i.min().item()
            desc_i['max'] = X_i.max().item()
            
            # append
            desc.append(desc_i)
        else:
            # find unique values
            unique = np.unique(X_i)
            
            # make temporary array
            tmp = []
            
            # iterate values
            for j, value in enumerate(unique):
                # copy data
                desc_j = deepcopy(desc_i)
                
                # identify 
                desc_j['value'] = value.item()
                
                # enter count descriptives
                indc = np.where(X_i == value)[0]
                desc_j['count'] = indc.shape[0]
                desc_j['proportion'] = indc.shape[0] / X_i.shape[0]
                
                # append
                desc.append(desc_j)
    
    return desc

"""Output helpers.
"""

def save_results(results: List[Dict[str, float]], path: str | Path) -> None:
    """Save results as comma-separated values.
    
    Parameters
    ----------
    results : List[Dict[str, float]]
        Test results structure (see model.stats.sample_tests, model.stats.pairwise_tests).
    path : str | Path
        Path for output file.
    """
    
    # make path
    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok = True)
    
    # convert to dictionary
    df = {
        key: [
            result[key]
            for result in results
        ]
        for key in results[0]
    }
    
    # convert to dataframe
    df = pd.DataFrame.from_dict(df)
    
    # save data
    df.to_csv(
        path.with_suffix('.csv'), 
        index = False
    )