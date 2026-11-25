"""Dataset filters.

Contains functions and helpers to filter the dataset for specific 
sample characteristics.
"""

from omegaconf import DictConfig, OmegaConf

import pandas as pd
import numpy as np

"""Filter implementations.
"""

def make_filter_range(filter: DictConfig, df: pd.DataFrame) -> np.ndarray:
    """Make a range filter.
    
    Parameters
    ----------
    filter : DictConfig
        Hydra configuration for filter.
    df : pd.DataFrame
        Dataset.
    
    Returns
    -------
    mask : np.ndarray
        Mask for the filter.
    """
    
    # make numpy array
    x = np.array(df[filter.var].tolist()).astype(float)
    
    # make range
    x_min = x.min() if not hasattr(filter, 'min') else float(filter.min)
    x_max = x.max() if not hasattr(filter, 'max') else float(filter.max)

    return (x >= x_min) & (x <= x_max)

def make_filter_incexc(filter: DictConfig, df: pd.DataFrame) -> np.ndarray:
    """Make an inclusion/exclusion filter.
    
    Parameters
    ----------
    filter : DictConfig
        Hydra configuration for filter.
    df : pd.DataFrame
        Dataset.
    
    Returns
    -------
    mask : np.ndarray
        Mask for the filter.
    """
    
    # make numpy array
    x = np.array(df[filter.var].tolist())
    
    # prepare filter
    x_exc = np.ones((x.shape[0],), dtype = bool)
    
    # check inclusions
    if getattr(filter.include, None) is not None:
        x_inc = np.isin(x, np.array(filter.include))
    else:
        x_inc = np.ones((x.shape[0],), dtype = bool)
    
    # check exclusions
    if getattr(filter.exclude, None) is not None:
        x_exc = np.isin(x, np.ndarray(filter.exclude)) == False
    else:
        x_exc = np.ones((x.shape[0],), dtype = bool)
    
    return x_inc & x_exc

def make_filter_notnan(filter: DictConfig, df: pd.DataFrame) -> np.ndarray:
    """Make a filter that excludes NaNs.
    
    Parameters
    ----------
    filter : DictConfig
        Hydra configuration for filter.
    df : pd.DataFrame
        Dataset.
    
    Returns
    -------
    mask : np.ndarray
        Mask for the filter.
    """
    
    # make numpy array
    x = np.array(df[filter.var].tolist())
    
    return np.isnan(x) == False

"""Filter registry and utility.
"""

# register filters
filters = {
    'range': make_filter_range,
    'incexc': make_filter_incexc,
    'notnan': make_filter_notnan
}

def make_filter(filter: DictConfig, df: pd.DataFrame) -> np.ndarray:
    """Make a filter.
    
    Parameters
    ----------
    filter : DictConfig
        Hydra configuration for filter.
    df : pd.DataFrame
        Dataset.
    
    Returns
    -------
    mask : np.ndarray
        Mask for the filter.
    """
    
    # check filter registry
    if filter.type not in filters:
        raise ValueError(
            f'Filter `{filter.type}` is an unknown type.'
        )
    
    return filters[filter.type](
        filter, df
    )