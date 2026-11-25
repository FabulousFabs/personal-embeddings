"""Inference.

Contains various inference methods.
"""

import torch
import numpy as np
import scipy

from dataclasses import dataclass
from .corrections import adjust_clip

from typing import Optional, Any

# setup registry
test_methods = {}

# register t-tests
test_methods['ttest'] = {}

# register t-test variants
test_methods['ttest']['1samp'] = scipy.stats.ttest_1samp
test_methods['ttest']['rel'] = scipy.stats.ttest_rel

# register wilcoxon
test_methods['wilcoxon'] = {}

# register wilcoxon variants
def wilcoxon_1samp(x: np.ndarray, popmean: float = 0.0, alternative: str = 'two-sided') -> Any:
    """Run a Wilcoxon test against an assumed population mean.
    
    Parameters
    ----------
    x : np.ndarray
        Vector of values.
    popmean: float, default=0.0
        Assumed population mean.
    alternative : str, default='two-sided'
        Alternative hypothesis. One of {'two-sided', 'greater', 'lower'}.
    
    Returns
    -------
    test : WilcoxonResult
        Result of the test (see scipy.stats.wilcoxon).
    """
    
    # compute difference
    dx = x - popmean
    
    # return test
    return scipy.stats.wilcoxon(dx, alternative = alternative)

test_methods['wilcoxon']['1samp'] = wilcoxon_1samp

def wilcoxon_rel(x: np.ndarray, y: np.ndarray, alternative: str = 'two-sided') -> Any:
    """Run a Wilcoxon test for two dependent samples.
    
    Parameters
    ----------
    x : np.ndarray
        Vector of values.
    y : np.ndarray
        Vector of baseline values.
    alternative : str, default='two-sided'
        Alternative hypothesis. One of {'two-sided', 'greater', 'lower'}.
    
    Returns
    -------
    test : WilcoxonResult
        Result of the test (see scipy.stats.wilcoxon).
    """
    
    # return test
    return scipy.stats.wilcoxon(x, y, alternative = alternative)

test_methods['wilcoxon']['rel'] = wilcoxon_rel

# register sign flip
test_methods['sign'] = {}

# register sign flip variants
@dataclass
class SignFlipResult:
    """Result from a sign flip permutation test.
    
    Attributes
    ----------
    statistic : float
        T-statistic for the given test; mostly a placeholder.
    pvalue : float
        Uncorrected p-value.
    df : float
        Degrees of freedom for test (as in t-test); mostly a placeholder.
    """
    
    statistic: float
    pvalue: float
    df: float

def sign_flip(dx: np.ndarray, alternative: str = 'two-sided', permutations: int = 100_000) -> SignFlipResult:
    """Run a sign-flip permutation test.
    
    Parameters
    ----------
    dx : np.ndarray
        (Difference) vector to test.
    alternative : str, default='two-sided'
        Alternative hypothesis. One of {'two-sided', 'greater', 'lower'}.
    permutations : int, default=100_000
        Number of permutations to compute.
    
    Returns
    -------
    test : SignFlipResult
        Result of the test (see model.stats.SignFlipResult).
    """
    
    # sample reference
    dr = np.random.choice([1, -1], size = (permutations, dx.shape[0])) * dx[None,:]
    
    # build means
    m_r = dr.mean(1)
    m_o = dx.mean()
    
    # compare to reference
    p_upper = ((m_r >= m_o).sum() + 1) / (permutations + 1)
    p_lower = ((m_r <= m_o).sum() + 1) / (permutations + 1)
    
    # check alternatives
    if alternative == 'greater':
        p_unc = p_upper
    elif alternative == 'lower':
        p_unc = p_lower
    elif alternative == 'two-sided':
        p_unc = np.min(
            adjust_clip(
                2 * np.array([p_lower, p_upper])
            )
        )
    else:
        raise ValueError(
            f'Alternative must be either greater, lower or two-sided, but ' + 
            f'got {alternative} instead.'
        )
    
    # compute df
    df = dx.shape[0] - 1
    
    # compute statistic
    statistic = m_o / (dx.std(ddof = 1) / np.sqrt(dx.shape[0]))
    
    # return result
    return SignFlipResult(
        statistic = statistic,
        pvalue = p_unc,
        df = df
    )
    
def sign_1samp(x: np.ndarray, popmean: float = 0.0, alternative: str = 'two-sided', permutations: int = 100_000) -> SignFlipResult:
    """Run a sign-flip permutation test against an assumed population mean.
    
    Parameters
    ----------
    x : np.ndarray
        Vector of values.
    popmean: float, default=0.0
        Assumed population mean.
    alternative : str, default='two-sided'
        Alternative hypothesis. One of {'two-sided', 'greater', 'lower'}.
    permutations : int, default=100_000
        Number of permutations to compute.
    
    Returns
    -------
    test : SignFlipResult
        Result of the test (see model.stats.SignFlipResult).
    """
    
    # compute difference
    dx = x - popmean
    
    # return test
    return sign_flip(
        dx, 
        alternative = alternative, 
        permutations = permutations
    )

test_methods['sign']['1samp'] = sign_1samp

def sign_rel(x: np.ndarray, y: np.ndarray, alternative: str = 'two-sided', permutations: int = 100_000) -> SignFlipResult:
    """Run a sign-flip permutation test for two dependent samples.
    
    Parameters
    ----------
    x : np.ndarray
        Vector of values.
    y : np.ndarray
        Vector of baseline values.
    alternative : str, default='two-sided'
        Alternative hypothesis. One of {'two-sided', 'greater', 'lower'}.
    permutations : int, default=100_000
        Number of permutations to compute.
    
    Returns
    -------
    test : SignFlipResult
        Result of the test (see model.stats.SignFlipResult).
    """
    
    # compute difference
    dxdy = x - y
    
    return sign_flip(
        dxdy,
        alternative = alternative,
        permutations = permutations
    )

test_methods['sign']['rel'] = sign_rel

# register permutation
test_methods['permutation'] = {}

# register permutation variants
@dataclass
class PermutationResult:
    """Result from a permutation test.
    
    Attributes
    ----------
    statistic : float
        T-statistic for the given test; mostly a placeholder.
    pvalue : float
        Uncorrected p-value.
    df : float
        Degrees of freedom for test (as in t-test); mostly a placeholder.
    """
    
    statistic: float
    pvalue: float
    df: float

def permutation(dx: np.ndarray, alternative: str = 'two-sided') -> PermutationResult:
    """Run a precomputed permutation test.
    
    Parameters
    ----------
    dx : np.ndarray
        (Difference) vector to test.
    alternative : str, default='two-sided'
        Alternative hypothesis. One of {'two-sided', 'greater', 'lower'}.
    
    Returns
    -------
    test : PermutationResult
        Result of the test (see model.stats.PermutationResult).
    """
    
    # check permutation count
    permutations = dx.shape[0]
    
    # compare to reference
    p_upper = ((dx <= 0.0).sum() + 1) / (permutations + 1)
    p_lower = ((dx >= 0.0).sum() + 1) / (permutations + 1)
    
    # check alternatives
    if alternative == 'greater':
        p_unc = p_upper
    elif alternative == 'lower':
        p_unc = p_lower
    elif alternative == 'two-sided':
        p_unc = np.min(
            adjust_clip(
                2 * np.array([p_lower, p_upper])
            )
        )
    else:
        raise ValueError(
            f'Alternative must be either greater, lower or two-sided, but ' + 
            f'got {alternative} instead.'
        )
    
    # compute df
    df = dx.shape[0] - 1
    
    # compute statistic
    statistic = dx.mean() / (dx.std(ddof = 1) / np.sqrt(dx.shape[0]))
    
    # return result
    return PermutationResult(
        statistic = statistic,
        pvalue = p_unc,
        df = df
    )

def permutation_1samp(x: np.ndarray, popmean: float = 0.0, alternative: str = 'two-sided') -> PermutationResult:
    """Run a precomputed permutation test against an assumed population mean.
    
    Parameters
    ----------
    x : np.ndarray
        Vector of values.
    popmean: float, default=0.0
        Assumed population mean.
    alternative : str, default='two-sided'
        Alternative hypothesis. One of {'two-sided', 'greater', 'lower'}.
    
    Returns
    -------
    test : PermutationResult
        Result of the test (see model.stats.PermutationResult).
    """
    
    # compute difference
    dx = x - popmean
    
    # return test
    return permutation(
        dx, 
        alternative = alternative,
    )

test_methods['permutation']['1samp'] = permutation_1samp

def permutation_rel(x: np.ndarray, y: np.ndarray, alternative: str = 'two-sided') -> PermutationResult:
    """Run a precomputed permutation test for two dependent samples.
    
    Parameters
    ----------
    x : np.ndarray
        Vector of values.
    y : np.ndarray
        Vector of baseline values.
    alternative : str, default='two-sided'
        Alternative hypothesis. One of {'two-sided', 'greater', 'lower'}.
    
    Returns
    -------
    test : PermutationResult
        Result of the test (see model.stats.SignFlipResult).
    """
    
    # compute difference
    dxdy = x - y
    
    return permutation(
        dxdy,
        alternative = alternative,
    )

test_methods['permutation']['rel'] = permutation_rel

def hypothesis_test(x: np.ndarray, y: Optional[np.ndarray] = None, test: str = 'ttest', variant: str = 'rel', popmean: float = 0.0, alternative: str = 'two-sided', permutations: int = 100_000) -> Any:
    """Run a hypothesis test.
    
    Parameters
    ----------
    x : np.ndarray
        Vector of values.
    y : np.ndarray | None, default=None
        Optional vector of baseline values.
    test : str, default='ttest'
        Test family to use. One of {'ttest', 'wilcoxon', 'sign', 'permutation'}.
    variant : str, default='rel'
        Test variant to use. One of {'rel', '1samp'}.
    popmean : float, default=0.0
        If variant is one-sample test, which population mean to assume.
    alternative : str, default='two-sided'
        Alternative hypothesis. One of {'two-sided', 'greater', 'lower'}.
    permutations : int, default=100_000
        If test family is sign-flip permutation test, number of permutations to compute.
    
    Returns
    -------
    test : Any
        Result of the test (see scipy.stats.ttest_rel, scipy.stats.ttest_1samp, scipy.stats.wilcoxon, model.stats.sign_flip).
    """
    
    # check test
    if test not in test_methods:
        raise ValueError(
            f'{test} is not a known test. Available options are ' + 
            f'{list(test_methods.keys())}.'
        )
    
    # check variant
    if variant not in test_methods[test]:
        raise ValueError(
            f'{test} does not have variant {variant}. Available ' + 
            f'options are {list(test_methods[test].keys())}.'
        )
    
    # make args
    args = (x,) if y is None else (x, y)

    # make options
    opts = dict(alternative = alternative)
    
    if test in ['sign']:
        opts['permutations'] = permutations
    
    if variant in ['1samp']:
        opts['popmean'] = popmean
    
    # return test
    return test_methods[test][variant](
        *args,
        **opts
    )