"""Embedding I/O.

Contains functions and helpers to read and write embeddings.
"""

from omegaconf import DictConfig, OmegaConf

import torch
import pandas as pd
import numpy as np
import csv

from typing import Dict

def read_embedding(file: str) -> Dict:
    """Read a TSV embedding.
    
    Parameters
    ----------
    file : str
        The file to read.
    
    Returns
    -------
    Dict[str, torch.Tensor]
        Embeddings.
    """
    
    # read file
    df = pd.read_csv(
        file,
        sep = '\t', quoting = csv.QUOTE_NONE
    )
    
    # make embedding
    embedding = {}
    
    for i in range(len(df)):
        # read entry
        entry = df.loc[i]
                
        # find word and vector
        voc = entry.word.lower()
        vec = torch.tensor(
            [
                float(entry[f'dim_{j}']) for j in range(len(df.columns) - 1)
            ]
        ).float()
        
        # register
        embedding[voc] = vec
    
    return embedding
    
def write_embedding(file: str, embedding: Dict) -> None:
    """Write a TSV embedding.
    
    Parameters
    ----------
    file : str
        The file to write.
    embedding : Dict[str, torch.Tensor]
        Embeddings.
    """
    
    # prepare outputs
    out = {'word': []}
    
    n_dim = embedding[
        list(embedding.keys())[0]
    ].shape[0]
    
    for i in range(n_dim):
        out[f'dim_{i}'] = []
    
    # populate outputs
    for word in embedding:
        out['word'].append(word.lower())
        for i in range(n_dim):
            out[f'dim_{i}'].append(embedding[word][i])
    
    # write embedding
    pd.DataFrame.from_dict(out).to_csv(
        file,
        sep = '\t', index = False, header = True, quoting = csv.QUOTE_NONE
    )