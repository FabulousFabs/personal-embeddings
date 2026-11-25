"""Convert raw data from Prolific and Pavlovia to a BIDS-compliant dataset.

Compliance was tested using the BIDS validator: https://bids-standard.github.io/bids-validator/
"""

# set flag for extended data coverage?
# only available for exp2
_FLAG_EXTENDED = True

import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger
from pathlib import Path

import os, glob

import datetime, time
import pandas as pd
import numpy as np
import json, csv

from typing import Tuple, Dict

def make_dataframe_ma(df: pd.DataFrame, pid: str, sid: str) -> Tuple[pd.DataFrame, Dict]:
    """Make a BIDS compliant dataframe and sidecar for data from the multiple arrangement task.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe.
    pid : str
        Participant identifier from prolific.
    sid : str
        Numerical subject identifier.
    
    Returns
    -------
    df_new : pd.DataFrame
        The new dataframe.
    sidecar : Dict
        The sidecar describing the dataframe.
    """
    
    # grab trials
    data = df.loc[
        (df.trial_type == 'item_mapping')
    ].reset_index(drop = True)
    
    # make boards
    boards = data.boards.tolist()
    identities = data.identities.tolist()
    reactions = data.rt.tolist()
    
    # prepare data
    out = {
        'pid': [],
        'sid': [],
        'stimulus_id': [],
        'trial_id': [],
        'trial_type': [],
        'board_id': [],
        'board_no': [],
        'word': [],
        'x': [],
        'y': [],
        'rt': []
    }
    
    # extended data preparation
    if _FLAG_EXTENDED:
        out['ix'] = []
        out['iy'] = []
    
    # sidecar preparation
    sidecar = {
        'TaskName': 'Multiple Arrangement',
        'Instructions': 'In this experiment, you will see circles surrounded by words. Please place the words inside the circle so that the distances between the words reflect how you understand the meanings of the words and their relationships to each other. ' + 
                        'To do this, please click on a word and drag and drop it to the desired position in the circle. You can move each word as often as you think is necessary. ' +
                        'Once you have arranged all the words in a circle, please press `Done`. ' + 
                        'There are no right or wrong answers. We are interested in your individual opinion.',
        'TaskDescription': 'In each trial, participants are visually presented with twenty words surrounding a circle. Their task is arranging the words in the circle such that distances between words reflect the (dis-)similarity of words as per their semantic judgement.',
        'CogAtlasID': '',
        'CogPOID': '',
        'InstitutionName': 'University Medical Centre Hamburg-Eppendorf',
        'InstitutionAddress': 'Martinistr. 52, 20246 Hamburg, Germany',
        'InstitutionalDepartmentName': 'Institute for Systems Neuroscience',
        'pid': {
            'LongName': 'Participant identifier',
            'Description': 'Pavlovia participant identifier.'
        },
        'sid': {
            'LongName': 'Subject identifier',
            'Description': 'Numeric subject identifier.'
        },
        'stimulus_id': {
            'LongName': 'Stimulus identifier',
            'Description': 'Identifier of the stimulus across participants.'
        },
        'trial_id': {
            'LongName': 'Trial identifier',
            'Description': 'Identification of trial of type item_mapping in task.'
        },
        'trial_type': {
            'LongName': 'Trial type',
            'Description': 'Type of trial in the task (only `ma`).'
        },
        'board_id': {
            'LongName': 'Board identifier',
            'Description': 'Identifier that allows mapping of boards between participants.'
        },
        'board_no': {
            'LongName': 'Board number',
            'Description': 'Identifier of the board in this subject.'
        },
        'word': {
            'LongName': 'Word',
            'Description': 'The word that was presented.'
        },
        'x': {
            'LongName': 'X-coordinate',
            'Description': 'Coordinate along the x-axis.'
        },
        'y': {
            'LongName': 'Y-coordinate',
            'Description': 'Coordinate along the y-axis.'
        },
        'rt': {
            'LongName': 'Reaction time',
            'Description': 'Reaction time for this circle.'
        }
    }
    
    # extended sidecar preparation
    if _FLAG_EXTENDED:
        sidecar['ix'] = {
            'LongName': 'Initial X-coordinate',
            'Description': 'Initial random coordinate along the x-axis at spawn.'
        }
        sidecar['iy'] = {
            'LongName': 'Initial Y-coordinate',
            'Description': 'Initial random coordinate along the y-axis at spawn.'
        }
    
    # loop over trials
    for i, (board, identity, rt_all) in enumerate(zip(boards, identities, reactions)):
        # unpack json
        X = json.loads(board)
        
        # unpack identity
        identity = np.array(identity.split(',')).astype(int)
        
        # check reaction times
        rt = [rt_all] * len(identity) if 'rt' not in X else X['rt']
        
        # check data
        for j in range(len(X['final'])):
            for k, v_j in enumerate(list(X['final'][j].keys())):
                # stack data
                out['pid'].append(pid)
                out['sid'].append(sid)
                out['stimulus_id'].append(identity[j])
                out['trial_id'].append(i)
                out['trial_type'].append('ma')
                out['board_id'].append(identity[j])
                out['board_no'].append(j)
                out['word'].append(v_j)
                if _FLAG_EXTENDED:
                    out['ix'].append(X['initial'][j][v_j][0])
                    out['iy'].append(X['initial'][j][v_j][1])
                out['x'].append(X['final'][j][v_j][0])
                out['y'].append(X['final'][j][v_j][1])
                out['rt'].append(rt[j] / 1000)
    
    # make dataframe
    df_new = pd.DataFrame.from_dict(out)

    return df_new, sidecar

def make_dataframe_triplet(df: pd.DataFrame, pid: str, sid: str) -> Tuple[pd.DataFrame, Dict]:
    """Make a BIDS compliant dataframe and sidecar for data from the triplet task.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe.
    pid : str
        Participant identifier from prolific.
    sid : str
        Numerical subject identifier.
    
    Returns
    -------
    df_new : pd.DataFrame
        The new dataframe.
    sidecar : Dict
        The sidecar describing the dataframe.
    """
    
    # grab trials
    data = df.loc[
        (df.trial_type == 'odd_one_out')
    ].reset_index(drop = True)
    
    # make unique stimulus ids
    stim_order = np.unique(data.options.tolist())
    stimulus_ids = [
        np.where(stim_order == stimulus)[0][0]
        for stimulus in data.options.tolist()
    ]
    
    # write data
    out = {
        'pid': [pid] * len(data),
        'sid': [sid] * len(data),
        'stimulus_id': stimulus_ids,
        'trial_id': np.array(data.no.tolist()).astype(int).tolist(),
        'trial_type': ['triplet'] * len(data),
        'block_id': np.array(data.block.tolist()).astype(int).tolist(),
        'option_0': data.options_0.tolist(),
        'option_1': data.options_1.tolist(),
        'option_2': data.options_2.tolist(),
        'choice_no': np.array(data.choice_idx.tolist()).astype(int).tolist(),
        'choice_word': np.array(
            [
                data.options_0.tolist(), data.options_1.tolist(), data.options_2.tolist()
            ]
        )[
            np.array(data.choice_idx.tolist()).astype(int),
            np.arange(len(data))
        ].tolist(),
        'rt': (np.array(data.rt.tolist()) / 1000).tolist()
    }
    
    sidecar = {
        'TaskName': 'Triplet',
        'Instructions': 'In each trial of this experiment, you will be presented with three words. Your task is to choose the word whose meaning does not fit with the other two. ' +
                        'To do so, please press the corresponding number key on your keyboard. For example, if you see (1) <i>dog</i>, (2) <i>cat</i>, (3) <i>graffiti</i> and you think <i>graffiti</i> does not fit, press the `3` key. ' + 
                        'There are no right or wrong answers. We are interested in your individual opinion. ' +
                        '<b>Attention:</b> You have only five seconds for each decision. Please respond as quickly and accurately as possible.',
        'TaskDescription': 'In each trial, participants are visually presented with three words. Their task is to select the one (by key press) that does not fit the other two, i.e. an odd-one-out task.',
        'CogAtlasID': '',
        'CogPOID': '',
        'InstitutionName': 'University Medical Centre Hamburg-Eppendorf',
        'InstitutionAddress': 'Martinistr. 52, 20246 Hamburg, Germany',
        'InstitutionalDepartmentName': 'Institute for Systems Neuroscience',
        'pid': {
            'LongName': 'Participant identifier',
            'Description': 'Pavlovia participant identifier.'
        },
        'sid': {
            'LongName': 'Subject identifier',
            'Description': 'Numeric subject identifier.'
        },
        'stimulus_id': {
            'LongName': 'Stimulus identifier',
            'Description': 'Identification of the stimulus across participants.'
        },
        'trial_id': {
            'LongName': 'Trial identifier',
            'Description': 'Identification of trial of type odd_one_out in task.'
        },
        'trial_type': {
            'LongName': 'Trial type',
            'Description': 'Type of trial in the task (only `triplet`).'
        },
        'block_id': {
            'LongName': 'Block identifier',
            'Description': 'Identification of block in odd_one_out task.'
        },
        'option_0': {
            'LongName': 'Option 0',
            'Description': 'Option 0 in stimulus presentation (Option 0 | Option 1 | Option 2).'
        },
        'option_1': {
            'LongName': 'Option 1',
            'Description': 'Option 1 in stimulus presentation (Option 0 | Option 1 | Option 2).'
        },
        'option_2': {
            'LongName': 'Option 2',
            'Description': 'Option 2 in stimulus presentation (Option 0 | Option 1 | Option 2).'
        },
        'choice_no': {
            'LongName': 'Numerical choice',
            'Description': 'Chosen option from stimulus presentation (Option 0 | Option 1 | Option 2).'
        },
        'choice_word': {
            'LongName': 'Word choice',
            'Description': 'Chosen option from stimulus presentation (Option 0 | Option 1 | Option 2).'
        },
        'rt': {
            'LongName': 'Reaction time',
            'Description': 'Reaction time for this trial.'
        }
    }
    
    df_new = pd.DataFrame.from_dict(out)
    
    return df_new, sidecar

@hydra.main(config_path = 'conf', config_name = 'config', version_base = None)
def make_bids(cfg: DictConfig) -> None:
    """Convert source data to BIDS dataset.
    
    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration.
    """
    
    # check source data directory
    if cfg.verbose:
        logger.info('Checking paths...')
    
    dir_scd = Path(cfg.data.path) / Path('sourcedata')
    
    if not dir_scd.exists():
        raise ValueError(
            f'Specified path `{dir_scd}` does not exist.'
        )
    
    # check derivatives path
    dir_dev = Path(cfg.data.path) / Path('derivatives')
    dir_dev.mkdir(parents = True, exist_ok = True)

    # check pavlovia and prolific directories
    dir_pav = dir_scd / Path('pavlovia')
    dir_pro = dir_scd / Path('prolific')
    
    if not dir_pav.exists():
        raise ValueError(
            f'Specified path `{dir_scd}` does not contain ' + 
            f'`pavlovia` folder.'
        )
    
    if not dir_pro.exists():
        raise ValueError(
            f'Specified path `{dir_scd}` does not contain ' + 
            f'`prolific` folder.'
        )
    
    # grab participant exports
    if cfg.verbose:
        logger.info(f'Loading participant exports...')
    
    f_pro = [
        name
        for name in glob.glob(
            str(dir_pro / Path('ses-1/*.csv'))
        )
    ][0]
    
    df_pro = pd.read_csv(f_pro)
    
    # make unique participant identifiers
    df_pro = df_pro.loc[
        (df_pro.Status == 'APPROVED')
    ].reset_index(drop = True)
    
    pids = np.unique(df_pro['Participant id'].tolist())
    sids = np.array(
        [
            f'{i:03d}'
            for i in range(1, len(pids) + 1, 1)
        ]
    )
    
    pid_to_sid = {
        pid: sid
        for pid, sid in zip(pids, sids)
    }
    
    sid_to_pid = {
        sid: pid
        for pid, sid in zip(pids, sids)
    }
    
    # loop over sessions
    if cfg.verbose:
        logger.info(f'Loading participant sessions...')
    
    # setup output path
    dir_out = Path(cfg.data.path)
    
    # setup session tracking
    ses_has_pid = {}
    ses_has_stm = {}
    ses_has_dur = {}
    ses_has_int = {}
    ses_has_tme = {}
    
    # setup source and output tracking
    all_sources = []
    all_outputs = []
    
    for subdir, dirs, _ in os.walk(str(dir_pav)):
        # skip top directory
        if len(dirs) > 0:
            continue
        
        # find session number
        n_ses = int(subdir.split('/')[-1].split('-')[-1])
        
        # make session tracker
        has_pid = {
            pid: False
            for pid in pids
        }
        
        # make session timestamps
        ses_stm = {
            pid: None
            for pid in pids
        }
        
        # make session durations
        ses_dur = {
            pid: None
            for pid in pids
        }
        
        # make session interest
        ses_int = {
            pid: None
            for pid in pids
        }
        
        # make timezone
        ses_tme = {
            pid: None
            for pid in pids
        }
        
        if cfg.verbose:
            logger.info(f'Preparing data from session {n_ses}...')
        
        # find files
        for f_in in glob.glob(str(Path(subdir) / Path('*.csv'))):
            # track sources
            all_sources.append(f_in)
            
            # load data file
            df_i = pd.read_csv(f_in)
            
            # determine participant
            pid = df_i.prolific_pid.tolist()[0]
            sid = pid_to_sid[pid]
            
            # update session tracker
            has_pid[pid] = True
            
            # update timestamp tracker
            tstr = '.'.join(' '.join(f_in.split('_')[-2:]).split('.')[:-1])
            tstr = tstr.replace('h', ':').split('.')[0]
            date = datetime.datetime.strptime(tstr, '%Y-%m-%d %H:%M')
            ses_stm[pid] = time.mktime(date.timetuple())
            
            # update duration tracker
            ses_dur[pid] = int(df_i.time_elapsed.tolist()[-1])
            
            # update interest tracker, if present
            if (df_i.trial_type == 'qa_slider').sum() > 0:
                # cut to interest data
                df_int = df_i.loc[
                    (df_i.trial_type == 'qa_slider')
                ].reset_index(drop = True)
                
                # add interest data
                ses_int[pid] = [
                    float(df_int[f'value_{int_i}'].tolist()[0])
                    for int_i in range(6)
                ]
            
            # update timezone tracker, if present
            if 'time_zone' in list(df_i.keys()):
                ses_tme[pid] = df_i['time_utc_offset'].tolist()[0]
            
            # determine task
            is_ma = (df_i.trial_type == 'item_mapping').sum() > 0
            
            if is_ma:
                # multiple arrangement data
                df_j, sc_j = make_dataframe_ma(df_i, pid, sid)
                task_j = 'ma'
            else:
                # triplet data
                df_j, sc_j = make_dataframe_triplet(df_i, pid, sid)
                task_j = 'triplet'
            
            # add source to sidecar
            sc_j['RawSources'] = [str(f_in)]
            
            # check folder
            d_out = dir_out / Path(f'sub-{sid}') / Path(f'ses-{n_ses}') / Path('beh')
            d_out.mkdir(parents = True, exist_ok = True)
            
            # write data
            df_out = d_out / Path(f'sub-{sid}_ses-{n_ses}_task-{task_j}_beh.tsv')
            df_j.to_csv(
                df_out, sep = '\t', index = False, header = True, quoting = csv.QUOTE_NONE
            )
            
            sc_out = d_out / Path(f'sub-{sid}_ses-{n_ses}_task-{task_j}_beh.json')
            with open(sc_out, 'w') as f:
                json.dump(sc_j, f, indent = 4, sort_keys = False)
            
            # track outputs
            all_outputs.append(df_out)

        # register session
        ses_has_pid[n_ses] = has_pid
        ses_has_stm[n_ses] = ses_stm
        ses_has_dur[n_ses] = ses_dur
        ses_has_int[n_ses] = ses_int
        ses_has_tme[n_ses] = ses_tme
        
        # make stimulus files
        d_out = dir_out / Path(f'stimuli')
        d_out.mkdir(parents = True, exist_ok = True)
        
        if is_ma:
            # multiple arrangement task
            unq_boards = np.unique(df_j.board_id.tolist())
            
            # grab counts
            n_boards = unq_boards.shape[0]
            n_per_board = len(df_j.loc[
                (df_j.board_id == 0)
            ])
            
            # prepare outputs
            out = {'stimulus_id': [], 'type': []}
            sdc = {
                'stimulus_id': {
                    'LongName': 'Board identifier',
                    'Description': 'Unique identifier of a board.'
                },
                'type': {
                    'LongName': 'Stimulus type',
                    'Description': 'Type of stimulus.'
                }
            }
            
            for i in range(n_per_board):
                out[f'word_{i}'] = []
                sdc[f'word_{i}'] = {
                    'LongName': f'Word {i}',
                    'Description': f'Word number {i} presented on this board.'
                }
            
            # loop over boards
            for idx in unq_boards:
                # select board
                board = df_j.loc[
                    (df_j.board_id == idx)
                ].reset_index(drop = True)
                
                # select words
                words = np.unique(board.word.tolist()).tolist()
                
                # add words
                for i in range(n_per_board):
                    out[f'word_{i}'].append(words[i])

                # add board
                out['stimulus_id'].append(idx)
                out['type'].append('text')
        else:
            # triplet task
            out = {
                'stimulus_id': df_j.stimulus_id.tolist(),
                'type': ['text'] * len(df_j),
                'word_0': df_j.option_0.tolist(),
                'word_1': df_j.option_1.tolist(),
                'word_2': df_j.option_2.tolist()
            }
            
            sdc = {
                'stimulus_id': {
                    'LongName': 'Triplet identifier',
                    'Description': 'Unique identifier of a triplet.'
                },
                'type': {
                    'LongName': 'Stimulus type',
                    'Description': 'Type of stimulus.'
                },
                'word_0': {
                    'LongName': 'Word 0',
                    'Description': 'Word 0 in triplet (Word 0 | Word 1 | Word 2).'
                },
                'word_1': {
                    'LongName': 'Word 1',
                    'Description': 'Word 1 in triplet (Word 0 | Word 1 | Word 2).'
                },
                'word_2': {
                    'LongName': 'Word 2',
                    'Description': 'Word 2 in triplet (Word 0 | Word 1 | Word 2).'
                }
            }
        
        # save stimulus data
        df_out = d_out / Path(f'stimuli_task-{task_j}.tsv')
        pd.DataFrame.from_dict(out).sort_values(by = ['stimulus_id']).to_csv(
            df_out, sep = '\t', index = False, header = True, quoting = csv.QUOTE_NONE
        )
        
        # write participant sidecar
        sc_out = d_out / Path(f'stimuli_task-{task_j}.json')
        with open(sc_out, 'w') as f:
            json.dump(sdc, f, indent = 4, sort_keys = False)

    # add participant descriptors
    if cfg.verbose:
        logger.info(f'Preparing participant descriptors...')

    out = {
        name: []
        for name in ['participant_id', 'sid', 'pid'] + \
                    [f'ses_{n_ses}' for n_ses in np.unique(list(ses_has_pid.keys()))] + \
                    [f'timestamp_{n_ses}' for n_ses in np.unique(list(ses_has_stm.keys()))] + \
                    [f'duration_{n_ses}' for n_ses in np.unique(list(ses_has_dur.keys()))] + \
                    list(cfg.data.participant_descriptors.keys())
    }
    
    if _FLAG_EXTENDED:
        for n_ses in np.unique(list(ses_has_pid.keys())):
            out[f'timezone_{n_ses}'] = []
            for i in range(6):
                out[f'interest_{n_ses}_{i}'] = []
    
    sdc = {
        'participant_id': {
            'LongName': 'Participant identifier',
            'Description': 'BIDS style participant identifier.'
        },
        'sid': {
            'LongName': 'Subject identifier',
            'Description': 'Numeric subject identifier.'
        },
        'pid': {
            'LongName': 'Participant identifier',
            'Description': 'Pavlovia participant identifier.'
        }
    }
    
    for n_ses in np.unique(list(ses_has_pid.keys())):
        sdc[f'ses_{n_ses}'] = {
            'LongName': f'Session {n_ses} status',
            'Description': f'Has session {n_ses} been completed?'
        }
        
        sdc[f'timestamp_{n_ses}'] = {
            'LongName': f'Session {n_ses} timestamp',
            'Description': f'Timestamp of session {n_ses} having been completed.'
        }
        
        sdc[f'duration_{n_ses}'] = {
            'LongName': f'Session {n_ses} duration',
            'Description': f'Duration of session {n_ses} having been completed.'
        }
        
        if _FLAG_EXTENDED:
            sdc[f'timezone_{n_ses}'] = {
                'LongName': f'Session {n_ses} timezone',
                'Description': f'Timezone in which session {n_ses} was completed.'
            }
            
            sdc[f'interest_{n_ses}_0'] = {
                'LongName': f'Session {n_ses} interest in (nature / outdoors)',
                'Description': f'Interested in (nature / outdoors) reported in session {n_ses} ranging [-1, 1] as [much less than average, much more than average].'
            }
            
            sdc[f'interest_{n_ses}_1'] = {
                'LongName': f'Session {n_ses} interest in (relationships / friendships)',
                'Description': f'Interested in (relationships / friendship) reported in session {n_ses} ranging [-1, 1] as [much less than average, much more than average].'
            }
            
            sdc[f'interest_{n_ses}_2'] = {
                'LongName': f'Session {n_ses} interest in (politics / society)',
                'Description': f'Interested in (politics / society) reported in session {n_ses} ranging [-1, 1] as [much less than average, much more than average].'
            }
            
            sdc[f'interest_{n_ses}_3'] = {
                'LongName': f'Session {n_ses} interest in (religion / spirituality)',
                'Description': f'Interested in (religion / spirituality) reported in session {n_ses} ranging [-1, 1] as [much less than average, much more than average].'
            }
            
            sdc[f'interest_{n_ses}_4'] = {
                'LongName': f'Session {n_ses} interest in (food / cuisine)',
                'Description': f'Interested in (food / cuisine) reported in session {n_ses} ranging [-1, 1] as [much less than average, much more than average].'
            }
            
            sdc[f'interest_{n_ses}_5'] = {
                'LongName': f'Session {n_ses} interest in (technology / engineering)',
                'Description': f'Interested in (technology / engineering) reported in session {n_ses} ranging [-1, 1] as [much less than average, much more than average].'
            }
    
    for i, (sid, pid) in enumerate(zip(sids, pids)):
        # grab entry
        entry = df_pro.loc[(df_pro['Participant id'] == pid)].reset_index(drop = True)
        
        # prewrite identifiers
        out['participant_id'].append(f'sub-{sid}')
        out['sid'].append(sid)
        out['pid'].append(pid)
        
        # prewrite session status
        for n_ses in np.unique(list(ses_has_pid.keys())):
            out[f'ses_{n_ses}'].append(ses_has_pid[n_ses][pid])
            out[f'timestamp_{n_ses}'].append(ses_has_stm[n_ses][pid])
            out[f'duration_{n_ses}'].append(ses_has_dur[n_ses][pid])
            
            if _FLAG_EXTENDED:
                out[f'timezone_{n_ses}'].append(ses_has_tme[n_ses][pid])
                for i in range(6):
                    out[f'interest_{n_ses}_{i}'].append(ses_has_int[n_ses][pid][i])
        
        # write requested additional data
        for idx in cfg.data.participant_descriptors:
            # grab request
            req = cfg.data.participant_descriptors[idx]
            
            # write data
            out[idx].append(
                entry[req.source].tolist()[0]
            )
            
            # write sidecar
            sdc[idx] = {
                'LongName': req['name'],
                'Description': req['description']
            }
    
    # write participant data
    df_out = dir_out / Path('participants.tsv')
    pd.DataFrame.from_dict(out).to_csv(
        df_out, sep = '\t', index = False, header = True, quoting = csv.QUOTE_NONE
    )
    
    df_out = dir_out / Path('derivatives') / Path('participants.tsv')
    pd.DataFrame.from_dict(out).to_csv(
        df_out, sep = '\t', index = False, header = True, quoting = csv.QUOTE_NONE
    )
    
    # write participant sidecar
    sc_out = dir_out / Path('participants.json')
    with open(sc_out, 'w') as f:
        json.dump(sdc, f, indent = 4, sort_keys = False)
    
    sc_out = dir_out / Path('derivatives') / Path('participants.json')
    with open(sc_out, 'w') as f:
        json.dump(sdc, f, indent = 4, sort_keys = False)
    
    # write dataset sidecar
    if cfg.verbose:
        logger.info('Preparing dataset descriptors...')
    
    sc_out = dir_out / Path('dataset_description.json')
    with open(sc_out, 'w') as f:
        # convert descriptives to dict
        sc = OmegaConf.to_container(
            cfg.data.dataset_descriptors,
            resolve = True
        )
        
        # add sources
        sc['SourceDatasets'] = [
            {
                "URL": source,
                "Version": datetime.date.today().strftime("%B %d %Y")
            }
            for source in all_sources
        ]
        
        json.dump(
            sc, f, 
            indent = 4, sort_keys = False
        )
    
    # write derivatives sidecar
    sc_out = dir_out / Path('derivatives') / Path('dataset_description.json')
    with open(sc_out, 'w') as f:
        # convert descriptives to dict
        sc = OmegaConf.to_container(
            cfg.data.derivative_descriptors,
            resolve = True
        )
        
        # add sources
        sc['SourceDatasets'] = [
            {
                "URL": str(source),
                "Version": datetime.date.today().strftime("%B %d %Y")
            }
            for source in all_outputs
        ]
        
        json.dump(
            sc, f, 
            indent = 4, sort_keys = False
        )
    
    # write readme
    if cfg.verbose:
        logger.info('Preparing dataset README...')
    
    readme = f"""### Dataset
## Identification
Name: {cfg.data.dataset_descriptors.Name}
Authors: {', '.join(cfg.data.dataset_descriptors.Authors)}

## Description
This dataset is part of a larger study examining idiosyncracies in semantic perception
of individual participants. Essentially, these data are useful for modeling subject 
variability in semantic tasks.
    """
    
    readme_out = dir_out / Path('README.md')
    
    with open(readme_out, 'w') as f:
        f.write(readme)
    
    readme = f"""### Dataset
## Identification
Name: {cfg.data.derivative_descriptors.Name}
Authors: {', '.join(cfg.data.derivative_descriptors.Authors)}

## Description
This dataset contains derivatives of `{cfg.data.dataset_descriptors.Name}`. In essence,
here we provide fitted models and evaluations per run of the pipeline to individualise
embeddings.
"""

    readme_out = dir_out / Path('derivatives') / Path('README.md')
    
    with open(readme_out, 'w') as f:
        f.write(readme)
    
    # log completion
    if cfg.verbose:
        logger.info('Conversion completed.')

if __name__ == '__main__':
    make_bids()