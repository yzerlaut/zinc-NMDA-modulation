"""
functions intended to be used in notebooks at the root of the repository
"""
import os
import numpy as np
from analyz.IO.files_manip import get_files_with_given_exts
from analyz.IO.npz import load_dict

def count_spikes(t, Vm, t0, t1, threshold=0):
    cond = (t>t0) & (t<t1)
    return len(np.argwhere((Vm[cond][:-1]<threshold) & (Vm[cond][1:]>=threshold)))

def build_full_dataset(key='passive',
                       filename_only=True,                       
                       nfile_start=0, nfile_stop=int(1e8)):
    """
    """
    filenames = get_files_with_given_exts('data/bg-modul/%s' % key, '.npz')
    RESP = {'filename':[]}
    K = ['Vm_soma', 't', 'BG_raster', 'STIM_raster']
    KEYS = ['seed', 'stimseed', 'alphaZn','syn_location',
            'bg_level', 'NSTIMs', 'ampa_only',
            'duration_per_bg_level', 'stim_delay']
    for key in KEYS+K:
        RESP[key] = []

    for i, fn in enumerate(filenames[nfile_start:nfile_stop]):
        if os.path.isfile(fn):
            try:
                data = load_dict(fn)
                if np.isfinite(data['Vm_soma'].max()):
                    RESP['filename'].append(fn)
                    for key in KEYS:
                        RESP[key].append(data['args'][key])
                    if not filename_only:
                        for key in K:
                            RESP[key].append(data[key])    
            except (KeyError, AttributeError):
                print(fn, 'not valid')
    for key in KEYS:
        RESP[key] = np.array(RESP[key])
        
    return RESP

def build_full_dataset_per_stim(key='passive',
                                nfile_start=0, nfile_stop=int(1e8),
                                filename_only=True,
                                spike_window=200):
    filenames = get_files_with_given_exts('data/bg-modul/%s' % key, '.npz')
    RESP_PER_STIM = {'Vm':[], 'freq':[], 'Nspike':[], 'spike':[], 'nstim':[], 'filename':[]}
    KEYS = ['seed', 'stimseed', 'alphaZn','syn_location', 'bg_level', 'ampa_only']
    for k in KEYS:
        RESP_PER_STIM[k] = []

    for fn in filenames[nfile_start:nfile_stop]:
        if not os.path.isdir(fn):
            data = load_dict(fn)
            if np.isfinite(data['Vm_soma'].max()) and ('stimseed' in data['args']):
                for i, nstim in enumerate(data['args']['NSTIMs']):
                    t0 = i*data['args']['duration_per_bg_level']
                    t1 = t0+data['args']['duration_per_bg_level']
                    tcond = (data['t']>=t0) & (data['t']<t1)
                    RESP_PER_STIM['Vm'].append(data['Vm_soma'][tcond])
                    RESP_PER_STIM['nstim'].append(nstim)
                    for k in KEYS:
                        RESP_PER_STIM[k].append(data['args'][k])
                    if key=='active':
                        n = count_spikes(data['t'], data['Vm_soma'],
                                            data['args']['stim_delay']+t0,
                                            data['args']['stim_delay']+t0+spike_window)
                    else:
                        n=0
                    RESP_PER_STIM['Nspike'].append(n)
                    RESP_PER_STIM['spike'].append(np.sign(n)) # 0/1 spike/no-spike
                    RESP_PER_STIM['freq'].append(n/spike_window*1e3)
                RESP_PER_STIM['t'] = data['t'][tcond]-t0
                RESP_PER_STIM['stim_delay'] = data['args']['stim_delay']
    for key in RESP_PER_STIM.keys():
        RESP_PER_STIM[key] = np.array(RESP_PER_STIM[key])
    return RESP_PER_STIM
