import numpy as np
from neural_network_dynamics import main as ntwk

from model import Model

from single_cell_sim import initialize_sim,\
    EXC_SYNAPSES_EQUATIONS, INH_SYNAPSES_EQUATIONS,\
    ON_INH_EVENT, ON_EXC_EVENT


morpho = ntwk.Morphology.from_swc_file(Model['morpho_file_1'])
SEGMENTS = ntwk.morpho_analysis.compute_segments(morpho)
nseg = len(SEGMENTS['x'])

basal_cond = ntwk.morpho_analysis.find_conditions(SEGMENTS,
                                            comp_type='dend',
                                            min_distance_to_soma=20e-6)
prox_cond = ntwk.morpho_analysis.find_conditions(SEGMENTS,
                                                 comp_type=['dend', 'soma'],
                                                 max_distance_to_soma=20e-6)

# spreading synapses for bg noise

Nsyn_Glut,\
    pre_to_iseg_Glut,\
    Nsyn_per_seg_Glut = ntwk.spread_synapses_on_morpho(SEGMENTS,
                                                       4,
                                                       cond=basal_cond,
                                                       density_factor=1./100./1e-12)

Nsyn_GABAprox,\
    pre_to_iseg_GABAprox,\
    Nsyn_per_seg_GABAprox = ntwk.spread_synapses_on_morpho(SEGMENTS,
                                                           2,
                                                           cond=prox_cond,
                                                           density_factor=1./100./1e-12)

Nsyn_GABAdist,\
    pre_to_iseg_GABAdist,\
    Nsyn_per_seg_GABAdist = ntwk.spread_synapses_on_morpho(SEGMENTS,
                                                           1,
                                                           cond=basal_cond,
                                                           density_factor=1./100./1e-12)
Nsyn_GABA = Nsyn_GABAdist+Nsyn_GABAprox
Nsyn_per_seg_GABA = Nsyn_per_seg_GABAdist+Nsyn_per_seg_GABAprox
pre_to_iseg_GABA = np.concatenate([pre_to_iseg_GABAprox,
                                   len(pre_to_iseg_GABAprox)+pre_to_iseg_GABAdist])


def run_single_trial(NSYN, Model,
                     n0 = 20,
                     Fexc=0, Finh=0.,
                     DT=300, delay=70, seed=0):

    Model['tstop'] = delay+len(NSYN)*DT
    
    t, neuron, SEGMENTS = initialize_sim(Model)

    if Fexc>0:
        spike_IDs, spike_times = ntwk.spikes_from_time_varying_rate(t,
                                                                0*t+Fexc,
                                                                    N=Nsyn_Glut, SEED=seed)
    else:
        spike_IDs, spike_times = np.empty(0, dtype=int), np.empty(0, dtype=float)

    for i, n in enumerate(NSYN):
        synapses_loc = pre_to_iseg_Glut[n0:n0+n]
        spike_times = np.concatenate([spike_times,
                                      delay+i*DT*np.ones(len(synapses_loc))])
        spike_IDs = np.concatenate([spike_IDs,np.arange(len(synapses_loc))])


    spike_IDs, spike_times = ntwk.deal_with_multiple_spikes_per_bin(spike_IDs, spike_times, t)
    Estim, ES = ntwk.process_and_connect_event_stimulation(neuron,
                                                           spike_IDs, spike_times,
                                                           pre_to_iseg_Glut,
                                                           EXC_SYNAPSES_EQUATIONS.format(**Model),
                                                           ON_EXC_EVENT.format(**Model))

    if Finh>0:
        spike_IDs, spike_times = ntwk.spikes_from_time_varying_rate(t,
                                                                    0*t+Finh,
                                                                    N=Nsyn_GABA, SEED=seed+1)

        Istim, IS = ntwk.process_and_connect_event_stimulation(neuron,
                                                               spike_IDs, spike_times,
                                                               pre_to_iseg_GABA,
                                                               INH_SYNAPSES_EQUATIONS.format(**Model),
                                                               ON_INH_EVENT.format(**Model))
    else:
        Istim, IS = None, None
    
    # recording and running
    M = ntwk.StateMonitor(neuron, ('v'), record=[0, synapses_loc[0]])
    
    # # Run simulation
    ntwk.run(DT*len(NSYN)*ntwk.ms)

    return np.array(M.t/ntwk.ms), np.array(M.v/ntwk.mV)[0,:]


CTRLs, CHELATEDs, LINs = [], [], []
NSYN = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500])

t0_stim = 50
ctrl = Model['Deltax0']
qNMDA = Model['qNMDA']


FREQS = [(0,0), (1., 0.1), (1,1), (2,1), (2,2), (5,4)]
for i, (fe, fi) in enumerate(FREQS):

    # control
    Model['Deltax0'], Model['qNMDA'] = ctrl, qNMDA
    t, v = run_single_trial(NSYN, Model, Fexc=fe, Finh=fe, seed=i**2)
    CTRLs.append(v)
    # chelated
    Model['Deltax0'], Model['qNMDA'] = 0, qNMDA
    t, v = run_single_trial(NSYN, Model, Fexc=fe, Finh=fe, seed=i**2)
    CHELATEDs.append(v)
    # # linear (AMPA only)
    # Model['Deltax0'], Model['qNMDA'] = 0, 0
    # t, v = run_single_trial(NSYN, Model)
    # LINs.append(v)
    
np.savez('data.npz', **{'CTRLs':np.array(CTRLs),
                        'CHELATEDs':CHELATEDs,
                        't':t, 'FREQS':FREQS, 'delay':np.array([70.]),
                        'NSYN':np.array(NSYN)})
