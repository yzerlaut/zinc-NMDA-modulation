import numpy as np
from neural_network_dynamics import main as ntwk

from model import Model

from single_cell_sim import initialize_sim,\
    EXC_SYNAPSES_EQUATIONS, INH_SYNAPSES_EQUATIONS,\
    ON_INH_EVENT, ON_EXC_EVENT


def distance(x, y, z): # soma is at (0, 0, 0)
    return np.sqrt(x**2+y**2+z**2)

def compute_branches_for_stimuli(SEGMENTS,
                                 AREA_THRESHOLD = 100e-12,
                                 DISTANCE_THRESHOLD = 50e-6):
    
    segments_of_branches = []

    for sn in np.unique(SEGMENTS['name']):
        if ('dend' in sn):
            dend_condition = (sn==SEGMENTS['name'])
            basal_condition = distance(SEGMENTS['x'], SEGMENTS['y'], SEGMENTS['z'])>DISTANCE_THRESHOLD
            if np.sum(SEGMENTS['area'][dend_condition & basal_condition])>AREA_THRESHOLD:
                segments_of_branches.append(SEGMENTS['index'][dend_condition & basal_condition])
                
    return segments_of_branches

def run_single_trial(Model, Nsyn_array,
                     DT=200, delay=70, seed=0, recovery=200):

    np.random.seed(seed)
    
    morpho = ntwk.Morphology.from_swc_file(Model['morpho_file'])
    SEGMENTS = ntwk.morpho_analysis.compute_segments(morpho)
    nseg = len(SEGMENTS['x'])
    segments_of_branches = compute_branches_for_stimuli(SEGMENTS)
    isegs_branch = segments_of_branches[Model['branch_index']]

    Model['tstop'] = delay+len(Nsyn_array)*DT
    
    t, neuron, SEGMENTS = initialize_sim(Model)
    
    spike_IDs, spike_times = np.empty(0, dtype=int), np.empty(0, dtype=float)

    for i, n in enumerate(Nsyn_array):

        nsyn = int(min([len(isegs_branch), n]))
        synapses_loc = np.random.choice(isegs_branch, nsyn, replace=False)
        spike_times = np.concatenate([spike_times, delay+i*DT*np.ones(len(synapses_loc))])
        spike_IDs = np.concatenate([spike_IDs, np.arange(len(synapses_loc))])
        it = 0
        while nsyn<n:
            nsyn2 = int(min([len(isegs_branch), n-nsyn]))
            synapses_loc = np.random.choice(isegs_branch, nsyn2, replace=False)
            spike_times = np.concatenate([spike_times,
                                          delay+i*DT*np.ones(len(synapses_loc))+(it+1)*Model['dt']])
            spike_IDs = np.concatenate([spike_IDs, np.arange(len(synapses_loc))])
            nsyn += nsyn2
            it+=1

    Estim, ES = ntwk.process_and_connect_event_stimulation(neuron,
                                                           spike_IDs, spike_times,
                                                           isegs_branch,
                                                           EXC_SYNAPSES_EQUATIONS.format(**Model),
                                                           ON_EXC_EVENT.format(**Model))

    # recording and running
    M = ntwk.StateMonitor(neuron, ('v'), record=[0, synapses_loc[0]])
    
    # # Run simulation
    ntwk.run((delay+recovery+DT*len(Nsyn_array))*ntwk.ms)

    return np.array(M.t/ntwk.ms), np.array(M.v/ntwk.mV)[0,:]




# def run_single_trial_with_noise(NSYN, Model,
#                                 n0 = 20,
#                                 Fexc=0, Finh=0.,
#                                 DT=300, delay=70, seed=0):

#     morpho = ntwk.Morphology.from_swc_file(Model['morpho_file'])
#     SEGMENTS = ntwk.morpho_analysis.compute_segments(morpho)
#     nseg = len(SEGMENTS['x'])
    
#     basal_cond = ntwk.morpho_analysis.find_conditions(SEGMENTS,
#                                                 comp_type='dend',
#                                                 min_distance_to_soma=20e-6)
#     prox_cond = ntwk.morpho_analysis.find_conditions(SEGMENTS,
#                                                      comp_type=['dend', 'soma'],
#                                                      max_distance_to_soma=20e-6)

#     # spreading synapses for bg noise

#     Nsyn_Glut,\
#         pre_to_iseg_Glut,\
#         Nsyn_per_seg_Glut = ntwk.spread_synapses_on_morpho(SEGMENTS,
#                                                            4,
#                                                            cond=basal_cond,
#                                                            density_factor=1./100./1e-12)

#     Nsyn_GABAprox,\
#         pre_to_iseg_GABAprox,\
#         Nsyn_per_seg_GABAprox = ntwk.spread_synapses_on_morpho(SEGMENTS,
#                                                                2,
#                                                                cond=prox_cond,
#                                                                density_factor=1./100./1e-12)

#     Nsyn_GABAdist,\
#         pre_to_iseg_GABAdist,\
#         Nsyn_per_seg_GABAdist = ntwk.spread_synapses_on_morpho(SEGMENTS,
#                                                                1,
#                                                                cond=basal_cond,
#                                                                density_factor=1./100./1e-12)
#     Nsyn_GABA = Nsyn_GABAdist+Nsyn_GABAprox
#     Nsyn_per_seg_GABA = Nsyn_per_seg_GABAdist+Nsyn_per_seg_GABAprox
#     pre_to_iseg_GABA = np.concatenate([pre_to_iseg_GABAprox,
#                                        len(pre_to_iseg_GABAprox)+pre_to_iseg_GABAdist])

    
#     Model['tstop'] = delay+len(NSYN)*DT
    
#     t, neuron, SEGMENTS = initialize_sim(Model)

#     if Fexc>0:
#         spike_IDs, spike_times = ntwk.spikes_from_time_varying_rate(t,
#                                                                 0*t+Fexc,
#                                                                     N=Nsyn_Glut, SEED=seed)
#     else:
#         spike_IDs, spike_times = np.empty(0, dtype=int), np.empty(0, dtype=float)

#     for i, n in enumerate(NSYN):
#         synapses_loc = pre_to_iseg_Glut[n0:n0+n]
#         spike_times = np.concatenate([spike_times,
#                                       delay+i*DT*np.ones(len(synapses_loc))])
#         spike_IDs = np.concatenate([spike_IDs,np.arange(len(synapses_loc))])


#     spike_IDs, spike_times = ntwk.deal_with_multiple_spikes_per_bin(spike_IDs, spike_times, t)
#     Estim, ES = ntwk.process_and_connect_event_stimulation(neuron,
#                                                            spike_IDs, spike_times,
#                                                            pre_to_iseg_Glut,
#                                                            EXC_SYNAPSES_EQUATIONS.format(**Model),
#                                                            ON_EXC_EVENT.format(**Model))

#     if Finh>0:
#         spike_IDs, spike_times = ntwk.spikes_from_time_varying_rate(t,
#                                                                     0*t+Finh,
#                                                                     N=Nsyn_GABA, SEED=seed+1)

#         Istim, IS = ntwk.process_and_connect_event_stimulation(neuron,
#                                                                spike_IDs, spike_times,
#                                                                pre_to_iseg_GABA,
#                                                                INH_SYNAPSES_EQUATIONS.format(**Model),
#                                                                ON_INH_EVENT.format(**Model))
#     else:
#         Istim, IS = None, None
    
#     # recording and running
#     M = ntwk.StateMonitor(neuron, ('v'), record=[0, synapses_loc[0]])
    
#     # # Run simulation
#     ntwk.run(DT*len(NSYN)*ntwk.ms)

#     return np.array(M.t/ntwk.ms), np.array(M.v/ntwk.mV)[0,:]


    
# CTRLs, CHELATEDs, LINs = [], [], []
# # NSYN = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 100])
# NSYN = np.logspace(np.log10(1), np.log10(1000), 30)

# t0_stim = 50
# ctrl = Model['Deltax0']
# qNMDA = Model['qNMDA']


# FREQS = [(0,0), (1., 0.1), (1,1), (2,1), (2,2), (5,4)]
# for i, (fe, fi) in enumerate(FREQS):

#     # control
#     Model['Deltax0'], Model['qNMDA'] = ctrl, qNMDA
#     t, v = run_single_trial(NSYN, Model, Fexc=fe, Finh=fe, seed=i**2)
#     CTRLs.append(v)
#     # chelated
#     Model['Deltax0'], Model['qNMDA'] = 0, qNMDA
#     t, v = run_single_trial(NSYN, Model, Fexc=fe, Finh=fe, seed=i**2)
#     CHELATEDs.append(v)
#     # # linear (AMPA only)
#     # Model['Deltax0'], Model['qNMDA'] = 0, 0
#     # t, v = run_single_trial(NSYN, Model)
#     # LINs.append(v)
    
# np.savez('data.npz', **{'CTRLs':np.array(CTRLs),
#                         'CHELATEDs':CHELATEDs,
#                         't':t, 'FREQS':FREQS, 'delay':np.array([70.]),
#                         'NSYN':np.array(NSYN)})

if __name__=='__main__':

    import sys
    from model import Model
    from analyz.workflow.batch_run import GridSimulation
    
    if sys.argv[1]=='syn-input':
        
        index = int(sys.argv[5])
        sim = GridSimulation(os.path.join('data', 'syn-input', 'spike-threshold-grid.npz'))
        sim.update_dict_from_GRID_and_index(index, Model) # update Model parameters
        
        Nsyn_array = np.logspace(np.log10(int(sys.argv[2])), np.log10(int(sys.argv[3])),
                                 int(sys.argv[4]), dtype=int)
        t, v = run_single_trial(Model, Nsyn_array)
        np.savez(os.path.join('data', 'syn-input', sim.params_filename(index)), **{'t':t, 'v':v})
        
        
    else:
        Nsyn_array = np.logspace(0, 3, 4)
        t, v = run_single_trial(Model, Nsyn_array, 0)
        from datavyz import ges as ge
        ge.plot(t, v)
        ge.show()
