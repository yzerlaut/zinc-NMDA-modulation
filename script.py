import numpy as np
import os, time
import multiprocessing as mp
import itertools, time

from single_cell_sim import *
from model import Model

LOCs = [74, 124, 313, 363, 474, 630, 680, 800, 886, 936, 1036, 1227, 1428, 1478, 1634, 1856, 2117, 2183, 2233, 2379, 2429, 3177, 3319, 3464, 3514]

def run_sim(Model,
            syn_packets=[{'loc':0, 'nsyn':5, 'label':'1'},
                         {'loc':1, 'nsyn':5, 'label':'2'}],
            activation_sequence=[{'time':200,
                                  'packet':0},
                                 {'time':400,
                                  'packet':1}],
            syn_packet_width=5,
            Nsyn_per_loc=20,
            post_duration=200.,
            stim_seed=2,
            active=False,
            ampa_only=False,
            chelated=False):


    Modelc = Model.copy()
    tstop = activation_sequence[-1]['time']+post_duration
    if chelated:
        Modelc['alphaZn'] = 0.

    t, neuron, SEGMENTS = initialize_sim(Model,
                                         active=active,
                                         tstop=tstop, verbose=False)

    np.random.seed(stim_seed)
    
    PACKETS, ulocs = [], np.unique([sp['loc'] for sp in syn_packets])
    for i, sp in enumerate(syn_packets):
        i0 = np.argwhere(sp['loc']==ulocs).flatten()[0] # to catch syn packets at the same location
        locs = LOCs[sp['loc']]+np.arange(Nsyn_per_loc)
        times = np.sort(np.random.randn(sp['nsyn'])*syn_packet_width)
        IDs = np.random.choice(i0*Nsyn_per_loc+np.arange(Nsyn_per_loc), sp['nsyn'], replace=False)
        labels = [sp['label'] for i in range(sp['nsyn'])]
        PACKETS.append({'times':times, 'IDs':IDs, 'locs':locs, 'labels':labels})
        
    syn_IDs, syn_times, syn_labels = np.empty(0, dtype=int), np.empty(0, dtype=float), np.empty(0, dtype=str)

    for i, act in enumerate(activation_sequence):
        syn_times = np.concatenate([syn_times,
                                    act['time']+PACKETS[act['packet']]['times']])
        syn_IDs = np.concatenate([syn_IDs,
                                  PACKETS[act['packet']]['IDs']])
        syn_labels = np.concatenate([syn_labels, PACKETS[act['packet']]['labels']])

    if ampa_only:
        Modelc['qNMDA'] = 0.
    elif chelated:
        Modelc['Deltax0'] = 0.
        
    Estim, ES = ntwk.process_and_connect_event_stimulation(neuron,
                                                           syn_IDs, syn_times,
                                                           np.concatenate([(LOCs[ul]+np.arange(Nsyn_per_loc)) for ul in ulocs]),
                                                           EXC_SYNAPSES_EQUATIONS.format(**Modelc),
                                                           ON_EXC_EVENT.format(**Modelc))

    # recording and running
    M = ntwk.StateMonitor(neuron, ('v'), record=[0]+[sp['loc'] for sp in syn_packets])
    S = ntwk.StateMonitor(ES, ('X', 'gAMPA', 'gRiseNMDA', 'gDecayNMDA', 'bZn'), record=[0])

    # # Run simulation
    ntwk.run(tstop*ntwk.ms)
    
    data = {'t':np.array(M.t/ntwk.ms), 
            'Vm_soma':np.array(M.v/ntwk.mV)[0,:],
            'bZn_syn':np.array(S.bZn)[0,:],
            'gAMPA_syn':np.array(S.gAMPA/ntwk.nS)[0,:],
            'X_syn':np.array(S.X)[0,:],
            'Vm_syn':np.array(M.v/ntwk.mV)[1,:],
            'syn_times': syn_times,
            'syn_IDs': syn_IDs,
            'syn_labels': syn_labels,
            'Model':Modelc}
    data['gNMDA_syn']= Model['qNMDA']*Model['nNMDA']*\
        (np.array(S.gDecayNMDA)[0,:]-np.array(S.gRiseNMDA)[0,:])\
        /(1+Model['etaMg']*Model['cMg']*np.exp(-data['Vm_syn']/Model['V0NMDA']))\
        *(1-Model['alphaZn']*data['bZn_syn'])
    
    for obj in [neuron, M, S, Estim, ES, SEGMENTS]:
        del(obj)
        
    return data

active = False

interstims = np.logspace(np.log10(30), np.log10(400), 18)
nsyns = 5+np.arange(18)
nlocs = len(LOCs)
delta_Vm = np.zeros((len(interstims), len(nsyns), nlocs))

def get_dVm(X):
    
    time.sleep(0.5)
    iDt, iNsyn, iloc = X # extract from array
    
    DT, nsyn = interstims[iDt], nsyns[iNsyn]
    #print('loc #%i, Nsyn=%i, dt=%.1f \n ' % (iloc+1, nsyn, DT))
    
    syn_packets=[{'loc':iloc, 'nsyn':nsyn, 'label':'1'}]

    data_freeZn = run_sim(Model, syn_packets=syn_packets, Nsyn_per_loc=nsyn,
                            activation_sequence=[{'time':20, 'packet':0}, {'time':20+DT, 'packet':0}],
                            chelated=False, active=active)
    data_chelatedZn = run_sim(Model, syn_packets=syn_packets, Nsyn_per_loc=nsyn,
                                activation_sequence=[{'time':20, 'packet':0}, {'time':20+DT, 'packet':0}],
                                chelated=True, active=active)
    return (np.max(data_chelatedZn['Vm_soma'])-np.max(data_freeZn['Vm_soma']))

n, P, tstart = 0, [], time.time()
for iDt, iNsyn, iloc in itertools.product(range(len(interstims)), range(len(nsyns)), range(nlocs)):
    n+=1
    P.append((iDt, iNsyn, iloc,))
    if n%mp.cpu_count()==0:
        print('sim #%i/%i (%.1f%%, %.1f min)' % (n, len(interstims)*len(nsyns)*nlocs, 100.*n/(len(interstims)*len(nsyns)*nlocs), 1./60.*(time.time()-tstart)))
        # we do all simulations
        pool = mp.Pool(mp.cpu_count())
        result = pool.map(get_dVm, P)
        for p, r in zip(P, result):
            delta_Vm[p[0], p[1], p[2]] = r
        P = []
        
np.save('delta_Vm.npy', {'delta_Vm':delta_Vm, 'interstims':interstims, 'nsyns':nsyns})    
