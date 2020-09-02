from model import Model
from single_cell_sim import *

def run_sim(Model,
            NSYNs=[4, 10, 16],
            syn_loc0 = 2000,
            t0=100, interstim=400,
            freq_stim=50, n_repeat=3,
            active=False,
            ampa_only=False,
            chelated=False):


    Modelc = Model.copy()
    tstop = t0+len(NSYNs)*interstim
    t, neuron, SEGMENTS = initialize_sim(Model,
                                         active=active,
                                         chelated_zinc=chelated,
                                         tstop=tstop)

    
    spike_IDs, spike_times = np.empty(0, dtype=int), np.empty(0, dtype=float)

    start = t0
    for n in NSYNs:
        
        synapses_loc = syn_loc0+np.arange(n)
        
        for i in range(n_repeat):
            spike_times = np.concatenate([spike_times,
                                          (start+i*1e3/freq_stim)*np.ones(len(synapses_loc))])
            spike_IDs = np.concatenate([spike_IDs,np.arange(len(synapses_loc))])

        start += interstim
        
    if ampa_only:
        Modelc['qNMDA'] = 0.
    elif chelated:
        Modelc['Deltax0'] = 0.
    Estim, ES = ntwk.process_and_connect_event_stimulation(neuron,
                                                           spike_IDs, spike_times,
                                                           synapses_loc,
                                                           EXC_SYNAPSES_EQUATIONS.format(**Modelc),
                                                           ON_EXC_EVENT.format(**Modelc))

    # recording and running
    M = ntwk.StateMonitor(neuron, ('v'), record=[0, syn_loc0])
    S = ntwk.StateMonitor(ES, ('X', 'gAMPA', 'gE_post', 'bZn'), record=[0])

    # # Run simulation
    ntwk.run(tstop*ntwk.ms)
    
    output = {'t':np.array(M.t/ntwk.ms),
              'Vm_soma':np.array(M.v/ntwk.mV)[0,:],
              'bZn_syn':np.array(S.bZn)[0,:],
              'gAMPA_syn':np.array(S.gAMPA/ntwk.nS)[0,:],
              'gNMDA_syn':np.array(S.gE_post/ntwk.nS)[0,:]-np.array(S.gAMPA/ntwk.nS)[0,:],
              'X_syn':np.array(S.X)[0,:],
              'Vm_syn':np.array(M.v/ntwk.mV)[1,:],
              'Model':Modelc}

    return output
    

if __name__=='__main__':

    run_sim(Model)
    # Nsyn = 10
    # if sys.argv[-1]=='AMPA-only':
    #     run_sim(Model)
        
