import os, sys
import numpy as np
from analyz.workflow.saving import filename_with_datetime
from neural_network_dynamics import main as ntwk
from single_cell_sim import initialize_sim, EXC_SYNAPSES_EQUATIONS, ON_EXC_EVENT
from analyz.IO.npz import load_dict

def run_single_sim(Model, stim,
                   Vcmd = 0,
                   Npicked=100,
                   seed=0):

    Model['VC-cmd'] = Vcmd
    Model['tstop'] = stim['t'][-1]
    
    # initialize voltage-clamp sim
    t, neuron, SEGMENTS = initialize_sim(Model, method='voltage-clamp')

    # find synapses
    basal_cond = ntwk.morpho_analysis.find_conditions(SEGMENTS,
                                                  comp_type='dend',
                                                  min_distance_to_soma=20e-6)
    Nsyn, pre_to_iseg,\
        Nsyn_per_seg = ntwk.spread_synapses_on_morpho(SEGMENTS,
                                                      10, # density
                                                      cond=basal_cond,
                                                      density_factor=1./100./1e-12)

    if type(Npicked) is int:
        Npicked = Npicked*np.ones(14, dtype=int)
        
    np.random.seed(seed)
    
    synapses_loc = np.random.choice(pre_to_iseg, np.max(Npicked))
    
    spike_IDs, spike_times = np.empty(0, dtype=int), np.empty(0, dtype=float)
    for te, n in zip(stim['events'], Npicked):
        spike_times = np.concatenate([spike_times,
                                      te*np.ones(n)])
        spike_IDs = np.concatenate([spike_IDs,np.arange(n)])
    
    Estim, ES = ntwk.process_and_connect_event_stimulation(neuron,
                                                           spike_IDs, spike_times,
                                                           synapses_loc,
                                                           EXC_SYNAPSES_EQUATIONS.format(**Model),
                                                           ON_EXC_EVENT.format(**Model))

    # recording and running
    M = ntwk.StateMonitor(neuron, ('v'), record=[0, synapses_loc[0]])
    S = ntwk.StateMonitor(ES, ('X', 'gAMPA', 'gRiseNMDA', 'gDecayNMDA', 'bZn'), record=[0])

    # # Run simulation
    ntwk.run(Model['tstop']*ntwk.ms)

    output = {'t':np.array(M.t/ntwk.ms), 'Vcmd':Vcmd}
    output['synapses_loc'] = synapses_loc
    output['Npicked'] = Npicked
    output['Vm_soma'] = np.array(M.v/ntwk.mV)[0,:]
    output['gAMPA_syn'] = np.array(S.gAMPA/ntwk.nS)[0,:]
    output['X_syn'] = np.array(S.X)[0,:]
    bZn, gRise, gDecay = np.array(S.bZn)[0,:], np.array(S.gRiseNMDA)[0,:], np.array(S.gDecayNMDA)[0,:]
    output['Vm_syn'] = np.array(M.v/ntwk.mV)[1,:]
    output['bZn_syn'] = bZn
    output['gNMDA_syn'] = Model['qNMDA']*Model['nNMDA']*(gDecay-gRise)/(1+0.3*np.exp(-output['Vm_syn']/Model['V0NMDA']))*(1.-Model['alphaZn']*bZn)
    output['Ic'] = (output['Vm_soma']-Model['VC-cmd'])*Model['VC-gclamp'] # nA

    for key in stim:
        if key not in output:
            output[key] = stim[key]
    return output


def build_stimulation():
    
    dt, tstop = 0.1, 5000.
    t = np.arange(int(tstop/dt))*dt
    model = {'t':t}
    events = np.empty(0)
    for cond, t0, freq_pulses, n_pulses in zip(['20Hz_condition', '3Hz_condition'],
                                               [500,1700], [20., 3.], [5, 9]) :

        new_events = t0+np.arange(n_pulses)*1e3/freq_pulses
        events = np.concatenate([events, new_events])
        model['%s_tstart' % cond] = t0
    model['events'] = events
    return model


calib_data = load_dict('data/exp_data_for_calibration.npz')


def compute_time_varying_synaptic_recruitment(Nsyn1, Nsyn2, Tnsyn,
                                              npulse1 = 5, freq1= 20.,
                                              npulse2 = 9, freq2= 3.):

    Npicked = []
    for i in range(npulse1):
        Npicked.append(Nsyn2+(Nsyn1-Nsyn2)*np.exp(-i*1e3/freq1/Tnsyn))
    for i in range(npulse2):
        Npicked.append(Nsyn2+(Nsyn1-Nsyn2)*np.exp(-i*1e3/freq2/Tnsyn))

    return np.array(Npicked, dtype=int)
    

    
    

if __name__=='__main__':


    import sys, os
    from model import Model

    if not os.path.isfile('data/passive-props.npz'):
        print('provide the ')
        
    Model['qAMPA'] = 0. # NBQX in experiments

    if sys.argv[1]=='chelated-zinc-calib':

        Tnmda, Nsyn1, Nsyn2, Tnsyn = sys.argv[2:]
        filename = '%s-%s-%s-%s.npz' % (Tnmda, Nsyn1, Nsyn2, Tnsyn)
        
        Model['alphaZn'] = 0
        Model['Deltax0'] = 0

        Model['tauDecayNMDA'] = float(Tnmda)
        stim = build_stimulation()

        Npicked = compute_time_varying_synaptic_recruitment(int(Nsyn1), int(Nsyn2), float(Tnsyn))
        output = run_single_sim(Model, stim, Npicked=Npicked)
        
        np.savez(os.path.join('data', 'calib', 'chelated-zinc', filename), **output)

    elif sys.argv[1]=='free-zinc-calib':


        alphaZn, tauRiseZn, tauDecayZn, Deltax0, deltax = sys.argv[2:]
        filename = '%s-%s-%s-%s-%s.npz' % (alphaZn, tauRiseZn, tauDecayZn, Deltax0, deltax)

        # using the chelated zinc configuration
        Tnmda, Nsyn1, Nsyn2, Tnsyn = 115.0, 60, 40, 500.0
        Npicked = compute_time_varying_synaptic_recruitment(int(Nsyn1), int(Nsyn2), float(Tnsyn))
        Model['tauDecayNMDA'] = float(Tnmda)

        
        Model['alphaZn'] = float(alphaZn)
        Model['tauRiseZn'] = float(tauRiseZn)
        Model['tauDecayZn'] = float(tauDecayZn)
        Model['Deltax0'] = float(Deltax0)
        Model['x0'] = float(Deltax0) # forced to the same value for now
        Model['deltax'] = float(deltax)

        stim = build_stimulation()
        output = run_single_sim(Model, stim, Npicked=Npicked)
        
        np.savez(os.path.join('data', 'calib', 'free-zinc', filename), **output)
        
    else:
        stim = build_stimulation()

        output = run_single_sim(Model, stim,
                                Npicked=compute_time_varying_synaptic_recruitment(90, 60, 600))
        # print('Residual: ', compute_residual(output))
        np.savez(filename_with_datetime('ctrl', folder='data/calib', extension='.npz'), **output)
