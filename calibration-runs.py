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
                                                      5, # density
                                                      cond=basal_cond,
                                                      density_factor=1./100./1e-12)

    if type(Npicked) is int:
        Npicked = Npicked*np.ones(14, dtype=int)
    np.random.seed(seed)
    spike_IDs, spike_times = np.empty(0, dtype=int), np.empty(0, dtype=float)
    for te, n in zip(stim['events'], Npicked):
        synapses_loc = np.random.choice(pre_to_iseg, n)
        spike_times = np.concatenate([spike_times,
                                      te*np.ones(len(synapses_loc))])
        spike_IDs = np.concatenate([spike_IDs,np.arange(len(synapses_loc))])
    
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

def compute_residual(mdata):

    Residual = 0
    for cond in ['20Hz_condition', '3Hz_condition']:

        tcond = (mdata['t']>(mdata['%s_tstart' % cond]-calib_data['DT0_%s' % cond])) &\
            (mdata['t']<mdata['%s_tstart' % cond]-calib_data['DT0_%s' % cond]+\
             calib_data['DTfull_%s' % cond])

        trace_model = 1e3*(mdata['Ic'][tcond]-mdata['Ic'][tcond][0])

        trace_exp = calib_data['Iexp_ctrl_%s' % cond]
        new_t = calib_data['t_%s' % cond]
        Residual += np.sum((trace_model-trace_exp)**2)/len(new_t) # normalized by sample size

    return Residual
    

def compute_time_varying_synaptic_recruitment(Nsyn1, Nsyn2, Tnsyn,
                                              npulse1 = 5, freq1= 20.,
                                              npulse2 = 9, freq2= 3.):

    Npicked = []
    for i in range(npulse1):
        Npicked.append(Nsyn2+(Nsyn1-Nsyn2)*np.exp(-i*1e3/freq1/Tnsyn))
    for i in range(npulse2):
        Npicked.append(Nsyn2+(Nsyn1-Nsyn2)*np.exp(-i*1e3/freq2/Tnsyn))

    return Npicked
    

    
    

if __name__=='__main__':

    from model import Model
    stim = build_stimulation()

    print(compute_time_varying_synaptic_recruitment(100, 80, 400))
    # output = run_single_sim(Model, stim, Npicked=100)
    # print('Residual: ', compute_residual(output))
    # np.savez(filename_with_datetime('ctrl', folder='data/calib', extension='.npz'), **output)
