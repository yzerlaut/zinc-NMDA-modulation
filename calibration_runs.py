import os, sys
import itertools
import numpy as np
from analyz.workflow.saving import filename_with_datetime
from neural_network_dynamics import main as ntwk
from single_cell_sim import initialize_sim, EXC_SYNAPSES_EQUATIONS, ON_EXC_EVENT
from analyz.IO.npz import load_dict

def run_single_sim(Model, stim,
                   Vcmd = 20,
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
    for cond, t0, freq_pulses, n_pulses in zip(['20Hz_protocol', '3Hz_protocol'],
                                               [500,1700], [20., 3.], [5, 9]) :

        new_events = t0+np.arange(n_pulses)*1e3/freq_pulses
        events = np.concatenate([events, new_events])
        model['%s_tstart' % cond] = t0
    model['events'] = events
    return model


def compute_time_varying_synaptic_recruitment(Nsyn1, Nsyn2, Tnsyn20Hz, Tnsyn3Hz,
                                              npulse1 = 5, freq1= 20.,
                                              npulse2 = 9, freq2= 3.):

    Npicked = []
    for i in range(npulse1):
        Npicked.append(Nsyn2+(Nsyn1-Nsyn2)*np.exp(-i*1e3/freq1/Tnsyn20Hz))
    for i in range(npulse2):
        Npicked.append(Nsyn2+(Nsyn1-Nsyn2)*np.exp(-i*1e3/freq2/Tnsyn3Hz))

    return np.array(Npicked, dtype=int)


def compute_chelated_residual(sim, index, calib_data, condition='chelated'):

    try:
        sim_data = load_dict(os.path.join('data', 'calib', sim.params_filename(index)+'.npz'))
        Residual = 1.
        for cond in ['20Hz_protocol', '3Hz_protocol']:
            tcond = (sim_data['t']>(sim_data['%s_tstart' % cond]-calib_data['DT0_%s' % cond])) &\
                (sim_data['t']<sim_data['%s_tstart' % cond]-calib_data['DT0_%s' % cond]+\
                 calib_data['DTfull_%s' % cond])
            
            trace_model = -1e3*(sim_data['Ic'][tcond]-sim_data['Ic'][tcond][0]) # - sign
            trace_exp = calib_data['Iexp_chelatedZn_%s' % cond]
            Residual *= 1+np.sum((trace_model-trace_exp)**2)/np.sum((trace_exp)**2)

    except FileNotFoundError:
        print(sim.params_filename(index)+'.npz', 'not found')
        Residual = 1e10
    return Residual


def compute_free_residual(sim, index, calib_data, condition='chelated'):

    Cdata = load_dict(load_dict('data/best_chelatedZn_config.npz')['filename'])
    
    try:
        sim_data = load_dict(os.path.join('data', 'calib', sim.params_filename(index)+'.npz'))
        Residual = 1.
        for cond in ['20Hz_protocol', '3Hz_protocol']:
            
            tcond = (sim_data['t']>(sim_data['%s_tstart' % cond]-calib_data['DT0_%s' % cond])) &\
                (sim_data['t']<sim_data['%s_tstart' % cond]-calib_data['DT0_%s' % cond]+\
                 calib_data['DTfull_%s' % cond])
            
            trace_model = -1e3*(sim_data['Ic'][tcond]-sim_data['Ic'][tcond][0]) # - sign
            trace_model0 = -1e3*(Cdata['Ic'][tcond]-Cdata['Ic'][tcond][0])
            
            trace_exp = calib_data['Iexp_freeZn_%s' % cond]
            trace_exp0 = calib_data['Iexp_chelatedZn_%s' % cond]

            # normalizing to peak in the exp
            first_peak_cond = (calib_data['t_%s' % cond]<calib_data['DT0_%s' % cond]+30)

            factor_exp = np.max(trace_exp[first_peak_cond])
            trace_exp /= factor_exp
            factor_exp0 = np.max(trace_exp0[first_peak_cond])
            trace_exp0 /= factor_exp0

            # normalizing to peak in the model
            factor_model = np.max(trace_model[first_peak_cond])
            trace_model /= factor_model
            factor_model0 = np.max(trace_model0[first_peak_cond])
            trace_model0 /= factor_model0
            
            diff_model = trace_model-trace_model0
            diff_exp = trace_exp-trace_exp0
            
            Residual *= 1+np.sum((trace_model-trace_exp)**2)/np.sum(trace_exp**2)
            Residual *= 1+np.sum((diff_model-diff_exp)**2)/np.sum(diff_exp**2)

    except FileNotFoundError:
        print(sim.params_filename(index)+'.npz', 'not found')
        Residual = 1e10
    return Residual


if __name__=='__main__':


    import sys, os
    from model import Model
    from analyz.workflow.batch_run import GridSimulation

    # if not os.path.isfile('data/passive-props.npz'):
    #     print('provide the ')
        
    Model['qAMPA'] = 0. # NBQX in experiments

    # loading data from previous calib !
    passive = load_dict('data/passive-props.npz')
    for key, val in passive.items():
        Model[key] = val

    # build the stimulation
    stim = build_stimulation()

    if len(sys.argv)==1:

        best_chelated_config = load_dict('data/best_chelatedZn_config.npz') 
        for key, val in best_chelated_config.items():
            Model[key] = val

        Npicked = compute_time_varying_synaptic_recruitment(Model['Nsyn1'], Model['Nsyn2'],
                                                            Model['Tnsyn20Hz'], Model['Tnsyn3Hz'])
        stim = build_stimulation()

        # 1
        # Model['alphaZn'], Model['Deltax0'] = 0, 0 # forcing "chelated-Zinc" condition
        # output = run_single_sim(Model, stim, Npicked=Npicked)
        # np.savez(os.path.join('data','calib', 'Best_chelatedZn.npz'), **output)
        
        # 2

        best_freeZn_config = load_dict('data/best_freeZn_config.npz') 
        for key, val in best_freeZn_config.items():
            Model[key] = val
        # Model['alphaZn'], Model['Deltax0'] = 0, 0 # forcing "chelated-Zinc" condition
        Model['tauRiseZn'] = 100.
        print(Model)
            
        stim = build_stimulation()
        output = run_single_sim(Model, stim, Npicked=Npicked)
        np.savez(os.path.join('data','calib', 'Best_freeZn.npz'), **output)
        
        
    elif sys.argv[1]=='chelated-zinc-calib':

        index = int(sys.argv[2])
        sim = GridSimulation(os.path.join('data', 'calib', 'chelated-zinc-calib-grid.npz'))

        sim.update_dict_from_GRID_and_index(index, Model) # update Model parameters

        Model['alphaZn'], Model['Deltax0'] = 0, 0 # forcing "chelated-Zinc" condition

        Npicked = compute_time_varying_synaptic_recruitment(Model['Nsyn1'], Model['Nsyn2'],
                                                            Model['Tnsyn20Hz'], Model['Tnsyn3Hz'])
        output = run_single_sim(Model, stim, Npicked=Npicked)
        
        np.savez(os.path.join('data', 'calib', sim.params_filename(index)), **output)


    elif sys.argv[1]=='chelated-zinc-calib-analysis':

        calib_data = load_dict('data/exp_data_for_calibration.npz')
        sim = GridSimulation(os.path.join('data', 'calib', 'chelated-zinc-calib-grid.npz'))
        Residuals = np.ones(int(sim.N))*np.inf
        for i in range(int(sim.N)):
            Residuals[i] = compute_chelated_residual(sim, i, calib_data)
        ibest = np.argmin(Residuals)
        best_chelated_config={'grid_index':ibest, 'filename':os.path.join('data','calib',sim.params_filename(ibest)+'.npz')}
        sim.update_dict_from_GRID_and_index(ibest, best_chelated_config) # update Model parameters
        np.savez('data/best_chelatedZn_config.npz', **best_chelated_config)
        print(best_chelated_config)            

    elif sys.argv[1]=='free-zinc-calib':

        index = int(sys.argv[2])
        sim = GridSimulation(os.path.join('data', 'calib', 'free-zinc-calib-grid.npz'))

        # loading data from previous calib !
        best_chelated_config = load_dict('data/best_chelatedZn_config.npz') 
        for key, val in best_chelated_config.items():
            Model[key] = val
        Npicked = compute_time_varying_synaptic_recruitment(Model['Nsyn1'], Model['Nsyn2'],
                                                            Model['Tnsyn20Hz'], Model['Tnsyn3Hz'])
        stim = build_stimulation()
        
        sim.update_dict_from_GRID_and_index(index, Model) # update Model parameters

        output = run_single_sim(Model, stim, Npicked=Npicked)
        
        np.savez(os.path.join('data', 'calib', sim.params_filename(index)), **output)
        
    elif sys.argv[1]=='free-zinc-calib-analysis':

        calib_data = load_dict('data/exp_data_for_calibration.npz')
        sim = GridSimulation(os.path.join('data', 'calib', 'free-zinc-calib-grid.npz'))
        Residuals = np.ones(int(sim.N))*np.inf
        for i in range(int(sim.N)):
            Residuals[i] = compute_free_residual(sim, i, calib_data)
        ibest = np.argmin(Residuals)
        best_free_zinc_config={'grid_index':ibest, 'filename':os.path.join('data','calib',sim.params_filename(ibest)+'.npz')}
        sim.update_dict_from_GRID_and_index(ibest, best_free_zinc_config) # update Model parameters
        np.savez('data/best_freeZn_config.npz', **best_free_zinc_config)
        print(best_free_zinc_config)
        
    else:
        # stim = build_stimulation()

        # output = run_single_sim(Model, stim,
        #                         Npicked=compute_time_varying_synaptic_recruitment(90, 60, 600, 50))
        # # print('Residual: ', compute_residual(output))
        # np.savez(filename_with_datetime('ctrl', folder='data/calib', extension='.npz'), **output)

        best_chelated_config = load_dict('data/best_chelatedZn_config.npz') 
        for key, val in best_chelated_config.items():
            Model[key] = val
        Npicked = compute_time_varying_synaptic_recruitment(Model['Nsyn1'], Model['Nsyn2'],
                                                            Model['Tnsyn20Hz'], Model['Tnsyn3Hz'])
        stim = build_stimulation()
        
