import os, sys
import itertools
import numpy as np
from analyz.workflow.saving import filename_with_datetime
from neural_network_dynamics import main as ntwk
from single_cell_sim import initialize_sim, EXC_SYNAPSES_EQUATIONS, ON_EXC_EVENT
from analyz.IO.npz import load_dict

LOCs = np.load('data/nmda-spike/locations.npy')

def run_single_sim(Model, args):
    
    Model['VC-cmd'] = args.Vcmd
    Model['alphaZn'] = args.alphaZn

    Tstim = args.n_pulses/args.freq_pulses*1e3
    stim_events = args.Tdiscard0+np.arange(args.n_pulses)*1e3/args.freq_pulses

    Model['tstop'] = args.Tdiscard0+Tstim+args.Tend
    Model['dt'] = args.dt
    
    # initialize voltage-clamp sim
    t, neuron, SEGMENTS = initialize_sim(Model, method='voltage-clamp')
    
    np.random.seed(args.seed)
    
    synapses_loc = LOCs[args.syn_location]+np.arange(args.Nsyn)
    
    spike_IDs, spike_times = np.empty(0, dtype=int), np.empty(0, dtype=float)
    for ie, te in enumerate(stim_events):
        spike_times = np.concatenate([spike_times,
                                      te*np.ones(args.Nsyn)])
        spike_IDs = np.concatenate([spike_IDs, np.arange(args.Nsyn)])

    Estim, ES = ntwk.process_and_connect_event_stimulation(neuron,
                                                           spike_IDs, spike_times,
                                                           synapses_loc,
                                                           EXC_SYNAPSES_EQUATIONS.format(**Model),
                                                           ON_EXC_EVENT.format(**Model))

    # recording and running
    M = ntwk.StateMonitor(neuron, ('v'), record=[0, synapses_loc[0]])
    S = ntwk.StateMonitor(ES, ('X', 'gAMPA', 'gRiseNMDA', 'gDecayNMDA', 'bZn'), record=[0])

    # # Run simulation
    ntwk.run((args.Tdiscard0+Tstim+args.Tend)*ntwk.ms)

    output = {'t':np.array(M.t/ntwk.ms), 'Vcmd':args.Vcmd}
    output['synapses_loc'] = synapses_loc
    output['Vm_soma'] = np.array(M.v/ntwk.mV)[0,:]
    output['gAMPA_syn'] = np.array(S.gAMPA/ntwk.nS)[0,:]
    output['X_syn'] = np.array(S.X)[0,:]
    bZn, gRise, gDecay = np.array(S.bZn)[0,:], np.array(S.gRiseNMDA)[0,:], np.array(S.gDecayNMDA)[0,:]
    output['Vm_syn'] = np.array(M.v/ntwk.mV)[1,:]
    output['bZn_syn'] = bZn
    output['gNMDA_syn'] = Model['qNMDA']*Model['nNMDA']*(gDecay-gRise)/(1+0.3*np.exp(-output['Vm_syn']/Model['V0NMDA']))*(1.-Model['alphaZn']*bZn)
    output['Ic'] = (output['Vm_soma']-Model['VC-cmd'])*Model['VC-gclamp'] # nA

    output['args'] = dict(vars(args))
    
    return output

def plot_sim(tdiscard=100, tend=1000):

    Data = np.load('data/pair_calib.npy', allow_pickle=True).item()

    cond = (Data['t']>tdiscard) & (Data['t']<tend)
    
    from datavyz import ge
    fig, ax= ge.figure(figsize=(2,2))
    for i in range(len(Data['alphaZn'])):
        ge.plot(Data['t'][cond], np.array(Data['Current'][i]).mean(axis=0)[cond],
                color=ge.viridis(i/len(Data['alphaZn'])), ax=ax, no_set=True)
    
    ge.show()
    
if __name__=='__main__':

    # import sys, os
    from model import Model as Model0
    Model = Model0.copy()
    
    Model['qAMPA'] = 0. # blocked with NBQX

    
    import argparse
    # First a nice documentation 
    parser=argparse.ArgumentParser(description=
    """ 
    Protocol to study the interaction betweeen background and evoked activity
    """
    ,formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("task",\
        help="""
        Task to be performed, either:
        - run
        - demo
        - plot
        """, default='demo')
    # # stim props
    parser.add_argument("-Nsl", "--N_syn_location",help="#", type=int, default=15)
    parser.add_argument("-sl", "--syn_location",help="#", type=int, default=20)
    parser.add_argument("--Nsyn",help="#", type=int, default=1)
    parser.add_argument("-aZn", "--alphaZn",
                        help="inhibition factor in free Zinc condition",
                        type=float, default=.35)
    parser.add_argument("--alphaZnMax", type=float, default=0.5)
    parser.add_argument("--Vcmd", type=float, default=20.)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--freq_pulses", type=float, default=20.)
    parser.add_argument("--n_pulses", type=int, default=5)
    parser.add_argument("--Tdiscard0", type=float, default=200)
    parser.add_argument("--Tend", type=float, default=200)
    parser.add_argument("-NaZn", "--NalphaZn", type=int, default=30)

    parser.add_argument("-s", "--seed", help="#", type=int, default=1)

    args = parser.parse_args()

    if args.task=='demo':
    
        data = run_single_sim(Model, args)

        from datavyz import ge

        ge.plot(data['t'], 1e3*data['Ic'])

        ge.show()

    elif args.task=='run':

        Data = {'Current':[], 'alphaZn':np.linspace(0, args.alphaZnMax, args.NalphaZn)}

        for args.alphaZn in Data['alphaZn']:
            Data['Current'].append([])
            for args.syn_location in np.random.choice(np.arange(len(LOCs)), args.N_syn_location):
                data = run_single_sim(Model, args)
                print(Model['alphaZn'])
                Data['Current'][-1].append(data['Ic'])
        Data['t'] = data['t']
        np.save('data/pair_calib.npy', Data)

    elif args.task=='plot':

        plot_sim()
        # from datavyz import ge

        # ge.plot(data['t'], 1e3*data['Ic'])

        # ge.show()

