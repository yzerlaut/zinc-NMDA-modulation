import sys, os
from model import Model
from single_cell_sim import *
from analyz.IO.npz import load_dict


def single_poisson_process_BG(Fbg, duration, tstart=0, seed=0):
    # time in [ms], frequency in [Hz]
    np.random.seed(seed)
    poisson = np.cumsum(np.random.exponential(1./Fbg, size=int(3*duration/1e3*Fbg)))
    poisson = poisson[poisson<=1e-3*duration]
    return tstart+1e3*poisson


def single_poisson_process_STIM(Fstim, tstart, duration):
    # time in [ms], frequency in [Hz]
    poisson = np.cumsum(np.random.exponential(1./Fstim, size=int(3*duration/1e3*Fstim)))
    poisson = poisson[poisson<=1e-3*duration]
    return tstart+1e3*poisson

def stim_single_event_per_synapse(window, Nsyn, Nsyn_stim, tstart=0., seed=0):
    # time in [ms], frequency in [Hz]
    np.random.seed(seed)
    events = [[] for n in range(Nsyn)]
    for n in np.random.choice(np.arange(Nsyn), Nsyn_stim, replace=False):
        events[n].append(tstart+window*np.random.uniform())
    return events

def spike_train_BG_and_STIM(Fbg, Fstim,
                            tstop=1000,
                            tstart=200,
                            duration=20):
    # time in [ms], frequency in [Hz]
    sp_bg = single_poisson_process_BG(Fbg, tstop)
    sp_stim = single_poisson_process_STIM(Fstim, tstart, duration)
    return np.sort(np.concatenate([sp_bg, sp_stim])), sp_bg, sp_stim

def run_sim_with_bg_levels(args, seed=0):

    from model import Model

    if args.chelated:
        Model['alphaZn'] = 0
    else:
        Model['alphaZn'] = args.alphaZn

    tstop = len(args.bg_levels)*args.duration_per_bg_level
        
    t, neuron, SEGMENTS = initialize_sim(Model, tstop=tstop, active=args.active)

    if not (args.Nstim<=args.Nsyn):
        print('Nstim need to be lower than Nsyn, but %i>%i' % (args.Nstim,args.Nsyn), '--> Set to max')
        args.Nstim = args.Nsyn
        
    if not (args.syn_location<len(LOCs)):
        print('syn_location "%i" too high, only %i locations available\n ---> syn_location index set to 0' % (args.syn_location, len(LOCs)))        
        args.syn_location = 0
        
    synapses_loc = LOCs[args.syn_location] + np.arange(args.Nsyn)

    BG, STIM = [[] for i in range(len(synapses_loc))], [[] for i in range(len(synapses_loc))]

    stim = stim_single_event_per_synapse(args.stim_duration,
                                         len(synapses_loc), args.Nstim,
                                         tstart=args.stim_delay, seed=seed)

    for ibg, bg in enumerate(args.bg_levels):
        
        for i in range(len(synapses_loc)):
            BG[i] = BG[i]+list(single_poisson_process_BG(bg, args.duration_per_bg_level,
                                                         tstart=ibg*args.duration_per_bg_level))
            STIM[i] = STIM[i]+[s+ibg*args.duration_per_bg_level for s in stim[i]]
            
    spike_IDs, spike_times = np.empty(0, dtype=int), np.empty(0, dtype=float)
    for i in range(len(synapses_loc)):
        spike_times = np.concatenate([spike_times,
                                      np.concatenate([STIM[i],BG[i]])])
        spike_IDs = np.concatenate([spike_IDs,i*np.ones(len(STIM[i])+len(BG[i]))])
    isorted = np.argsort(spike_times)
    spike_times = spike_times[isorted]
    spike_IDs = spike_IDs[isorted]

    Estim, ES = ntwk.process_and_connect_event_stimulation(neuron,
                                                           spike_IDs, spike_times,
                                                           synapses_loc,
                                                           EXC_SYNAPSES_EQUATIONS.format(**Model),
                                                           ON_EXC_EVENT.format(**Model))
        
    # recording and running
    M = ntwk.StateMonitor(neuron, ('v'), record=[0, synapses_loc[0]])
    S = ntwk.StateMonitor(ES, ('X', 'gAMPA', 'gRiseNMDA', 'gDecayNMDA', 'bZn'), record=[0])

    # # Run simulation
    ntwk.run(tstop*ntwk.ms)
    
    output = {'t':np.array(M.t/ntwk.ms),
              'syn_locations':synapses_loc,
              'bg_levels':np.array(args.bg_levels),
              'Vm_soma':np.array(M.v/ntwk.mV)[0,:],
              'bZn_syn':np.array(S.bZn)[0,:],
              'gAMPA_syn':np.array(S.gAMPA/ntwk.nS)[0,:],
              'X_syn':np.array(S.X)[0,:],
              'Vm_syn':np.array(M.v/ntwk.mV)[1,:],
              'Model':Model, 'args':vars(args)}
    
    output['gNMDA_syn']= Model['qNMDA']*Model['nNMDA']*\
        (np.array(S.gDecayNMDA)[0,:]-np.array(S.gRiseNMDA)[0,:])\
        /(1+Model['etaMg']*Model['cMg']*np.exp(-output['Vm_syn']/Model['V0NMDA']))\
        *(1-Model['alphaZn']*output['bZn_syn'])

    fn = 'data-loc-%i' % args.syn_location
    if args.chelated:
        np.savez('data/bg-modul/%s-chelated-Zinc.npz' % fn, **output)
    else:
        np.savez('data/bg-modul/%s-free-Zinc.npz' % fn, **output)


if __name__=='__main__':
    
    LOCs = np.load('data/nmda-spike/locations.npy')
    loc_syn0 = LOCs[1]
    NSYNs=[2, 4, 6, 8, 10, 12]


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
        - plot
        """, default='plot')
    parser.add_argument("--stim_delay",help="[ms]", type=float, default=400)
    parser.add_argument("--duration_per_bg_level",help="[ms]", type=float, default=1000)
    parser.add_argument("--stim_duration",help="[ms]", type=float, default=20)
    # background props
    parser.add_argument("--bg_level",help="[ms]", type=float, default=2)
    parser.add_argument("--bg_levels",help="[ms]", type=float, default=[], nargs='*')
    # stim props
    parser.add_argument("-sl", "--syn_location",help="#", type=int, default=0)
    parser.add_argument("--syn_locations",help="#", type=int, default=[], nargs='*')
    parser.add_argument("--Nsyn",help="#", type=int, default=20)
    parser.add_argument("--Nstim",help="# < Nsyn", type=int, default=15)
    parser.add_argument("-c", "--chelated", help="chelated Zinc condition", action="store_true")
    parser.add_argument("--active", help="with active conductances", action="store_true")
    parser.add_argument("-aZn", "--alphaZn", help="inhibition factor in free Zinc condition",
                        type=float, default=.35)
    parser.add_argument("-s", "--seed",help="#", type=int, default=1)


    args = parser.parse_args()
    
    if len(args.bg_levels)==0:
        args.bg_levels =[args.bg_level]

    # # dealing with the available synaptic locations
    # if len(args.syn_locations)>0:
    #     syn_locations = []
    #     for s in args.syn_locations:
    #         if s<len(LOCs):
    #             syn_locations.append(s)
    #     args.syn_locations = syn_locations
    # if len(args.syn_locations)==0 and (args.syn_location<len(LOCs)):
    #     args.syn_locations =[args.syn_location]
    # elif len(args.syn_locations)==0:
    #     print('syn_location "%i" too high, only  %i locations available\n ---> syn_location index set to 0' % (args.syn_location, len(LOCs)))
    #     args.syn_locations =[0]

        
    if args.task=='run':
        run_sim_with_bg_levels(args, seed=0)
    else:
        fn = 'data-loc-%i' % args.syn_location
        if args.chelated:
            data = load_dict('data/bg-modul/%s-chelated-Zinc.npz' % fn)
        else:
            data = load_dict('data/bg-modul/%s-free-Zinc.npz' % fn)

        ge.plot(data['t'], data['Vm_soma'])
        ge.show()
        

