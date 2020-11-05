import sys, os
from model import Model
from single_cell_sim import *
from analyz.IO.npz import load_dict

LOCs = np.load('data/nmda-spike/locations.npy')

def single_poisson_process_BG(Fbg, duration, tstart=0, seed=1):
    # time in [ms], frequency in [Hz]
    np.random.seed(seed)
    poisson = np.cumsum(np.random.exponential(1./Fbg, size=int(3*duration/1e3*Fbg)))
    poisson = poisson[poisson<=1e-3*duration]
    return tstart+1e3*poisson


def single_poisson_process_STIM(Fstim, tstart, duration, seed=0):
    # time in [ms], frequency in [Hz]
    np.random.seed(seed)
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
                            bg_seed=1,
                            stim_seed=1,
                            duration=20):
    # time in [ms], frequency in [Hz]
    sp_bg = single_poisson_process_BG(Fbg, tstop, seed=bg_seed)
    sp_stim = single_poisson_process_STIM(Fstim, tstart, duration, seed=stim_seed)
    return np.sort(np.concatenate([sp_bg, sp_stim])), sp_bg, sp_stim

def filename(args):
    if args.ampa_only:
        fn = os.path.join('data', 'bg-modul',
                          'data-loc-%i-seed-%i-ampa-active-%s.npz' % (args.syn_location, args.seed, args.active))
    elif args.bg_level>0:
        fn = os.path.join('data', 'bg-modul',
                          'data-bg-level-%.2f-loc-%i-seed-%i-alphaZn-%.2f-active-%s.npz' % (args.bg_level, args.syn_location, args.seed, args.alphaZn, args.active))
    else:
        fn = os.path.join('data', 'bg-modul',
                          'data-loc-%i-seed-%i-alphaZn-%.2f-active-%s.npz' % (args.syn_location, args.seed, args.alphaZn, args.active))
        
    return fn
    
def run_sim_with_bg_levels(args):

    from model import Model

    if args.ampa_only:
        Model['alphaZn'] = 0
        Model['qNMDA'] = 0.
    else:
        Model['alphaZn'] = args.alphaZn

    np.random.seed(args.seed)
    
    tstop = len(args.bg_levels)*len(args.NSTIMs)*len(args.stimSEEDS)*len(args.bgSEEDS)*args.duration_per_bg_level
        
    t, neuron, SEGMENTS = initialize_sim(Model, tstop=tstop, active=args.active)

    if not (args.syn_location<len(LOCs)):
        print('syn_location "%i" too high, only %i locations available\n ---> syn_location index set to 0' % (args.syn_location, len(LOCs)))
        args.syn_location = 0
        
    if not args.use_preloaded_presynact:
        
        synapses_loc = LOCs[args.syn_location]+np.arange(args.Nsyn)
        
        BG, STIM = [[] for i in range(len(synapses_loc))], [[] for i in range(len(synapses_loc))]

        for ibg, bg in enumerate(args.bg_levels):

            if args.verbose:
                print('bg_levels: ', ibg, bg, args.bg_levels)

            for ibgseed, bgseed in enumerate(args.bgSEEDS):

                if args.verbose:
                    print('bg_seed: ', ibgseed, bgseed, args.bgSEEDS)

                for istim, nstim in enumerate(args.NSTIMs):

                    if args.verbose:
                        print('stim: ', istim, nstim, args.NSTIMs)

                    for istimseed, stimseed in enumerate(args.stimSEEDS):

                        if args.verbose:
                            print('stim seed: ', istimseed, stimseed, args.stimSEEDS)

                        stim = stim_single_event_per_synapse(args.stim_duration,
                                                             len(synapses_loc), nstim,
                                                             tstart=args.stim_delay,
                                                             seed=5+stimseed+10*args.seed)

                        t0 = (ibg*len(args.NSTIMs)*len(args.stimSEEDS)*len(args.bgSEEDS)+\
                              ibgseed*len(args.NSTIMs)*len(args.stimSEEDS)+\
                              istim*len(args.stimSEEDS)+istimseed)*args.duration_per_bg_level

                        for i in range(len(synapses_loc)):
                            if bg>0:
                                bg_spikes = single_poisson_process_BG(bg, args.duration_per_bg_level,
                                                                      tstart=t0,
                                                                      seed=1+i+10*ibg+100*istim+1000*istimseed+10000*ibgseed+100000*args.seed)
                                BG[i] = BG[i]+list(bg_spikes)

                            STIM[i] = STIM[i]+[s+t0 for s in stim[i]]

        spike_IDs, spike_times = np.empty(0, dtype=int), np.empty(0, dtype=float)
        for i in range(len(synapses_loc)):
            spike_times = np.concatenate([spike_times,
                                          np.concatenate([STIM[i],BG[i]])])
            spike_IDs = np.concatenate([spike_IDs,i*np.ones(len(STIM[i])+len(BG[i]))])
        isorted = np.argsort(spike_times)
        spike_times = spike_times[isorted]
        spike_IDs = spike_IDs[isorted]

        spike_IDs, spike_times = ntwk.deal_with_multiple_spikes_per_bin(spike_IDs, spike_times, t,
                                                                        verbose=True)
        if args.save_presynaptic_input:
            np.save(os.path.join('data', 'bg-modul', 'presyn-data.npy'),
                    {'spike_IDs':spike_IDs,
                     'BG':BG, 'STIM':STIM,
                     'synapses_loc':synapses_loc,
                     't':t, 'tstop':tstop,
                     'spike_times':spike_times})
    else:
        pre = np.load(os.path.join('data', 'bg-modul', 'presyn-data.npy'), allow_pickle=True).item()
        spike_IDs, spike_times = pre['spike_IDs'], pre['spike_times']
        synapses_loc = pre['synapses_loc']
        t, tstop = pre['t'], pre['tstop']
        # BG, STIM = pre['BG'], pre['STIM']
    
    Estim, ES = ntwk.process_and_connect_event_stimulation(neuron,
                                                           spike_IDs, spike_times,
                                                           synapses_loc,
                                                           EXC_SYNAPSES_EQUATIONS.format(**Model),
                                                           ON_EXC_EVENT.format(**Model))
        
    # recording and running
    M = ntwk.StateMonitor(neuron, ('v'), record=[0, synapses_loc[0]])
    S = ntwk.StateMonitor(ES, ('X', 'gAMPA', 'gRiseNMDA', 'gDecayNMDA', 'bZn'), record=[0])

    # # Run simulation
    print('running simulation [...]', t[-1])

    if args.active:
        ntwk.defaultclock.dt = 0.01*ntwk.ms

    ntwk.run(tstop*ntwk.ms)
    
    output = {'t':np.array(M.t/ntwk.ms),
              # 'BG_raster':BG,
              # 'STIM_raster':STIM,
              'syn_locations':synapses_loc,
              'bg_levels':np.array(args.bg_levels),
              'Vm_soma':np.array(M.v/ntwk.mV)[0,:],
              # 'bZn_syn':np.array(S.bZn)[0,:], # REMOVED TO DECREASE FILE SIZE !!
              # 'gAMPA_syn':np.array(S.gAMPA/ntwk.nS)[0,:],
              # 'X_syn':np.array(S.X)[0,:],
              # 'Vm_syn':np.array(M.v/ntwk.mV)[1,:],
              'Model':Model, 'args':vars(args)}

    # REMOVED TO DECREASE FILE SIZE !!
    # output['gNMDA_syn']= Model['qNMDA']*Model['nNMDA']*\
    #     (np.array(S.gDecayNMDA)[0,:]-np.array(S.gRiseNMDA)[0,:])\
    #     /(1+Model['etaMg']*Model['cMg']*np.exp(-output['Vm_syn']/Model['V0NMDA']))\
    #     *(1-Model['alphaZn']*output['bZn_syn'])

    np.savez(filename(args), **output)
    

def analyze_sim(data, data2):

    args = data['args']
    print(args)
    fig, AX = ge.figure(axes=(len(args['bg_levels']),2), wspace=0.1)
    fig2, AX2 = ge.figure(axes=(1,len(args['bg_levels'])), figsize=(1.,.7), hspace=0.1, bottom=1.5)
    fig.suptitle('n=%i bg seeds, n=%i stim seeds, loc #%i' % (len(args['bgSEEDS']), len(args['stimSEEDS']), args['syn_location']), size=10)

    tcond = (data['t']>=0) & (data['t']<args['duration_per_bg_level'])
    output = {'t':data['t'][tcond]-args['stim_delay']}
    output['free'] = np.zeros((len(args['bg_levels']), len(args['bgSEEDS']), len(args['NSTIMs']), len(args['stimSEEDS']), len(output['t'])))
    output['chelated'] = np.zeros((len(args['bg_levels']), len(args['bgSEEDS']), len(args['NSTIMs']), len(args['stimSEEDS']), len(output['t'])))
    
    for ibg, bg in enumerate(args['bg_levels']):

        for ibgseed, bgseed in enumerate(args['bgSEEDS']):
            
            for istim, nstim in enumerate(args['NSTIMs']):
                
                for istimseed, stimseed in enumerate(args['stimSEEDS']):
                    t0 = (ibg*len(args['NSTIMs'])*len(args['stimSEEDS'])*len(args['bgSEEDS'])+\
                          ibgseed*len(args['NSTIMs'])*len(args['stimSEEDS'])+\
                          istim*len(args['stimSEEDS'])+istimseed)*args['duration_per_bg_level']
                    
                    tcond = (data['t']>=t0) & (data['t']<t0+args['duration_per_bg_level'])
                    output['free'][ibg,ibgseed,istim,istimseed,:] = data['Vm_soma'][tcond]
                    output['chelated'][ibg,ibgseed,istim,istimseed,:] = data2['Vm_soma'][tcond]

    ylim, ylim2 = [np.inf, -np.inf], [np.inf, -np.inf]
    for ibg, bg in enumerate(args['bg_levels']):
        for istim, nstim in zip(range(len(args['NSTIMs']))[::-1] ,args['NSTIMs'][::-1]):
            # raw responses
            y0 = output['free'][ibg,:,istim,:,:].mean(axis=(0,1))
            y1 = output['chelated'][ibg,:,istim,:,:].mean(axis=(0,1))
            AX[0][ibg].plot(output['t'], y0,lw=1,color=ge.red_to_blue(istim/len(args['NSTIMs'])))
            AX[1][ibg].plot(output['t'], y1,lw=1,color=ge.red_to_blue(istim/len(args['NSTIMs'])))
            ylim = [min([ylim[0],y1.min(),y0.min()]), max([ylim[0],y1.max(),y0.max()])]
            # max responses
        y0 = output['free'][ibg,:,:,:,:].max(axis=(0,2,3))
        y1 = output['chelated'][ibg,:,:,:,:].max(axis=(0,2,3))
        AX2[ibg].plot(args['NSTIMs'], y0,lw=1,color='k')
        AX2[ibg].plot(args['NSTIMs'], y1,lw=1,color=ge.green)
        ylim2 = [min([ylim2[0],y1.min(),y0.min()]), max([ylim2[0],y1.max(),y0.max()])]
        ge.annotate(AX[0][ibg],'%.1fHz'%bg,(1,1),color=ge.purple,ha='right',va='top')
        ge.annotate(AX2[ibg],'%.1fHz'%bg,(0,1),color=ge.purple,va='top')
        
    for ibg, bg in enumerate(args['bg_levels']):
        if ibg==0:
            ge.set_plot(AX[0][ibg], ['left'], ylim=ylim, xticks=[0,500])
            ge.set_plot(AX[1][ibg], ylim=ylim, xticks=[0,500], xlabel='time (ms)')
        else:
            ge.set_plot(AX[0][ibg], [], ylim=ylim, xticks=[0,500], xticks_labels=[])
            ge.set_plot(AX[1][ibg], ['bottom'], ylim=ylim, xticks=[0,500], xlabel='time (ms)')
        ge.set_plot(AX2[ibg], ['left'], ylim=ylim2, ylabel='peak (mV)')
    ge.set_plot(AX2[ibg], ylim=ylim2, ylabel='peak (mV)', xlabel='syn. #')
    ge.annotate(AX[0][0], 'free-Zinc', (0,1), bold=True, size='large')
    ge.annotate(AX[1][0], 'chelated-Zinc', (0,1), bold=True, color=ge.green, size='large')
    
                
def plot_sim(data, data2):
    """
    for demo data only not thought to handle different seeds
    """
    args = data['args']

    AE = []
    for i in range(len(args['bg_levels'])):
        AE.append([[1,4]])
        AE.append([[1,2]])
        AE.append([[1,1]])
    
    fig, AX = ge.figure(axes_extents=AE, figsize=(3.5,.25), wspace=0., left=.2)
    
    for ibg, bg in enumerate(args['bg_levels']):

        t0 = ibg*len(args['NSTIMs'])*len(args['stimSEEDS'])*len(args['bgSEEDS'])*args['duration_per_bg_level']
        t1 = (ibg+1)*len(args['NSTIMs'])*len(args['stimSEEDS'])*len(args['bgSEEDS'])*args['duration_per_bg_level']

        tcond = (data['t']>=t0) & (data['t']<t1)
        
        AX[3*ibg].plot(data2['t'][tcond], data2['Vm_soma'][tcond], color=ge.green, label='chelated-Zinc', lw=1.5)
        AX[3*ibg].plot(data['t'][tcond], data['Vm_soma'][tcond], color='k', label='free-Zinc', lw=1.5)
        AX[3*ibg].plot([t0,t1], [-75,-75], 'k--', lw=0.5)
        
        for i, sp0 in enumerate(data['BG_raster']):
            sp = np.array(sp0)
            cond = (sp>=t0) & (sp<t1)
            AX[3*ibg+1].scatter(sp[cond], i*np.ones(len(sp[cond])), color=ge.purple, s=2)
        for i, sp0 in enumerate(data['STIM_raster']):
            sp = np.array(sp0)
            cond = (sp>=t0) & (sp<t1)
            AX[3*ibg+1].scatter(sp[cond], i*np.ones(len(sp[cond])), color=ge.orange, s=4)
            
        ge.annotate(AX[3*ibg+1], '$\\nu_{bg}$=%.1fHz' % bg, (0,0), color=ge.purple, rotation=90, ha='right')
        ge.set_plot(AX[3*ibg], [], xlim=[t0, t1])#, ylabel='Vm (mV)')
        ge.draw_bar_scales(AX[3*ibg], Ybar_label_format="%.1fmV", Xbar_label_format="%.0fms",
                           Ybar_fraction=.25, Xbar_fraction=.05, loc=(0.005,.99), orientation='right-bottom')
        ge.annotate(AX[3*ibg], ' -75mV', (t1,-75), xycoords='data', va='center')
        ge.set_plot(AX[3*ibg+1], [], xlim=[t0, t1])#, ylabel='synapse ID')
        ge.set_plot(AX[3*ibg+2], [], xlim=[t0, t1])#, ylabel='synapse ID')
        # AX[3*ibg+2].axis('off')

    y1 = AX[1].get_ylim()[1]
    for istim, nstim in enumerate(args['NSTIMs']):
        ge.annotate(AX[1], ' %i syn.' % nstim,
                    (args['stim_delay']+istim*args['duration_per_bg_level']*len(args['stimSEEDS']),y1/2),
                    xycoords='data', color=ge.orange, va='center', ha='right', rotation=90)
        
    

if __name__=='__main__':
    
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
        - full
        - run
        - plot
        """, default='plot')
    parser.add_argument("--stim_delay",help="[ms]", type=float, default=600)
    parser.add_argument('-dbl', "--duration_per_bg_level",help="[ms]", type=float, default=2000)
    parser.add_argument("--stim_duration",help="[ms]", type=float, default=20)
    # background props
    parser.add_argument("--bg_level",help="[Hz]", type=float, default=-1)
    parser.add_argument("--bg_levels",help="[Hz]", type=float, default=[0, 1, 2, 3, 4, 5, 6], nargs='*')
    parser.add_argument("--NbgSEEDS",help="#", type=int, default=1)
    parser.add_argument("--bgSEEDS",help="#", type=int, default=[0], nargs='*')
    # stim props
    parser.add_argument("-sl", "--syn_location",help="#", type=int, default=1)
    parser.add_argument("-Nsl", "--N_syn_location",help="#", type=int, default=5)
    parser.add_argument("--Nsyn",help="#", type=int, default=20)
    parser.add_argument("--NSTIMs",help="# < Nsyn", type=int,
                        default=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18], nargs='*')
    # parser.add_argument("--NSTIMs",help="# < Nsyn", type=int,
    #                     default=range(20), nargs='*')
    parser.add_argument("--stimSEEDS",help="#", type=int, default=[0], nargs='*')
    # loop over locations
    parser.add_argument("--syn_locations",help="#", type=int, default=[], nargs='*')
    # model variations
    # parser.add_argument("-c", "--chelated",
    #                     help="chelated Zinc condition",
    #                     default='False')
    parser.add_argument("--use_preloaded_presynact", action="store_true")
    parser.add_argument("--save_presynaptic_input", action="store_true")
    parser.add_argument("--active",
                        help="with active conductances",
                        action="store_true")
    parser.add_argument("--ampa_only",
                        help="ampa only",
                        action="store_true")
    parser.add_argument("-aZn", "--alphaZn",
                        help="inhibition factor in free Zinc condition",
                        type=float, default=.4)
    parser.add_argument("-s", "--seed", help="#", type=int, default=10)
    parser.add_argument('-v', "--verbose", action="store_true")

    args = parser.parse_args()

    # if args.chelated:
    #     args.alphaZn = 0

    if args.NbgSEEDS>1:
        args.bgSEEDS = np.arange(args.NbgSEEDS)
        
    if args.bg_level>=0:
        args.bg_levels =[args.bg_level]

    if args.task=='run':
        run_sim_with_bg_levels(args)
            
    elif args.task=='analyze':
        # args.chelated  = False
        data = load_dict(filename(args))
        # args.chelated  = True
        # data2 = load_dict(filename(args))
        analyze_sim(data, data)
        ge.show()

    elif args.task=='full':
        for args.alphaZn in [0., 0.3, 0.45]:
            run_sim_with_bg_levels(args)
        args.ampa_only = True
        run_sim_with_bg_levels(args, seed=args.seed)
                
                
    elif args.task=='plot':
        data = load_dict(filename(args))
        plot_sim(data, data)
        ge.show()
        
    elif args.task=='active-demo':

        from model import Model

        ntwk.defaultclock.dt = 0.025*ntwk.ms

        active, chelated = True, True
        t, neuron, SEGMENTS = initialize_sim(Model, active=active)

        tstop = args.duration_per_bg_level
        synapses_loc = LOCs[args.syn_location]+np.arange(args.Nsyn)

        stim = stim_single_event_per_synapse(args.stim_duration,
                                             len(synapses_loc), len(synapses_loc),
                                             tstart=args.stim_delay,
                                             seed=args.seed)
        
        BG, STIM = [[] for i in range(len(synapses_loc))], [[] for i in range(len(synapses_loc))]

        for i in range(len(synapses_loc)):
            BG.append([])
            if args.bg_level>0:
                bg_spikes = single_poisson_process_BG(args.bg_level, args.duration_per_bg_level,
                                                      tstart=0,
                                                      seed=args.seed+2*i)
                BG[i] = BG[i]+list(bg_spikes)

            STIM[i] = STIM[i]+[s for s in stim[i]]

        spike_IDs, spike_times = np.empty(0, dtype=int), np.empty(0, dtype=float)
        for i in range(len(synapses_loc)):
            spike_times = np.concatenate([spike_times,
                                          np.concatenate([STIM[i],BG[i]])])
            spike_IDs = np.concatenate([spike_IDs,i*np.ones(len(STIM[i])+len(BG[i]))])

        isorted = np.argsort(spike_times)
        spike_times = spike_times[isorted]
        spike_IDs = spike_IDs[isorted]

        # spike_IDs, spike_times = ntwk.deal_with_multiple_spikes_per_bin(spike_IDs, spike_times, t,
        #                                                                 verbose=True)

        Estim, ES = ntwk.process_and_connect_event_stimulation(neuron,
                                                               spike_IDs, spike_times,
                                                               synapses_loc,
                                                               EXC_SYNAPSES_EQUATIONS.format(**Model),
                                                               ON_EXC_EVENT.format(**Model))


        # recording and running
        if active:
            M = ntwk.StateMonitor(neuron, ('v', 'InternalCalcium'), record=[0, synapses_loc[0]])
        else:
            M = ntwk.StateMonitor(neuron, ('v'), record=[0, synapses_loc[0]])
            S = ntwk.StateMonitor(ES, ('X', 'gAMPA', 'gE_post', 'bZn'), record=[0])

        # # Run simulation
        print('running simulation [...]')
        ntwk.run(tstop*ntwk.ms)

        from datavyz import ges as ge
        fig, AX = ge.figure(axes=(1,2),figsize=(2,1))

        AX[0].plot(np.array(M.t/ntwk.ms), np.array(M.v/ntwk.mV)[0,:], label='soma')
        ge.plot(np.array(M.t/ntwk.ms), np.array(M.v/ntwk.mV)[1,:], label='dend', ax=AX[0])
        if active:
            ge.plot(np.array(M.t/ntwk.ms), np.array(M.InternalCalcium/ntwk.nM)[1,:],
                    label='dend', ax=AX[1], axes_args={'ylabel':'[Ca2+] (nM)', 'xlabel':'time (ms)'})
        else:
            AX[1].plot(np.array(M.t/ntwk.ms), np.array(S.gAMPA/ntwk.nS)[0,:], ':', color=ge.orange, label='gAMPA')
            AX[1].plot(np.array(M.t/ntwk.ms), np.array(S.gE_post/ntwk.nS)[0,:]-np.array(S.gAMPA/ntwk.nS)[0,:], color=ge.orange, label='gNMDA')

        ge.legend(AX[0])
        ge.legend(AX[1])
        ge.show()


    else:
        pass
