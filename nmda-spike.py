import sys, os
from model import Model
from single_cell_sim import *
from analyz.IO.npz import load_dict

def run_sim(Model,
            NSYNs=[3, 6, 9, 12, 15],
            syn_loc0 = 124,
            t0=100, interstim=400,
            freq_stim=50, n_repeat=3,
            active=False,
            ampa_only=False,
            chelated=False):


    Modelc = Model.copy()
    tstop = t0+len(NSYNs)*interstim
    if chelated:
        Modelc['alphaZn'] = 0.

    t, neuron, SEGMENTS = initialize_sim(Model,
                                         active=active,
                                         tstop=tstop)
    
    spike_IDs, spike_times = np.empty(0, dtype=int), np.empty(0, dtype=float)

    start, events = t0, []
    for n in NSYNs:
        
        synapses_loc = syn_loc0+np.arange(n)
        events.append(start)
        
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
    S = ntwk.StateMonitor(ES, ('X', 'gAMPA', 'gRiseNMDA', 'gDecayNMDA', 'bZn'), record=[0])

    # # Run simulation
    ntwk.run(tstop*ntwk.ms)
    
    output = {'t':np.array(M.t/ntwk.ms), 'NSYNs':np.array(NSYNs), 'events':np.array(events),
              'Vm_soma':np.array(M.v/ntwk.mV)[0,:],
              'bZn_syn':np.array(S.bZn)[0,:],
              'gAMPA_syn':np.array(S.gAMPA/ntwk.nS)[0,:],
              'X_syn':np.array(S.X)[0,:],
              'Vm_syn':np.array(M.v/ntwk.mV)[1,:],
              'Model':Modelc}
    output['gNMDA_syn']= Model['qNMDA']*Model['nNMDA']*\
        (np.array(S.gDecayNMDA)[0,:]-np.array(S.gRiseNMDA)[0,:])\
        /(1+Model['etaMg']*Model['cMg']*np.exp(-output['Vm_syn']/Model['V0NMDA']))\
        *(1-Model['alphaZn']*output['bZn_syn'])
    
    return output


def demo_plot():
    
    fig, AX = ge.figure(axes=(1,2), figsize=(2,1))

    for key, color, l in zip(['AMPA-only', 'chelated-Zinc', 'free-Zinc'],
                             [ge.blue, 'k', ge.green],
                             ['$g_{AMPA}$', '$g_{NMDA}$ (chelated)', '$g_{NMDA}$ (free)']):
        data = load_dict('data/nmda-spike/demo-%s.npz' % key)
        AX[0].plot(data['t'], data['Vm_soma'], color=color, label=key)
        AX[1].plot(data['t'], data['gNMDA_syn'], color=color, label=l+'  ')

    for e, n in zip(data['events'], data['NSYNs']):
        ge.annotate(AX[1], ' $N_{syn}$=%i' % n, (e, 0),
                    xycoords='data', rotation=90, ha='right', size='xx-small')
        
    ge.legend(AX[0], size='x-small')
    ge.legend(AX[1], size='xx-small', ncol=3)
    ge.set_plot(AX[0], ylabel='$V_m$ (mV)')
    ge.set_plot(AX[1], ylabel='$g$ (nS)', xlabel='time (ms)')


def fig_full():
    pass
    
if __name__=='__main__':

    try:
        LOCs = np.load('data/nmda-spike/locations.npy')
        loc_syn0 = LOCs[1]
    except BaseException:
        pass
    NSYNs=[2, 4, 6, 8, 10, 12, 14]

    if sys.argv[-1]=='syn-demo':

        from datavyz import nrnvyz, ges as ge
        
        _, neuron, SEGMENTS = initialize_sim(Model)
        vis = nrnvyz(SEGMENTS, ge=ge)
        
        fig, AX = ge.figure(axes=(len(NSYNs),1),
                            figsize=(.8,1.2), wspace=0, left=0, top=0.3, bottom=0, right=0)
        for nsyn, ax in zip(NSYNs, AX):
            ge.title(ax, '$N_{syn}$=%i' % nsyn, size='xx-small')
            vis.plot_segments(SEGMENTS['comp_type']!='axon',
                              bar_scale_args=dict(Ybar=50, Xbar_label='50um', Xbar=50, Ybar_label='',
                                                  loc='top-right', xyLoc=(-110,90), size='xx-small'),
                              ax=ax)
            vis.add_dots(ax, loc_syn0+np.arange(nsyn), 10, ge.orange)
        ge.show()
        
    elif sys.argv[-1]=='run-demo':

        syn_loc0 = 9
        NSYNs=[2, 4, 6, 8, 10]
        data = run_sim(Model, ampa_only=True, NSYNs=NSYNs, syn_loc0=LOCs[syn_loc0])
        np.save('data/nmda-spike/demo-AMPA-only.npy', data)
        data = run_sim(Model, ampa_only=False, chelated=True, NSYNs=NSYNs, syn_loc0=LOCs[syn_loc0])
        np.save('data/nmda-spike/demo-chelated-Zinc.npy', data)
        data = run_sim(Model, ampa_only=False, chelated=False, NSYNs=NSYNs, syn_loc0=LOCs[syn_loc0])
        np.save('data/nmda-spike/demo-free-Zinc.npy', data)

    elif sys.argv[-1]=='run-full':
        
        NSYNs=np.arange(1, 15)
        for loc in LOCs:
            data = run_sim(Model, ampa_only=True, NSYNs=NSYNs, syn_loc0=loc)
            np.save('data/nmda-spike/data-loc-%i-AMPA-only.npy' % loc, data)
            data = run_sim(Model, ampa_only=False, chelated=True, NSYNs=NSYNs, syn_loc0=loc)
            np.save('data/nmda-spike/data-loc-%i-chelated-Zinc.npy' % loc, data)
            data = run_sim(Model, ampa_only=False, chelated=False, NSYNs=NSYNs, syn_loc0=loc)
            np.save('data/nmda-spike/data-loc-%i-free-Zinc.npy' % loc, data)

    elif sys.argv[-1]=='fig-full':

        fig_full()
            
    elif sys.argv[-1]=='demo-plot':
        
        from datavyz import ges as ge
        demo_plot()
        ge.show()

    elif sys.argv[-1]=='locations':
        
        MIN_DISTANCE = 50e-6 # m
        from datavyz import ges as ge
        _, neuron, SEGMENTS = initialize_sim(Model)
        vis = nrnvyz(SEGMENTS, ge=ge)
        fig, ax = ge.figure(figsize=(.8,1.1), left=0, top=0.3, bottom=0, right=0)
        vis.plot_segments(SEGMENTS['comp_type']!='axon',
                          # bar_scale_args=dict(Ybar=50, Xbar_label='50um', Xbar=50, Ybar_label='', loc='bottom-left', size='xx-small'),
                          bar_scale_args=dict(Ybar=50, Xbar_label='50um', Xbar=50, Ybar_label='',
                                              loc='top-right', xyLoc=(-110,90), size='xxx-small'),
                          ax=ax)

        soma = (SEGMENTS['name']=='soma')
        distance = np.sqrt((SEGMENTS['x']-SEGMENTS['x'][soma][0])**2+\
                           (SEGMENTS['y']-SEGMENTS['y'][soma][0])**2+\
                           (SEGMENTS['z']-SEGMENTS['z'][soma][0])**2)
        cond = (SEGMENTS['comp_type']=='dend') & (distance>MIN_DISTANCE)
        
        nsyn=30

        LOCs =[]
        for loc_syn0 in SEGMENTS['index'][cond][::50]:
            comp_name = SEGMENTS['name'][loc_syn0]
            if (np.sum(cond[loc_syn0+np.arange(nsyn)])==nsyn) &\
               len(np.unique(SEGMENTS['name'][loc_syn0+np.arange(nsyn)]))==1:
                # meaning all points are on the same branch
                LOCs.append(loc_syn0)
                
        for i, loc_syn0 in enumerate(LOCs):
            vis.add_dots(ax, loc_syn0+np.arange(nsyn), 10, color=ge.tab20(i%20))

        ge.show()
        print(len(LOCs), 'segments')
        np.save('data/nmda-spike/locations.npy', np.array(LOCs))
        fig.savefig(os.path.join('figures','all-syn-locs.svg'))
    else:
        print("""
        Should be used as either:
        - python nmda-spikes.py syn-demo
        - python nmda-spikes.py run-demo
        - python nmda-spikes.py demo-plot
        - python nmda-spikes.py locations
        """)
