import sys, os
from model import Model
from single_cell_sim import *
from analyz.IO.npz import load_dict

def run_sim(Model,
            NSYNs=[3, 6, 9, 12, 15],
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
    S = ntwk.StateMonitor(ES, ('X', 'gAMPA', 'gRiseNMDA', 'gDecayNMDA', 'bZn'), record=[0])

    # # Run simulation
    ntwk.run(tstop*ntwk.ms)
    
    output = {'t':np.array(M.t/ntwk.ms),
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

    for key, color in zip(['AMPA-only', 'chelated-Zinc', 'free-Zinc'],
                          [ge.blue, 'k', ge.green]):
        data = load_dict('data/nmda-spike/demo-%s.npz' % key)
        AX[0].plot(data['t'], data['Vm_soma'], color=color, label=key)
        AX[1].plot(data['t'], data['gNMDA_syn'], color=color)
    ge.legend(AX[0], size='x-small')
    ge.set_plot(AX[0], ylabel='$V_m$ (mV)')
    ge.set_plot(AX[1], ylabel='conductance (nS)', xlabel='time (ms)')

    
    
if __name__=='__main__':
    
    if sys.argv[-1]=='syn-demo':

        from datavyz import ges as ge
        loc_syn0 = 1000
        NSYNs=[3, 6, 9, 12, 15]
        from datavyz import nrnvyz

        _, neuron, SEGMENTS = initialize_sim(Model)
        vis = nrnvyz(SEGMENTS, ge=ge)
        
        fig, AX = ge.figure(axes=(len(NSYNs),1),
                            figsize=(.8,1.2), wspace=0, left=0, top=0.3, bottom=0, right=0)
        for nsyn, ax in zip(NSYNs, AX):
            ge.title(ax, '$N_{syn}=%i$'%nsyn, size='xx-small')
            vis.plot_segments(SEGMENTS['comp_type']!='axon', bar_scale_args={}, ax=ax)
            vis.add_dots(ax, loc_syn0+np.arange(nsyn), 10, ge.orange)
        ge.show()
        
    elif sys.argv[-1]=='run-demo':
        
        data = run_sim(Model, ampa_only=True)
        np.savez('data/nmda-spike/demo-AMPA-only.npz', **data)
        data = run_sim(Model, ampa_only=False, chelated=True)
        np.savez('data/nmda-spike/demo-chelated-Zinc.npz', **data)
        data = run_sim(Model, ampa_only=False, chelated=False)
        np.savez('data/nmda-spike/demo-free-Zinc.npz', **data)
        
    elif sys.argv[-1]=='demo-plot':
        
        from datavyz import ges as ge
        demo_plot()
        ge.show()

    else:
        MIN_DISTANCE = 50e-6 # m
        from datavyz import ges as ge
        _, neuron, SEGMENTS = initialize_sim(Model)
        vis = nrnvyz(SEGMENTS, ge=ge)
        fig, ax = ge.figure(figsize=(1.,1.5), left=0, top=0.3, bottom=0, right=0)
        vis.plot_segments(SEGMENTS['comp_type']!='axon', bar_scale_args={}, ax=ax)

        soma = (SEGMENTS['name']=='soma')
        distance = np.sqrt((SEGMENTS['x']-SEGMENTS['x'][soma][0])**2+\
                           (SEGMENTS['y']-SEGMENTS['y'][soma][0])**2+\
                           (SEGMENTS['z']-SEGMENTS['z'][soma][0])**2)
        cond = (SEGMENTS['comp_type']=='dend') & (distance>MIN_DISTANCE)
        
        nsyn=15
        N=0
        for loc_syn0 in SEGMENTS['index'][cond][::50]:
            if np.min(distance[loc_syn0+np.arange(nsyn)])>MIN_DISTANCE:
                vis.add_dots(ax, loc_syn0+np.arange(nsyn), 10, ge.orange)
                N+=1
        print(N, 'segments')

        ge.show()
