import numpy as np
import sys, argparse, os

from neural_network_dynamics import main as ntwk
from model import Model

morpho = ntwk.Morphology.from_swc_file(Model['morpho_file_1'])
SEGMENTS = ntwk.morpho_analysis.compute_segments(morpho)

Nsyn = 4
PROX = 220+np.arange(Nsyn)
DIST = 400+np.arange(Nsyn)

N_PULSES = np.arange(1, 16, 2)
DELAYS = np.arange(1, 9)

from datavyz import ges as ge

# in case not used as a modulus
if __name__=='__main__':
    
    import argparse
    # First a nice documentation 
    parser=argparse.ArgumentParser(description=
     """ 
     Reproducing Branco et al. (2011)
     """
    ,formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("-p", "--protocol", help="""
    either:
    - 'morpho'
    - 'syn-number'
    - 'syn-timing'
    """, default='morpho')
    parser.add_argument("-cZn", "--chelated_zinc",  action="store_true")
    parser.add_argument("-a", "--active",  action="store_true")
    parser.add_argument("--plot",  action="store_true")
    args = parser.parse_args()

    if args.chelated_zinc:
        Model['Deltax0'] = 0. # removing Zinc modulation !!
        suffix = 'chelatedZn'
    else:
        suffix = 'freeZn'
    if args.active:
        suffix += '-active'

    tstop, t0_stim = 300, 20
    
    from analyz.IO.npz import load_dict
    
    if args.protocol=='morpho':
    
        from datavyz import nrnvyz
        vis = nrnvyz(SEGMENTS, ge=ge)
        fig1, ax= vis.plot_segments(SEGMENTS['comp_type']!='axon', bar_scale_args={},
                                fig_args=dict(figsize=(1.5,2.4), left=0, bottom=0, top=0, right=0))
        vis.add_dots(ax, PROX, 50, ge.orange)
        vis.add_dots(ax, DIST, 50, ge.orange)
        ge.show()

    elif args.protocol=='syn-number':

        tstop, t0_stim = 300, 20
        if args.plot:
            data = load_dict(os.path.join('data', 'branco-syn-number-%s.npz' % suffix))
            
            fig, AX = ge.figure(axes=(1,2), figsize=(.7,1), right=4., hspace=0.1, bottom=0.1)
            ge.title(AX[0], '%i to %i synapses (by 2)' % (N_PULSES[0], N_PULSES[-1]))
            for synapses_loc, label, ax in zip([PROX,DIST], ['prox','dist'], AX):
                for i in range(len(data['v-%s' % label])):
                    ax.plot(data['t'], data['v-%s' % label][i], label=label, color='k', lw=1)
            
                ge.set_plot(ax, [], ylim=[-76, -45], xlim=[0, tstop])
                ge.draw_bar_scales(ax,
                                   Xbar=50., Xbar_label='50ms',
                                   Ybar=5., Ybar_label='5mV',
                                   loc='top-right')
            ge.show()
            
        else:
                
            from single_cell_sim import *
            if args.chelated_zinc:
                Model['Deltax0'] = 0. # removing Zinc modulation !!
            
            data = {'t':[], 'v-prox':[], 'v-dist':[]}
            for synapses_loc, label in zip([PROX,DIST], ['prox','dist']):
                for n_pulses in N_PULSES:
                    t, neuron, SEGMENTS = initialize_sim(Model, active=args.active)
                    spike_times = t0_stim + np.arange(n_pulses)*0.1
                    spike_IDs = np.arange(n_pulses)%Nsyn
                    Estim, ES = ntwk.process_and_connect_event_stimulation(neuron,
                                                                           spike_IDs, spike_times,
                                                                           synapses_loc,
                                                                           EXC_SYNAPSES_EQUATIONS.format(**Model),
                                                                           ON_EXC_EVENT.format(**Model))

                    # recording and running
                    M = ntwk.StateMonitor(neuron, ('v'), record=[0, synapses_loc[0]])

                    # # Run simulation
                    ntwk.run(tstop*ntwk.ms)
                    data['v-%s' % label].append(np.array(M.v/ntwk.mV)[0,:])
            data['t'] = np.array(M.t/ntwk.ms)
            np.savez(os.path.join('data', 'branco-syn-number-%s.npz' % suffix), **data)
    
    elif args.protocol=='syn-timing':

        n_pulses = 15
        
        if args.plot:
            data = load_dict(os.path.join('data', 'branco-syn-timing-%s.npz' % suffix))
            
            fig, AX = ge.figure(axes=(1,2), figsize=(.7,1), right=4., hspace=0.1, bottom=0.1)
            ge.title(AX[0], '%i to %ims interval' % (DELAYS[0], DELAYS[-1]))
            for synapses_loc, label, ax in zip([PROX,DIST], ['prox','dist'], AX):
                for i in range(len(data['v-%s' % label])):
                    ax.plot(data['t'], data['v-%s' % label][i], label=label, color='k', lw=1)
            
                ge.set_plot(ax, [], ylim=[-76, -45], xlim=[0, tstop])
                ge.draw_bar_scales(ax,
                                   Xbar=50., Xbar_label='50ms',
                                   Ybar=5., Ybar_label='5mV',
                                   loc='top-right')

            
            ge.set_plot(ax, [], ylim=[-76, -45], xlim=[0, tstop])
            ge.draw_bar_scales(ax,
                               Xbar=50., Xbar_label='50ms',
                               Ybar=5., Ybar_label='5mV',
                               loc='top-right')
            ge.show()
            
        else:
            
            from single_cell_sim import *
            if args.chelated_zinc:
                Model['Deltax0'] = 0. # removing Zinc modulation !!
                
            data = {'t':[], 'v-prox':[], 'v-dist':[]}
            
            for synapses_loc, label in zip([PROX,DIST], ['prox','dist']):
                for delay in DELAYS:

                    t, neuron, SEGMENTS = initialize_sim(Model, active=args.active)
                    spike_times = t0_stim + 10*delay + np.arange(1, n_pulses+1)*delay
                    spike_IDs = np.arange(n_pulses)%Nsyn

                    
                    Estim, ES = ntwk.process_and_connect_event_stimulation(neuron,
                                                                           spike_IDs, spike_times,
                                                                           synapses_loc,
                                                                           EXC_SYNAPSES_EQUATIONS.format(**Model),
                                                                           ON_EXC_EVENT.format(**Model))

                    # recording and running
                    M = ntwk.StateMonitor(neuron, ('v'), record=[0, synapses_loc[0]])

                    # # Run simulation
                    ntwk.run(tstop*ntwk.ms)
                    data['v-%s' % label].append(np.array(M.v/ntwk.mV)[0,:])
                    
            data['t'] = np.array(M.t/ntwk.ms)
            np.savez(os.path.join('data', 'branco-syn-timing-%s.npz' % suffix), **data)
            

    else:

        pass
        # fig, AXboth = ge.figure(axes=(2,2), figsize=(.7,1), right=4., hspace=0.1, bottom=0.1)
        # for AX, color, suffix in zip([[AXboth[0][0], AXboth[1][0]], [AXboth[0][1], AXboth[1][1]]], [ge.green, 'k'], ['chelatedZn', 'freeZn']):
        #     data = load_dict(os.path.join('data', 'branco-syn-timing-%s.npz' % suffix))
        #     ge.title(AX[0], suffix.replace('Zn', ' Zinc'), color=color)
        #     for synapses_loc, label, ax in zip([PROX,DIST], ['prox','dist'], AX):
        #         for i in range(len(data['v-%s' % label])):
        #             ax.plot(data['t'], data['v-%s' % label][i], label=label, color=color, lw=1)
        #         ge.set_plot(ax, [], ylim=[-76, -45], xlim=[0, tstop])

        #     ge.draw_bar_scales(AX[1],
        #                        Xbar=50., Xbar_label='50ms',
        #                        Ybar=5., Ybar_label='5mV',
        #                        loc='top-right')
        # ge.show()
