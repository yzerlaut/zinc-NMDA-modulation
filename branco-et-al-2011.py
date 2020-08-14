import numpy as np
import sys

from neural_network_dynamics import main as ntwk
from model import Model
Model['alphaZn'] = 0. # removing Zinc modulation !!


morpho = ntwk.Morphology.from_swc_file(Model['morpho_file_1'])
SEGMENTS = ntwk.morpho_analysis.compute_segments(morpho)

Nsyn = 4
PROX = 210+np.arange(Nsyn)
DIST = 400+np.arange(Nsyn)

N_PULSES = np.arange(1, 16, 2)
DELAYS = np.arange(1, 9)

from datavyz import ges as ge

if sys.argv[-1]=='morpho':
    
    from datavyz import nrnvyz
    vis = nrnvyz(SEGMENTS, ge=ge)
    fig1, ax= vis.plot_segments(SEGMENTS['comp_type']!='axon', bar_scale_args={},
                                fig_args=dict(figsize=(1.5,2.4), left=0, bottom=0, top=0, right=0))
    vis.add_dots(ax, PROX, 50, ge.orange)
    vis.add_dots(ax, DIST, 50, ge.orange)
    ge.show()

elif sys.argv[-1]=='syn-number':

    from single_cell_sim import *
    
    fig, AX = ge.figure(axes=(1,2), figsize=(.7,1), right=4., hspace=0.1, bottom=0.1)

    tstop, t0_stim = 300, 20

    ge.title(AX[0], '%i to %i synapses (by 2)' % (N_PULSES[0], N_PULSES[-1]))
    
    for synapses_loc, label, ax in zip([PROX,DIST], ['prox','dist'], AX):
        for n_pulses in N_PULSES:
            t, neuron, SEGMENTS = initialize_sim(Model)
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

            ax.plot(np.array(M.t/ntwk.ms), np.array(M.v/ntwk.mV)[0,:], label=label, color='k', lw=1)
        ge.set_plot(ax, [], ylim=[-76, -45], xlim=[0, tstop])
        ge.draw_bar_scales(ax,
                           Xbar=50., Xbar_label='50ms',
                           Ybar=5., Ybar_label='5mV',
                           loc='top-right')
    ge.show()
    
    
elif sys.argv[-1]=='syn-timing':

    from single_cell_sim import *
    
    fig, AX = ge.figure(axes=(1,2), figsize=(.7,1), right=4., hspace=0.1, bottom=0.1)

    tstop, t0_stim = 300, 20

    ge.title(AX[0], '%i to %ims interval' % (DELAYS[0], DELAYS[-1]))

    n_pulses = 15
    for synapses_loc, label, ax in zip([PROX,DIST], ['prox','dist'], AX):
        for delay in DELAYS:
            t, neuron, SEGMENTS = initialize_sim(Model)
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

            ax.plot(np.array(M.t/ntwk.ms), np.array(M.v/ntwk.mV)[0,:], label=label, color='k', lw=1)
        ge.set_plot(ax, [], ylim=[-76, -45], xlim=[0, tstop])
        ge.draw_bar_scales(ax,
                           Xbar=50., Xbar_label='50ms',
                           Ybar=5., Ybar_label='5mV',
                           loc='top-right')
    ge.show()
