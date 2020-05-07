import os
import numpy as np

import neural_network_dynamics.main as ntwk # my custom layer on top of Brian2
from datavyz import ge

from datavyz import nrnvyz


##########################################################
# -- EQUATIONS FOR THE SYNAPTIC AND CELLULAR BIOPHYSICS --
##########################################################
# cable theory:
eqs='''
Im = ({gL}*siemens/meter**2) * (({EL}*mV) - v) : amp/meter**2
Is = gE * (({Ee}*mV) - v) + gI * (({Ei}*mV) - v) + gclamp*(vc - v) : amp (point current)
gE : siemens
gI : siemens
gclamp : siemens
vc : volt # Voltage-clamp command
'''

# synaptic dynamics:

# -- excitation (NMDA-dependent)
from model import Model
EXC_SYNAPSES_EQUATIONS = '''dX/dt = -X/({tauDecayZn}*ms) : 1 (clock-driven)
                            dbZn/dt = (-bZn+ 1/(1+exp(-(X-{x0})/{deltax})))/({tauRiseZn}*ms) : 1 (clock-driven)
                            dgRiseAMPA/dt = -gRiseAMPA/({tauRiseAMPA}*ms) : 1 (clock-driven)
                            dgDecayAMPA/dt = -gDecayAMPA/({tauDecayAMPA}*ms) : 1 (clock-driven)
                            dgRiseNMDA/dt = -gRiseNMDA/({tauRiseNMDA}*ms) : 1 (clock-driven)
                            dgDecayNMDA/dt = -gDecayNMDA/({tauDecayNMDA}*ms) : 1 (clock-driven)
                            gAMPA = ({qAMPA}*nS)*{nAMPA}*(gDecayAMPA-gRiseAMPA) : siemens
                            gNMDA = ({qNMDA}*nS)*{nNMDA}*(gDecayNMDA-gRiseNMDA)/(1+0.3*exp(-v_post/({V0NMDA}*mV)))*(1-{alphaZn}*bZn) : siemens
                            gE_post = gAMPA+gNMDA : siemens (summed)'''
ON_EXC_EVENT = 'gDecayAMPA += 1; gRiseAMPA += 1; gDecayNMDA += 1; gRiseNMDA += 1; X+={Deltax0}*(1-X)'

# -- inhibition (NMDA-dependent)
INH_SYNAPSES_EQUATIONS = '''dgRiseGABA/dt = -gRiseGABA/({tauRiseGABA}*ms) : 1 (clock-driven)
                            dgDecayGABA/dt = -gDecayGABA/({tauDecayGABA}*ms) : 1 (clock-driven)
                            gI_post = {nGABA}*({qGABA}*nS)*(gDecayGABA-gRiseGABA) : siemens (summed)''' 
ON_INH_EVENT = 'gRiseGABA += 1; gDecayGABA += 1'


###################################################
# ---------- SIMULATION PARAMS  ----------------- #
###################################################

def initialize_sim(Model,
                   method='current-clamp',
                   Vclamp=0.,
                   verbose=False):

    # simulation params
    t = np.arange(int(Model['tstop']/Model['dt']))*Model['dt']
    
    np.random.seed(Model['seed'])
    
    # loading a morphology:
    morpho = ntwk.Morphology.from_swc_file(Model['morpho_file'])
    SEGMENTS = ntwk.morpho_analysis.compute_segments(morpho)
        
    neuron = ntwk.SpatialNeuron(morphology=morpho,
                                model=eqs.format(**Model),
                                Cm=Model['cm'] * ntwk.uF / ntwk.cm ** 2,
                                Ri=Model['Ri'] * ntwk.ohm * ntwk.cm)
    
    neuron.gclamp = 0 # everywhere
    
    if method=='voltage-clamp':
        gL = Model['gL']*ntwk.siemens/ntwk.meter**2/Model['VC-gL-reduction-factor']
        neuron.vc = Model['VC-cmd']*ntwk.mV
        neuron.gclamp[0] = Model['VC-gclamp']*ntwk.uS # >100 times somatic conductance
        neuron.v = Model['VC-cmd']*ntwk.mV
    else:
        gL = Model['gL']*ntwk.siemens/ntwk.meter**2
        neuron.v = Model['EL']*ntwk.mV # Vm initialized to E

    return t, neuron, SEGMENTS

def set_background_network_stim(t, neuron, SEGMENTS, Model):
    
    Nsyn_Exc, pre_to_iseg_Exc, Nsyn_per_seg_Exc = ntwk.spread_synapses_on_morpho(SEGMENTS,
                                                                         Model['DensityAMPA'],
                                                                cond=SEGMENTS['comp_type']=='dend',
                                                                density_factor=1./100./1e-12,
                                                                verbose=True)
    Espike_IDs, Espike_times = ntwk.spikes_from_time_varying_rate(t, 0*t+Model['Fexc_bg'], N=Nsyn_Exc)
    Estim, ES = ntwk.process_and_connect_event_stimulation(neuron,
                                                           Espike_IDs, Espike_times,
                                                           pre_to_iseg_Exc,
                                                           EXC_SYNAPSES_EQUATIONS.format(**Model),
                                                           ON_EXC_EVENT.format(**Model))

    Nsyn_Inh, pre_to_iseg_Inh, Nsyn_per_seg_Inh = ntwk.spread_synapses_on_morpho(SEGMENTS,
                                                                         Model['DensityGABA'],
                                                                cond=SEGMENTS['comp_type']!='apic',
                                                                density_factor=1./100./1e-12,
                                                                verbose=True)
    Ispike_IDs, Ispike_times = ntwk.spikes_from_time_varying_rate(t, 0*t+Model['Finh_bg'], N=Nsyn_Inh)
    Istim, IS = ntwk.process_and_connect_event_stimulation(neuron,
                                                           Ispike_IDs, Ispike_times,
                                                           pre_to_iseg_Inh,
                                                           INH_SYNAPSES_EQUATIONS.format(**Model),
                                                           ON_INH_EVENT.format(**Model))
    
    ES.X, ES.bZn = 0, 0

    return Estim, ES, Istim, IS

def run(neuron, Model, Estim, ES, Istim, IS):

    # recording and running
    M = ntwk.StateMonitor(neuron, ('v'), record=[0, # soma
                                                 Model['stim_apic_compartment_index']])
    S = ntwk.StateMonitor(ES, ('X', 'gAMPA', 'gNMDA', 'bZn'), record=[0])
    
    # # Run simulation
    ntwk.run(Model['tstop']*ntwk.ms)

    output = {'t':np.array(M.t/ntwk.ms),
              'Vm_soma':np.array(M.v/ntwk.mV)[0,:],
              'bZn_syn':np.array(S.bZn)[0,:],
              'gAMPA_syn':np.array(S.gAMPA/ntwk.nS)[0,:],
              'gNMDA_syn':np.array(S.gNMDA/ntwk.nS)[0,:],
              'X_syn':np.array(S.X)[0,:],
              'Vm_syn':np.array(M.v/ntwk.mV)[1,:]}

    output['Ic'] = (output['Vm_soma']-Model['VC-cmd'])*Model['VC-gclamp'] # nA
    return output

def plot_signals(output, ge=None):
    fig, AX = ge.figure(axes_extents=[[[3,1]],
                                      [[3,1]],
                                      [[3,1]],
                                      [[3,1]]])
    cond = output['t']>150
    AX[0].plot(output['t'][cond], output['bZn_syn'][cond], '-', color=ge.colors[0])
    AX[1].plot(output['t'][cond], output['Vm_syn'][cond], '-', color=ge.colors[1])
    AX[2].plot(output['t'][cond], output['Vm_soma'][cond], '-', color=ge.colors[2])
    return fig

    

if __name__=='__main__':
    
    from model import Model
    # Model['tsyn_stim'] = 170
    t, neuron, SEGMENTS = initialize_sim(Model)
    
    # Estim, ES, Istim, IS = set_background_network_stim(t, neuron, SEGMENTS, Model)
    # output = run(neuron, Model, Estim, ES, Istim, IS)
    
    # from datavyz import ges as ge
    # plot_signals(output, ge=ge)
    # ge.show()

    synapses_loc = np.arange(5)
    spike_IDs, spike_times = np.empty(0, dtype=int), np.empty(0, dtype=float)
    t0_stim, n_pulses, freq_pulses = 50, 3, 20
    for i in range(n_pulses):
        spike_times = np.concatenate([spike_times,
                                      (t0_stim+i*1e3/freq_pulses)*np.ones(len(synapses_loc))])
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
    
