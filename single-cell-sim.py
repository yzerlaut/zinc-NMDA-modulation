import os
import numpy as np

import neural_network_dynamics.main as ntwk # my custom layer on top of Brian2
from datavyz import ge

from datavyz.nrn_morpho import plot_nrn_shape, coordinate_projection, add_dot_on_morpho # plotting neuronal morphologies


##########################################################
# -- EQUATIONS FOR THE SYNAPTIC AND CELLULAR BIOPHYSICS --
##########################################################
# cable theory:
eqs='''
Im = gL * (EL - v) : amp/meter**2
Is = gE * (Ee - v) + gI * (Ei - v)  + gclamp*(vc - v) : amp (point current)
gE : siemens
gI : siemens
gclamp : siemens
vc : volt # Voltage-clamp command
'''
# synaptic dynamics:
# -- excitation (NMDA-dependent)
EXC_SYNAPSES_EQUATIONS = '''dX/dt = -X/tauDecayZn : 1 (clock-driven)
                            dbZn/dt = (-bZn+ 1/(1+exp(-(X-x0)/deltax)))/tauRiseZn : 1 (clock-driven)
                            dgRiseAMPA/dt = -gRiseAMPA/tauRiseAMPA : 1 (clock-driven)
                            dgDecayAMPA/dt = -gDecayAMPA/tauDecayAMPA : 1 (clock-driven)
                            dgRiseNMDA/dt = -gRiseNMDA/tauRiseNMDA : 1 (clock-driven)
                            dgDecayNMDA/dt = -gDecayNMDA/tauDecayNMDA : 1 (clock-driven)
                            gE_post = 0*qAMPA*nAMPA*(gDecayAMPA-gRiseAMPA)+qNMDA*nNMDA*(gDecayNMDA-gRiseNMDA)/(1+0.3*exp(-v/V0NMDA))*(1-alphaZn*bZn) : siemens (summed)''' 
ON_EXC_EVENT = 'gDecayAMPA += 1; gRiseAMPA += 1; gDecayNMDA += 1; gRiseNMDA += 1; X+=Deltax0*(1-X)'
# -- inhibition (NMDA-dependent)
INH_SYNAPSES_EQUATIONS = '''dgGABA/dt = -gGABA/tauGABA : siemens (clock-driven)
                            gI_post = nGABA*gGABA : siemens (summed)''' 
ON_INH_EVENT = 'gGABA += wGABA'

def double_exp_normalization(T1, T2):
    return 1./(T2/T1-1)*((T2/T1)**(T2/(T2-T1)))

###################################################
# ---------- SIMULATION PARAMS  ----------------- #
###################################################

def initialize_sim(Model,
                   method='current-clamp',
                   Vclamp=0.,
                   verbose=False):

    # simulation params
    dt, tstop = Model['dt']*ntwk.ms, Model['tstop']*ntwk.ms
    np.random.seed(Model['seed'])
    # loading a morphology:
    morpho = ntwk.Morphology.from_swc_file(Model['morpho_file'])
    # fetching all compartments
    COMP_LIST, SEG_INDICES = ntwk.morpho_analysis.get_compartment_list(morpho)
    SEGMENTS = ntwk.morpho_analysis.compute_segments(morpho)
    
    if verbose:
        print('List of available compartment *types* for this morphology:',
              ntwk.morpho_analysis.list_compartment_types(COMP_LIST))
        
    # somatic compartment
    soma = ntwk.morpho_analysis.get_compartment_list(morpho,
                        inclusion_condition='comp.type=="soma"')[0]
    
    # restriction to basal dendrite
    DEND_COMP_LIST, DEND_INDICES = ntwk.morpho_analysis.get_compartment_list(morpho,
                                                    inclusion_condition='comp.type=="dend"')

    
    neuron = ntwk.SpatialNeuron(morphology=morpho,
                                model=eqs,
                                Cm=Model['cm'] * ntwk.uF / ntwk.cm ** 2,
                                Ri=Model['Ri'] * ntwk.ohm * ntwk.cm)
    neuron.gclamp = 0
    EL = Model['EL']*ntwk.mV
    
    if method=='voltage-clamp':
        gL = 10.*Model['gL']*1e-3*ntwk.siemens/ntwk.cm**2
        vc = Vclamp*ntwk.volt
        neuron.gclamp[0] = 10e-7*ntwk.siemens # >100 times somatic conductance
        neuron.v = vc
    else:
        gL = Model['gL']*1e-3*ntwk.siemens/ntwk.cm**2
        neuron.v = EL # Vm initialized to E
        
    Ee, Ei = Model['Ee']*ntwk.mV, Model['Ei']*ntwk.mV
    

    # AMPA parameters
    qAMPA = Model['qAMPA']*ntwk.nS
    tauRiseAMPA, tauDecayAMPA = Model['tauRiseAMPA']*ntwk.ms, Model['tauDecayAMPA']*ntwk.ms
    nAMPA = double_exp_normalization(tauRiseAMPA, tauDecayAMPA)
    # GABA parameters
    qGABA = Model['qGABA']*ntwk.nS
    tauRiseGABA, tauDecayGABA = Model['tauRiseGABA']*ntwk.ms, Model['tauDecayGABA']*ntwk.ms
    nGABA = double_exp_normalization(tauRiseGABA, tauDecayGABA)
    #NMDA parameters
    qNMDA = Model['qNMDA']*ntwk.nS
    tauRiseNMDA, tauDecayNMDA = Model['tauRiseNMDA']*ntwk.ms, Model['tauDecayNMDA']*ntwk.ms
    nNMDA = double_exp_normalization(tauRiseNMDA, tauDecayNMDA)
    V0NMDA = Model['V0NMDA']*ntwk.mV
    # Zinc-NMDA parameters
    alphaZn = Model['alphaZn']
    tauRiseZn, tauDecayZn = Model['tauRiseZn']*ntwk.ms, Model['tauDecayZn']*ntwk.ms
    Deltax0, deltax, x0 = Model['Deltax0'], Model['deltax'], Model['x0']

    synapse_ID, location_ID, time_ID = [], [], []
    # ===> Evoked activity at that specific location:
    for k in range(Model['Nsyn_synch_stim']):
        synapse_ID.append(0)
        time_ID.append(Model['tsyn_stim']+Model['dt']*(k+1))

    # # ===> Background activity everywhere
    # # excitation
    # for syn in range(1, Model['Nsyn_Ebg']+1):
    #     for e, event in enumerate(np.cumsum(np.random.exponential(1e3/Model['Fexc_bg'],
    #                                      size=int(1.3e-3*Model['tstop']*Model['Fexc_bg'])))):
    #         synapse_ID.append(syn)
    #         time_ID.append(event+e*Model['dt'])

    excitatory_stimulation = ntwk.SpikeGeneratorGroup(Model['Nsyn_Ebg']+1,
                                                      np.array(synapse_ID),
                                                      np.array(time_ID)*ntwk.ms)
    ES = ntwk.Synapses(excitatory_stimulation, neuron,
                       model=EXC_SYNAPSES_EQUATIONS,
                       on_pre=ON_EXC_EVENT)
    # connecting evoked activity synapse
    ES.connect(i=0, j=Model['stim_apic_compartment_index'])
    # connecting evoked activity synapse
    for syn in range(1, Model['Nsyn_Ebg']+1):
        ES.connect(i=syn, j=np.random.choice(range(900,1600))) # to be fixed
    ES.X, ES.bZn = 0, 0
    
    # # inhibition
    # synapseI_ID, timeI_ID = [], []
    # for syn in range(Model['Nsyn_Ibg):
    #     for e, event in enumerate(np.cumsum(np.random.exponential(1e3/Model['Finh_bg,
    #                                      size=int(1.3e-3*Model['tstop*Model['Finh_bg)))):
    #         synapseI_ID.append(syn)
    #         timeI_ID.append(event+e*Model['dt)

    # if Model['Nsyn_Ibg>0:
    #     inhibitory_stimulation = ntwk.SpikeGeneratorGroup(Model['Nsyn_Ibg,
    #                                                   np.array(synapseI_ID),
    #                                                   np.array(timeI_ID)*ntwk.ms)
    #     IS = ntwk.Synapses(inhibitory_stimulation, neuron,
    #                    model=INH_SYNAPSES_EQUATIONS,
    #                    on_pre=ON_INH_EVENT)
    # # connecting evoked activity synapse
    # for syn in range(Model['Nsyn_Ibg):
    #     IS.connect(i=syn, j=np.random.choice(range(900,1600))) # to be fixed

    # recording and running
    M = ntwk.StateMonitor(neuron, ('v'), record=[0, # soma
                                                 Model['stim_apic_compartment_index']])
    S = ntwk.StateMonitor(ES, ('X', 'bZn'), record=[0])
    
    # # Run simulation
    ntwk.run(tstop)

    output = {'t':np.array(M.t/ntwk.ms),
              'Vm_soma':np.array(M.v/ntwk.mV)[0,:],
              'bZn_syn':np.array(S.bZn)[0,:],
              'X_syn':np.array(S.X)[0,:],
              'Vm_syn':np.array(M.v/ntwk.mV)[1,:]}

    output['Ic'] = (output['Vm_soma']-Vclamp)*neuron.gclamp[0]
    return output

def plot_signals(output, ge=None):

    fig, AX = ge.figure(axes_extents=[[[3,1]],
                                      [[3,1]],
                                      [[3,1]]])
    cond = output['t']>10
    # plotting
    # AX[0].plot(output['t'], output['bZn_syn'], '-', color=ge.colors[0])
    # AX[0].plot(output['t'], output['X_syn'], '-', color=ge.colors[0])
    AX[0].plot(output['t'][cond], 1e6*(output['Ic'][cond]-output['Ic'][cond][0]),
               '-', color=ge.colors[0])
    # AX[0].plot(output['t'], output['X_syn'], '-', color=ge.colors[0])
    AX[1].plot(output['t'][cond], output['Vm_syn'][cond], '-', color=ge.colors[1])
    AX[2].plot(output['t'][cond], output['Vm_soma'][cond], '-', color=ge.colors[2])
    
    # for i, ax, label in zip(range(3), AX[1:], ['synaptic location', 'tuft start', 'soma']):
    #     ge.set_plot(ax, ['left'], ylabel='mV')
    #     ge.annotate(ax, '$V_m$ at %s' % label, (1., 0.), ha='right', va='top', color=ge.colors[i])
        
    # Tbar = 20 # ms
    # AX[1].plot([output['t'][-1], output['t'][-1]-Tbar], AX[1].get_ylim()[1]*np.ones(2), '-',
    #            color=ge.default_color)    
    # ge.annotate(AX[1], '%sms' % Tbar, (.98, 1.), ha='right', color=ge.default_color)
    
    return fig

    

if __name__=='__main__':
    
    from model import Model

    output = initialize_sim(Model, method='voltage-clamp')
    from datavyz import ges as ge
    plot_signals(output, ge=ge)
    ge.show()
