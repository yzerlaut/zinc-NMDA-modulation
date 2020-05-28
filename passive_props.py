import os
import numpy as np

import neural_network_dynamics.main as ntwk # my custom layer on top of Brian2
from datavyz import ge

from scipy.optimize import minimize


##########################################################
# -- EQUATIONS FOR THE SYNAPTIC AND CELLULAR BIOPHYSICS --
##########################################################
# cable theory:
eqs='''
Im = ({gL}*siemens/meter**2) * (({EL}*mV) - v) : amp/meter**2
I : amp (point current) # applied current
'''


def step(t, t0, t1):
    return (np.sign(t-t0)-np.sign(t-t1))/2.

def heaviside(t, t1):
    return (np.sign(t-t1)+1)/2.

def func_to_fit(t, coeffs, t0=50, t1=150):
    Ibsl, IbslShift, IexpComp, Tau = coeffs
    return Ibsl+\
        step(t, t0, t1)*(-IbslShift-IexpComp*np.exp(-(t-t0)/Tau))+\
        heaviside(t, t1)*IexpComp*np.exp(-(t-t1)/Tau)

###################################################
# ---------- SIMULATION PARAMS  ----------------- #
###################################################

def initialize_sim(Model,
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
    
    gL = Model['gL']*ntwk.siemens/ntwk.meter**2
    neuron.v = Model['EL']*ntwk.mV # Vm initialized to E

    return t, neuron, SEGMENTS


def run_voltage_clamp_protocol(Model,
                               clamps = [{'value':0., 'duration':1000.}],
                               with_plot=False, ge=None):

    t, neuron, SEGMENTS = initialize_sim(Model)
    
    # recording and running
    M = ntwk.StateMonitor(neuron, ('v', 'I'), record=[0])

    t = 0
    for clamp in clamps:

        neuron.I[0] = clamp['value']*ntwk.pA
        ntwk.run(clamp['duration']*ntwk.ms)
        t += clamp['duration']

    output = {'t':np.array(M.t/ntwk.ms),
              'Ic':np.array(np.array(M.I/ntwk.pA)[0,:]),
              'Vm_soma':np.array(M.v/ntwk.mV)[0,:]}

    return output


if __name__=='__main__':

    from model import Model
    
    output = run_voltage_clamp_protocol(Model,
                                        clamps = [{'value':0., 'duration':50},
                                                  {'value':200, 'duration':100},
                                                  {'value':0., 'duration':50}])
    import sys
    sys.path.append('/home/yann/work/cortical-physio-icm') # append CPI to your path to be able to import
    from electrophy.intracellular.passive_props import perform_ICcharact

    # print(perform_ICcharact(1e-3*output['t'], 1e-3*output['Vm_soma'],
    #                         t0=50e-3, t1=150e-3, with_plot=False))
    
    perform_ICcharact(1e-3*output['t'], 1e-3*output['Vm_soma'],
                      t0=50e-3, t1=150e-3, with_plot=True)
    ge.show()
