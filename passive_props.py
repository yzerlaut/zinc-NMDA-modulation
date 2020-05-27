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
Is = gclamp*(vc - v) : amp (point current)
gclamp : siemens
vc : volt # Voltage-clamp command
'''


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


def run_voltage_clamp_protocol(Model,
                               clamps = [{'value':0., 'duration':1000.}]):

    t, neuron, SEGMENTS = initialize_sim(Model, method='voltage-clamp')
    
    # recording and running
    M = ntwk.StateMonitor(neuron, ('v'), record=[0])

    t = 0
    for clamp in clamps:

        neuron.vc = clamp['value']*ntwk.mV
        ntwk.run(clamp['duration']*ntwk.ms)
        t += clamp['duration']

    output = {'t':np.array(M.t/ntwk.ms),
              'Vm_soma':np.array(M.v/ntwk.mV)[0,:]}
    output['Ic'] = (output['Vm_soma']-Model['VC-cmd'])*Model['VC-gclamp'] # nA
    return output


if __name__=='__main__':

    from model import Model
    
    output = run_voltage_clamp_protocol(Model,
                                        clamps = [{'value':0., 'duration':50},
                                                  {'value':-5., 'duration':50},
                                                  {'value':0., 'duration':50}])
    from datavyz import ge

    ge.plot(output['Ic'])
    ge.show()
