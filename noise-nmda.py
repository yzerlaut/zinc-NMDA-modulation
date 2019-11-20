import sys, pathlib, os
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np


import neural_network_dynamics.main as ntwk # my custom layer on top of Brian2
from graphs.my_graph import graphs # my custom data visualization library
from graphs.nrn_morpho import * # plotting neuronal morphologies

mg = graphs()

# loading a morphology:
filename = os.path.join('neural_network_dynamics', 'single_cell_integration', 'morphologies', 'Jiang_et_al_2015', 'L5pyr-j140408b.CNG.swc')
morpho = ntwk.Morphology.from_swc_file(filename)

print(morpho.apic[150.*ntwk.um].indices)
print(morpho.apic[3])

COMP_LIST = get_compartment_list(morpho)
SEGMENT_LIST = get_segment_list(morpho)

gL = 1e-4*ntwk.siemens/ntwk.cm**2
EL = -70*ntwk.mV
Ee, Ei = 0*ntwk.mV, -80*ntwk.mV

eqs='''
Im = gL * (EL - v) : amp/meter**2
Is = gE * (Ee - v) + gI * (Ei - v) : amp (point current)
gE : siemens
gI : siemens
'''
neuron = ntwk.SpatialNeuron(morphology=morpho,
                            model=eqs,
                            Cm=1 * ntwk.uF / ntwk.cm ** 2,
                            Ri=100 * ntwk.ohm * ntwk.cm)
neuron.v = EL

evoked_stimulation = ntwk.SpikeGeneratorGroup(2, np.arange(1), 5.*np.ones(1)*ntwk.ms)

# AMPA parameters
wAMPA = 1.*ntwk.nS
tauAMPA = 5.*ntwk.ms

#NMDA parameters
wNMDA = 1.*ntwk.nS
tauRiseNMDA, tauDecayNMDA = 3.*ntwk.ms, 70.*ntwk.ms
V0NMDA = -1./0.08*ntwk.mV

EXC_SYNAPSES_EQUATIONS = '''dgAMPA/dt = -gAMPA/tauAMPA : siemens
                           dgRiseNMDA/dt = -gRiseNMDA/tauRiseNMDA : 1
                           dgDecayNMDA/dt = -gDecayNMDA/tauDecayNMDA : 1
                           gE_post = gAMPA+wNMDA*(gDecayNMDA-gRiseNMDA)/(1+0.3*exp(-v/V0NMDA)) : siemens (summed)''' 
ON_EXC_EVENT = 'gAMPA += wAMPA; gDecayNMDA += 1; gRiseNMDA += 1'

# Evoked
ES = ntwk.Synapses(evoked_stimulation, neuron,
                  model=EXC_SYNAPSES_EQUATIONS,
                  on_pre=ON_EXC_EVENT)
ES.connect(i=0, j=morpho.apic2.apic2.apic2.apic2.apic2.apic2.apic[10.*ntwk.um])


APIC_COMP_LIST = []
for c in COMP_LIST:
    if c.type=='apic':
        APIC_COMP_LIST.append(c)
        
fig, ax = mg.figure(figsize=(0.6,2),
                    left=0., top=1., bottom=0., right=1.)
# plot full dendritic arbor and highlight apical tuft
fig, ax = plot_nrn_shape(mg, COMP_LIST, dend_color='gray', apic_color='b', axon_color='None', ax=ax)
# highlight soma
ax.scatter([1e6*SEGMENT_LIST['xcoords'][0]], [1e6*SEGMENT_LIST['ycoords'][0]], s=100, edgecolors='k', marker='o', facecolors='none', lw=3)
ax.scatter([1e6*SEGMENT_LIST['xcoords'][ES.j[0]]], [1e6*SEGMENT_LIST['ycoords'][ES.j[0]]], s=100, edgecolors='b', marker='o', facecolors='none', lw=3)

# # # recording and running
M = ntwk.StateMonitor(neuron, ('v'), record=np.arange(len(neuron.v)))
ntwk.run(100.*ntwk.ms)

figT, axT = mg.figure(figsize=(3,1))
t = np.array(M.t/ntwk.ms)
Vm_soma = np.array(M.v/ntwk.mV)[0,:]
Vm_dend = np.array(M.v/ntwk.mV)[ES.j[0],:]
axT.plot(t, Vm_soma, 'k-')
axT.plot(t, Vm_dend, 'b-')

# mg.show()

