import sys, pathlib, os
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np


import neural_network_dynamics.main as ntwk # my custom layer on top of Brian2
from graphs.my_graph import graphs # my custom data visualization library
from graphs.nrn_morpho import * # plotting neuronal morphologies

mg = graphs()


###################################################
# ---------- SIMULATION PARAMS  ----------------- #
###################################################

dt, tstop = 0.1*ntwk.ms, 200*ntwk.ms

############################################
# ---------- MORPHOLOGY  ----------------- #
############################################

# loading a morphology:
filename = os.path.join('neural_network_dynamics', 'single_cell_integration', 'morphologies', 'Jiang_et_al_2015', 'L5pyr-j140408b.CNG.swc')
morpho = ntwk.Morphology.from_swc_file(filename)

def get_compartment_list(morpho,
                         inclusion_condition='True',
                         without_axon=False):
    """
    condition should be of the form: 'comp.z>130'
    """
    
    COMP_LIST, INCLUSION = [], []
    exec("COMP_LIST.append(morpho); comp = COMP_LIST[-1]; INCLUSION.append("+inclusion_condition+")")
    
    TOPOL = str(morpho.topology())
    TT = TOPOL.split('\n')
    condition, comp, ii = True, None, 0
    for index, t in enumerate(TT[1:-1]):

        # exec("COMP_LIST.append(morpho."+t.split(' .')[-1]+"); comp = COMP_LIST[-1]; INCLUSION.append("+inclusion_condition+"); print("+inclusion_condition+")")
        exec("COMP_LIST.append(morpho."+t.split(' .')[-1]+"); comp = COMP_LIST[-1]; INCLUSION.append("+inclusion_condition+")")

        if without_axon and (len(t.split('axon'))>1):
            INCLUSION[-1] = False
    COMPARTMENT_LIST = []
    for c,i in zip(COMP_LIST, INCLUSION):
        if i:
            COMPARTMENT_LIST.append(c)
    return COMPARTMENT_LIST, np.arange(len(COMP_LIST))[np.array(INCLUSION, dtype=bool)]


# FULL LIST
COMP_LIST, INDICES = get_compartment_list(morpho)#, inclusion_condition='comp.x.mean()>200*ntwk.um')
soma = COMP_LIST[0]
# ONLY APICAL
COMP_LIST_APICAL, INDICES_APICAL = get_compartment_list(morpho,
                                                        inclusion_condition='np.sqrt(comp.y.mean()**2+comp.x.mean()**2+comp.z.mean()**2)>220*ntwk.um')

fig, ax = mg.figure(figsize=(0.6,2),
                    left=0., top=1., bottom=0., right=1.)
_, ax = plot_nrn_shape(mg, COMP_LIST, dend_color='gray', apic_color='gray', axon_color='None', ax=ax)
_, ax = plot_nrn_shape(mg, [soma]+COMP_LIST_APICAL,
                       dend_color='b', apic_color='b', axon_color='None', ax=ax) # plotting requires [soma], to be fixed


##################################################
# ---------- BIOPHYSICAL PROPS ----------------- #
##################################################

SEGMENT_LIST = get_segment_list(morpho)

gL = 1e-4*ntwk.siemens/ntwk.cm**2
EL = -60*ntwk.mV
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

###############################################
# ---------- SYNAPTIC MODEL ----------------- #
###############################################

# AMPA parameters
wAMPA = 0.5*ntwk.nS
tauAMPA = 2.*ntwk.ms

#NMDA parameters
wNMDA = 1.*ntwk.nS
tauRiseNMDA, tauDecayNMDA = 3.*ntwk.ms, 70.*ntwk.ms
V0NMDA = 1./0.08*ntwk.mV
# let's determine the peak-time for normalization
time = np.linspace(0, 100, int(1e3))*ntwk.ms
psp = lambda t: np.exp(-t/tauDecayNMDA)-np.exp(-t/tauRiseNMDA)
ANMDA = 1./psp(time[np.argmax(psp(time))]) # normalization in paper

EXC_SYNAPSES_EQUATIONS = '''dgAMPA/dt = -gAMPA/tauAMPA : siemens
                           dgRiseNMDA/dt = -gRiseNMDA/tauRiseNMDA : 1
                           dgDecayNMDA/dt = -gDecayNMDA/tauDecayNMDA : 1
                           gE_post = gAMPA+wNMDA*ANMDA*(gDecayNMDA-gRiseNMDA)/(1+0.3*exp(-v/V0NMDA)) : siemens (summed)''' 
ON_EXC_EVENT = 'gAMPA += wAMPA; gDecayNMDA += 1; gRiseNMDA += 1'

#####################################################
# ---------- SYNAPTIC STIMULATION ----------------- #
#####################################################

# ===> Evoked activity

stim_apic_index = 14
evoked_stimulation = ntwk.SpikeGeneratorGroup(2, np.arange(1), 5.*np.ones(1)*ntwk.ms)
ES = ntwk.Synapses(evoked_stimulation, neuron,
                  model=EXC_SYNAPSES_EQUATIONS,
                  on_pre=ON_EXC_EVENT)
ES.connect(i=0, j=INDICES_APICAL[stim_apic_index])


# ===> Background activity 

# Nsyn = 1
# Nspikes = 4
# spk_times = np.cumsum(np.random.exponential(.4, size=Nspikes))*tstop/Nspikes
# background_stimulation = ntwk.SpikeGeneratorGroup(Nsyn, # ids of synapses
#                                                   np.random.choice(np.arange(Nsyn), Nspikes), # picking which spike
#                                                   spk_times)
# BG = ntwk.Synapses(background_stimulation, neuron,
#                    model=EXC_SYNAPSES_EQUATIONS,
#                    on_pre=ON_EXC_EVENT)
# for i in range(Nsyn):
#     BG.connect(i=i, j=COMP_LIST_APICAL[stim_apic_index])
# BG.connect(i=i, j=COMP_LIST[np.random.choice(INDICES_APICAL)])

        
# # highlight soma & recorded point
ax.scatter([1e6*SEGMENT_LIST['xcoords'][0]], [1e6*SEGMENT_LIST['ycoords'][0]], s=100, edgecolors='k', marker='o', facecolors='none', lw=3)
ax.scatter([1e6*SEGMENT_LIST['xcoords'][stim_apic_index]], [1e6*SEGMENT_LIST['ycoords'][stim_apic_index]], s=100, edgecolors='b', marker='o', facecolors='none', lw=3)

# # # # recording and running
M = ntwk.StateMonitor(neuron, ('v'), record=[0, INDICES_APICAL[stim_apic_index]])#np.arange(len(neuron.v)))
ntwk.run(500.*ntwk.ms)

figT, axT = mg.figure(figsize=(3,1))
t = np.array(M.t/ntwk.ms)
Vm_soma = np.array(M.v/ntwk.mV)[0,:]
Vm_dend = np.array(M.v/ntwk.mV)[1,:]#[ES.j[0],:]
axT.plot(t, Vm_soma, 'k-')
axT.plot(t, Vm_dend, 'b-')

figT.savefig('figures/temp.svg')
# # mg.show()

# fig.savefig('figures/temp.svg')



