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
Is = gE * (Ee - v) + gI * (Ei - v) : amp (point current)
gE : siemens
gI : siemens
'''
# synaptic dynamics:
# -- excitation (NMDA-dependent)
EXC_SYNAPSES_EQUATIONS = '''dgAMPA/dt = -gAMPA/tauAMPA : siemens (clock-driven)
                           dgRiseNMDA/dt = -gRiseNMDA/tauRiseNMDA : 1 (clock-driven)
                           dgDecayNMDA/dt = -gDecayNMDA/tauDecayNMDA : 1 (clock-driven)
                           gE_post = gAMPA+wNMDA*ANMDA*(gDecayNMDA-gRiseNMDA)/(1+0.3*exp(-v/V0NMDA)) : siemens (summed)''' 
ON_EXC_EVENT = 'gAMPA += wAMPA; gDecayNMDA += 1; gRiseNMDA += 1'
# -- inhibition (NMDA-dependent)
INH_SYNAPSES_EQUATIONS = '''dgGABA/dt = -gGABA/tauGABA : siemens (clock-driven)
                            gI_post = gGABA : siemens (summed)''' 
ON_INH_EVENT = 'gGABA += wGABA'

###################################################
# ---------- SIMULATION PARAMS  ----------------- #
###################################################

def run_sim(args):

    # simulation params
    dt, tstop = args.dt*ntwk.ms, args.tstop*ntwk.ms
    np.random.seed(args.seed)
    # loading a morphology:
    morpho = ntwk.Morphology.from_swc_file(args.morpho)
    # fetching all compartments
    COMP_LIST, SEG_INDICES = ntwk.morpho_analysis.get_compartment_list(morpho)
    SEGMENTS = ntwk.morpho_analysis.compute_segments(morpho)
    
    if args.verbose:
        print('List of available compartment *types* for this morphology:',
              ntwk.morpho_analysis.list_compartment_types(COMP_LIST))
        
    # somatic compartment
    soma = ntwk.morpho_analysis.get_compartment_list(morpho,
                        inclusion_condition='comp.type=="soma"')[0]
    
    # restriction to apical tuft
    APICAL_COMP_LIST, APICAL_NSEG_INDICES = ntwk.morpho_analysis.get_compartment_list(morpho,
                                                    inclusion_condition='comp.type=="apic"')
    # inclusion_condition='(comp.type=="apic") and np.sqrt(comp.y.mean()**2+comp.x.mean()**2+comp.z.mean()**2)>130*ntwk.um')
    
    # full dendritic
    DEND_COMP_LIST, DEND_INDICES = ntwk.morpho_analysis.get_compartment_list(morpho,
                                   inclusion_condition='comp.type in ["dend", "apic"]')

    gL = args.gL*ntwk.siemens/ntwk.cm**2
    EL = args.EL*ntwk.mV
    Ee, Ei = args.Ee*ntwk.mV, args.Ei*ntwk.mV

    neuron = ntwk.SpatialNeuron(morphology=morpho,
                                model=eqs,
                                Cm=args.cm * ntwk.uF / ntwk.cm ** 2,
                                Ri=args.Ri * ntwk.ohm * ntwk.cm)
    
    neuron.v = EL # Vm initialized to E

    # AMPA parameters
    wAMPA, tauAMPA = args.wAMPA*ntwk.nS, args.tauAMPA*ntwk.ms
    #NMDA parameters
    wNMDA = args.wNMDA*ntwk.nS
    tauRiseNMDA, tauDecayNMDA = args.tauRiseNMDA*ntwk.ms, args.tauDecayNMDA*ntwk.ms
    V0NMDA = args.V0NMDA*ntwk.mV
    # GABA parameters
    wGABA, tauGABA = args.wGABA*ntwk.nS, args.tauGABA*ntwk.ms
    # let's determine the peak-time for normalization
    time = np.linspace(0, 100, int(1e3))*ntwk.ms
    psp = lambda t: np.exp(-t/tauDecayNMDA)-np.exp(-t/tauRiseNMDA)
    ANMDA = 1./psp(time[np.argmax(psp(time))]) # normalization in paper

    synapse_ID, location_ID, time_ID = [], [], []
    # ===> Evoked activity at that specific location:
    for k in range(args.Nsyn_synch_stim):
        synapse_ID.append(0)
        time_ID.append(args.tsyn_stim+args.dt*(k+1))

    # ===> Background activity everywhere
    # excitation
    for syn in range(1, args.Nsyn_Ebg+1):
        for e, event in enumerate(np.cumsum(np.random.exponential(1e3/args.Fexc_bg,
                                         size=int(1.3e-3*args.tstop*args.Fexc_bg)))):
            synapse_ID.append(syn)
            time_ID.append(event+e*args.dt)

    excitatory_stimulation = ntwk.SpikeGeneratorGroup(args.Nsyn_Ebg+1,
                                                      np.array(synapse_ID),
                                                      np.array(time_ID)*ntwk.ms)
    ES = ntwk.Synapses(excitatory_stimulation, neuron,
                       model=EXC_SYNAPSES_EQUATIONS,
                       on_pre=ON_EXC_EVENT)
    # connecting evoked activity synapse
    ES.connect(i=0, j=args.stim_apic_compartment_index)
    # connecting evoked activity synapse
    for syn in range(1, args.Nsyn_Ebg+1):
        ES.connect(i=syn, j=np.random.choice(range(900,1600))) # to be fixed
    
    # inhibition
    synapseI_ID, timeI_ID = [], []
    for syn in range(args.Nsyn_Ibg):
        for e, event in enumerate(np.cumsum(np.random.exponential(1e3/args.Finh_bg,
                                         size=int(1.3e-3*args.tstop*args.Finh_bg)))):
            synapseI_ID.append(syn)
            timeI_ID.append(event+e*args.dt)

    if args.Nsyn_Ibg>0:
        inhibitory_stimulation = ntwk.SpikeGeneratorGroup(args.Nsyn_Ibg,
                                                      np.array(synapseI_ID),
                                                      np.array(timeI_ID)*ntwk.ms)
        IS = ntwk.Synapses(inhibitory_stimulation, neuron,
                       model=INH_SYNAPSES_EQUATIONS,
                       on_pre=ON_INH_EVENT)
    # connecting evoked activity synapse
    for syn in range(args.Nsyn_Ibg):
        IS.connect(i=syn, j=np.random.choice(range(900,1600))) # to be fixed

    # recording and running
    M = ntwk.StateMonitor(neuron, ('v'), record=[0, # soma
                                                 args.rec2_apic_compartment_index,
                                                 args.stim_apic_compartment_index])
    
    # Run simulation
    ntwk.run(tstop)

    output = {'t':np.array(M.t/ntwk.ms),
              'Vm_soma':np.array(M.v/ntwk.mV)[0,:],
              'Vm_apic':np.array(M.v/ntwk.mV)[1,:],
              'Vm_syn':np.array(M.v/ntwk.mV)[2,:]}
    
    return output

def plot_nrn_and_signals(args, output, ge=None):

    if ge is None:
        from datavyz import ges as ge
        
    # loading the morphology
    morpho = ntwk.Morphology.from_swc_file(args.morpho)
    SEGMENTS = ntwk.morpho_analysis.compute_segments(morpho)

    fig, AX = ge.figure(figsize=(1.5, 1.),
                        left=.1, bottom=.1,
                        hspace=.1,
                        grid=[(0,0,1,3),
                              (1,0,2,1),
                              (1,1,2,1),
                              (1,2,2,1)])

    _, ax = plot_nrn_shape(ge, SEGMENTS, ax=AX[0],
                           comp_type=['apic', 'dend', 'soma'])
    
    ge.annotate(fig, args.morpho.split(os.path.sep)[-1].split('.CNG')[0], (0.01, 0.01))
    
    add_dot_on_morpho(ge, AX[0], SEGMENTS, args.stim_apic_compartment_index, color=ge.colors[0])
    add_dot_on_morpho(ge, AX[0], SEGMENTS, args.rec2_apic_compartment_index, color=ge.colors[1])
    add_dot_on_morpho(ge, AX[0], SEGMENTS, 0, color=ge.colors[2])

    # plotting
    AX[1].plot(output['t'], output['Vm_syn'], '-', color=ge.colors[0])
    AX[2].plot(output['t'], output['Vm_apic'], '-', color=ge.colors[1])
    AX[3].plot(output['t'], output['Vm_soma'], '-', color=ge.colors[2])
    
    for i, ax, label in zip(range(3), AX[1:], ['synaptic location', 'tuft start', 'soma']):
        ge.set_plot(ax, ['left'], ylabel='mV')
        ge.annotate(ax, '$V_m$ at %s' % label, (1., 0.), ha='right', va='top', color=ge.colors[i])
        
    Tbar = 20 # ms
    AX[1].plot([output['t'][-1], output['t'][-1]-Tbar], AX[1].get_ylim()[1]*np.ones(2), '-',
               color=ge.default_color)    
    ge.annotate(AX[1], '%sms' % Tbar, (.98, 1.), ha='right', color=ge.default_color)
    
    return fig

    


if __name__=='__main__':
    
    import argparse
    # First a nice documentation 
    parser=argparse.ArgumentParser(description=""" 
     Running simulations of morphologically-detailed models using Brian2
     """
    ,formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-p', "--protocol",
                        help="either: 'run' / 'plot' / 'demo' ")

    ############################################
    # ---------- MORPHOLOGY  ----------------- #
    ############################################
    parser.add_argument("--morpho", help='path to the ".swc" morphology ',
                        default=os.path.join('neural_network_dynamics',
                                             'single_cell_integration',
                                             'morphologies',
                                             'Jiang_et_al_2015',
                                             'L23pyr-j150407a.CNG.swc'))
    #L23pyr-j150407a.CNG.swc
    #L23pyr-j150811a.CNG.swc
    
    ###################################################
    # ---------- SIMULATION PARAMS  ----------------- #
    ###################################################
    parser.add_argument("--tstop", help='[ms]', type=float, default=200.)
    parser.add_argument("--dt", help='[ms]', type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    ##################################################
    # ---------- BIOPHYSICAL PROPS ----------------- #
    ##################################################
    parser.add_argument("--gL", help='[S/cm2]', type=float, default=1e-4)
    parser.add_argument("--cm", help='[uF/cm2]', type=float, default=1.)
    parser.add_argument("--Ri", help='[Ohm*cm]', type=float, default=100.)
    parser.add_argument("--EL", help='[mV]', type=float, default=-70.)
    ###################################################
    # ---------- SYNAPTIC PARAMS  ----------------- #
    ###################################################
    parser.add_argument("--Ee", help='[mV]', type=float, default=0.)
    parser.add_argument("--Ei", help='[mV]', type=float, default=-80.)
    parser.add_argument("--wAMPA", help='[nS]', type=float, default=0.5)
    parser.add_argument("--wNMDA", help='[nS]', type=float, default=1.)
    parser.add_argument("--wGABA", help='[nS]', type=float, default=1.)
    parser.add_argument("--tauAMPA", help='[ms]', type=float, default=2.)
    parser.add_argument("--tauGABA", help='[ms]', type=float, default=5.)
    parser.add_argument("--tauRiseNMDA", help='[ms]', type=float, default=3.)
    parser.add_argument("--tauDecayNMDA", help='[ms]', type=float, default=70.)
    parser.add_argument("--V0NMDA", help='[mV]', type=float, default=1./0.08)
    #############################################################
    # ---------- SYNAPTIC STIMULATION PARAMS  ----------------- #
    #############################################################
    # evoked
    parser.add_argument("--stim_apic_compartment_index", type=int, default=1652)
    parser.add_argument("--rec2_apic_compartment_index", type=int, default=860)
    parser.add_argument("--Nsyn_synch_stim", type=int, default=5)
    parser.add_argument("--tsyn_stim", type=int, default=10.)
    # bg
    parser.add_argument("--Nsyn_Ebg", type=int, default=0)
    parser.add_argument("--Nsyn_Ibg", type=int, default=0)
    parser.add_argument("--Fexc_bg", type=float, default=10.)
    parser.add_argument("--Finh_bg", type=float, default=10.)
    
    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                        action="store_true")
    parser.add_argument("-s", "--save", help="save the figures",
                        action="store_true")
    parser.add_argument("--filename", '-f', help="filename",type=str, default='data.npz')
    parser.add_argument("--fig", help="filename for saving a figure" ,type=str, default='')
    
    args = parser.parse_args()

    if args.protocol=='run':
        run_sim(args)
    elif args.protocol=='plot':
        pass
    else:
        output = run_sim(args)
        fig = plot_nrn_and_signals(args, output)
        if args.fig=='':
            ge.show()
        else:
            fig.savefig(args.fig)
