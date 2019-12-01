import sys, pathlib, os
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np

import neural_network_dynamics.main as ntwk # my custom layer on top of Brian2
from graphs.my_graph import graphs # my custom data visualization library
from graphs.nrn_morpho import plot_nrn_shape, coordinate_projection, add_dot_on_morpho # plotting neuronal morphologies

# mg = graphs('dark_emacs_png')
mg = graphs('screen')

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
EXC_SYNAPSES_EQUATIONS = '''dgAMPA/dt = -gAMPA/tauAMPA : siemens (clock-driven)
                           dgRiseNMDA/dt = -gRiseNMDA/tauRiseNMDA : 1 (clock-driven)
                           dgDecayNMDA/dt = -gDecayNMDA/tauDecayNMDA : 1 (clock-driven)
                           gE_post = gAMPA+wNMDA*ANMDA*(gDecayNMDA-gRiseNMDA)/(1+0.3*exp(-v/V0NMDA)) : siemens (summed)''' 
ON_EXC_EVENT = 'gAMPA += wAMPA; gDecayNMDA += 1; gRiseNMDA += 1'


stim_apic_compartment_index, stim_apic_compartment_seg = -1, 0
rec2_apic_compartment_index, rec2_apic_compartment_seg = -3, 0

###################################################
# ---------- SIMULATION PARAMS  ----------------- #
###################################################

def run_sim(args):

    # simulation params
    dt, tstop = args.dt*ntwk.ms, args.tstop*ntwk.ms
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
    # let's determine the peak-time for normalization
    time = np.linspace(0, 100, int(1e3))*ntwk.ms
    psp = lambda t: np.exp(-t/tauDecayNMDA)-np.exp(-t/tauRiseNMDA)
    ANMDA = 1./psp(time[np.argmax(psp(time))]) # normalization in paper

    # ===> Evoked activity at that specific location:
    synapse_ID, location_ID, time_ID = [0], [(stim_apic_compartment_index, stim_apic_compartment_seg)], [10.]
    for k in range(10):
        synapse_ID.append(0)
        location_ID.append((stim_apic_compartment_index, stim_apic_compartment_seg))
        time_ID.append(10.+0.1*(k+1))

        evoked_stimulation = ntwk.SpikeGeneratorGroup(len(synapse_ID),
                                                      np.array(synapse_ID),
                                                      np.array(time_ID)*ntwk.ms)
        ES = ntwk.Synapses(evoked_stimulation, neuron,
                           model=EXC_SYNAPSES_EQUATIONS,
                           on_pre=ON_EXC_EVENT)
        # for i in range(len(synapse_ID)):
        #     ES.connect(i=i, j=APICAL_NSEG_INDICES[location_ID[i][0]][location_ID[i][1]])

    ES.connect(i=0, j=SEGMENTS['index'][-1])

    # recording and running
    M = ntwk.StateMonitor(neuron, ('v'), record=[0, # soma
            APICAL_NSEG_INDICES[rec2_apic_compartment_index][rec2_apic_compartment_seg],
            APICAL_NSEG_INDICES[stim_apic_compartment_index][stim_apic_compartment_seg]])
    
    # Run simulation
    ntwk.run(tstop)

    output = {'t':np.array(M.t/ntwk.ms),
              'Vm_soma':np.array(M.v/ntwk.mV)[0,:],
              'Vm_apic':np.array(M.v/ntwk.mV)[1,:],
              'Vm_syn':np.array(M.v/ntwk.mV)[2,:]}
    
    return output

def plot_nrn_and_signals(args, output):

    # loading the morphology
    morpho = ntwk.Morphology.from_swc_file(args.morpho)
    soma = ntwk.morpho_analysis.get_compartment_list(morpho,
                                                     inclusion_condition='comp.type=="soma"')[0]
    DEND_COMP_LIST, DEND_INDICES = ntwk.morpho_analysis.get_compartment_list(morpho,
                                   inclusion_condition='comp.type in ["dend", "apic"]')
    
    fig, AX = mg.figure(figsize=(.95,.38),
                        left=.1, bottom=.1,
                        wspace=.6, hspace=.1,
                        grid=[(0,0,1,3),
                              (1,0,3,1),
                              (1,1,3,1),
                              (1,2,3,1)])

    _, ax = plot_nrn_shape(mg, DEND_COMP_LIST,
                           soma_comp=soma,
                           ax=AX[0])
    
    mg.title(AX[0], args.morpho.split(os.path.sep)[-1].split('.CNG')[0])
    
    add_dot_on_morpho(mg, AX[0], DEND_COMP_LIST[args.stim_apic_compartment_index],
                      index=stim_apic_compartment_seg,
                      soma_comp=soma, color=mg.colors[0])
    
    add_dot_on_morpho(mg, AX[0], DEND_COMP_LIST[args.rec2_apic_compartment_index],
                      index=rec2_apic_compartment_seg,
                      soma_comp=soma, color=mg.colors[1])
    
    add_dot_on_morpho(mg, AX[0], COMP_LIST[0],
                      index=0,
                      soma_comp=soma, color=mg.colors[2])

    # plotting
    AX[1].plot(output['t'], output['Vm_syn'], '-', color=mg.colors[0])
    AX[2].plot(output['t'], output['Vm_apic'], '-', color=mg.colors[1])
    AX[3].plot(output['t'], output['Vm_soma'], '-', color=mg.colors[2])
    
    for i, ax, label in zip(range(3), AX[1:], ['synaptic location', 'tuft start', 'soma']):
        mg.set_plot(ax, ['left'], ylabel='mV')
        mg.annotate(ax, '$V_m$ at %s' % label, (1., 0.), ha='right', va='top',
                    color=mg.colors[i])
    Tbar = 20 # ms
    AX[1].plot([output['t'][-1], output['t'][-1]-Tbar], AX[1].get_ylim()[1]*np.ones(2), '-',
               color=mg.default_color)    
    mg.annotate(AX[1], '%sms' % Tbar, (.98, 1.), ha='right', color=mg.default_color)
    # fig.savefig('figures/temp.png')



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
                                             'L5pyr-j140408b.CNG.swc'))
    ###################################################
    # ---------- SIMULATION PARAMS  ----------------- #
    ###################################################
    parser.add_argument("--tstop", help='[ms]', type=float, default=200.)
    parser.add_argument("--dt", help='[ms]', type=float, default=0.1)
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
    parser.add_argument("--tauAMPA", help='[ms]', type=float, default=2.)
    parser.add_argument("--tauRiseNMDA", help='[ms]', type=float, default=3.)
    parser.add_argument("--tauDecayNMDA", help='[ms]', type=float, default=70.)
    parser.add_argument("--V0NMDA", help='[mV]', type=float, default=1./0.08)
    ###################################################
    # ---------- SYNAPTIC PARAMS  ----------------- #
    ###################################################
    parser.add_argument("--stim_apic_compartment_index", type=int, default=10)
    parser.add_argument("--rec2_apic_compartment_index", type=int, default=10)
    
    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                        action="store_true")
    parser.add_argument("-s", "--save", help="save the figures",
                        action="store_true")
    parser.add_argument("--filename", '-f', help="filename",type=str, default='data.npz')

    
    args = parser.parse_args()

    if args.protocol=='run':
        run_sim(args)
    elif args.protocol=='plot':
        pass
    else:
        output = run_sim(args)
        plot_nrn_and_signals(args, output)
        mg.show()
