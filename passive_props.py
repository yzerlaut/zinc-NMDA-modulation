import os
import numpy as np

import neural_network_dynamics.main as ntwk # my custom layer on top of Brian2
from datavyz import ge

from scipy.optimize import minimize

import sys, pathlib
sys.path.append(os.path.join(str(pathlib.Path(__file__).resolve().parents[1]), 'cortical-physio-icm'))
from electrophy.intracellular.passive_props import perform_ICcharact


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


def run_model(gl, cm, debug=False):

    Model["gL"] = gl
    Model["cm"] = cm
    output = run_voltage_clamp_protocol(Model,
                                        clamps = [{'value':0., 'duration':50},
                                                  {'value':200, 'duration':100},
                                                  {'value':0., 'duration':50}],
                                        with_plot=debug)
    if debug:
        from datavyz import ges as ge
        fig, ax, Rm, Cm = perform_ICcharact(1e-3*output['t'], 1e-3*output['Vm_soma'],
                                   t0=50e-3, t1=150e-3, with_plot=True, ge=ge)
        ge.show()
    else:
        Rm, Cm = perform_ICcharact(1e-3*output['t'],
                                   1e-3*output['Vm_soma'],
                                   t0=50e-3, t1=150e-3)
        
    return Rm, Cm


def find_best_membrane_params(Rm, Cm, debug=False):

    def to_minimize(coefs):
        RmT, CmT = run_model(*coefs,debug=debug)
        return np.abs(Rm-RmT)/Rm*np.abs(Cm-CmT)/Cm
        
    res = minimize(to_minimize,
                   [1., .8], method='SLSQP', bounds=[(0.1,10.),
                                                     (0.5,2.)])
    print(res)
    
    return res.x

import itertools
def grid_search(Rm, Cm, N=20,
                gl0=0.1, gl1=10.,
                cm0=0.5, cm1=2., debug=False):

    RMS, CMS, gLS, cMS = [], [], [], []
    
    for gl, cm in itertools.product(np.linspace(gl0, gl1, N),
                                    np.linspace(cm0, cm1, N)):
        RmT, CmT = run_model(gl, cm, debug=debug)
        RMS.append(RmT)
        CMS.append(CmT)
        gLS.append(gl)
        cMS.append(cm)

    imin = np.argmin(np.abs(np.array(RMS)-Rm)/Rm*np.abs(np.array(CMS)-Cm)/Cm)
    return gLS[imin], cMS[imin]
    
    
if __name__=='__main__':

    import sys
    from model import Model



        
    if sys.argv[1]=='calib':

        Rm=float(sys.argv[2])*1e6
        Cm=float(sys.argv[3])*1e-12
        N=int(sys.argv[4])
        print(Rm, Cm, N)
        gl_best, cm_best = grid_search(Rm, Cm, N=N)
        print('gl_best=%.2f, cm_best=%.2f' % (gl_best, cm_best))
        RmB, CmB = run_model(gl_best, cm_best)
        print('gives Rm=%.2fMOhm, Cm=%.2fpF' % (1e-6*RmB, 1e12*CmB))
        np.savez('data/passive-props.npz', **{'gl':gl_best, 'cm':cm_best})

        
    # gl_best, cm_best = find_best_membrane_params(Rm, Cm, debug=False)
    # print('gl_best=%.2f, cm_best=%.2f' % (gl_best, cm_best))
    # RmB, CmB = run_model(gl_best, cm_best)
    # print('gives Rm=%.2fMOhm, Cm=%.2fpF' % (1e-6*RmB, 1e12*CmB))

