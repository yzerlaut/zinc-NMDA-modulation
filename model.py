import numpy as np

from analyz.IO.npz import load_dict
def add_value_to_study_data(key, value, study='study.npz'):
    Model2 = load_dict(study)
    Model2[key] = value
    np.savez(study, **Model2)
    

Model = {
    ############ - morpho - ##########################
    'morpho_file':'neural_network_dynamics/single_cell_integration/morphologies/Jiang_et_al_2015/L23pyr-j150407a.CNG.swc',  
    # 'morpho_file':'data/L23pyr-basal-only.swc',  
    'morpho_file_1':'neural_network_dynamics/single_cell_integration/morphologies/Jiang_et_al_2015/L23pyr-j150407a.CNG.swc',
    'morpho_file_2':'neural_network_dynamics/single_cell_integration/morphologies/Jiang_et_al_2015/L23pyr-j150811a.CNG.swc',
    ##################################################
    # ---------- BIOPHYSICAL PROPS ----------------- #
    ##################################################
    "gL": 0.29, # [pS/um2] = 10*[S/m2] # FITTED --- Farinella et al. 0.5pS/um2 = 0.5*1e-12*1e12 S/m2, NEURON default: 1mS/cm2 -> 10pS/um2
    "cm": 0.91, # [uF/cm2] FITTED
    "Ri": 100., # [Ohm*cm]'
    "EL": -75, # [mV]
    #################################################
    # ---------- SYNAPTIC PARAMS  ----------------- #
    #################################################
    'Ee':0,# [mV]
    'Ei':-80,# [mV]
    'qAMPA':1.,# [nS] # Destexhe et al., 1998: "0.35 to 1.0 nS"
    'qNMDA':1.*2.7,# [nS] # NMDA-AMPA ratio=2.7
    'qGABA':1.,# [nS] # Destexhe et al., 1998: "0.25 to 1.2 nS"
    'tauRiseAMPA':0.5,# [ms], Destexhe et al. 1998: 0.4 to 0.8 ms
    'tauDecayAMPA':5,# [ms], Destexhe et al. 1998: "the decay time constant is about 5 ms (e.g., Hestrin, 1993)"
    'tauRiseGABA':0.5,# [ms] Destexhe et al. 1998
    'tauDecayGABA':5,# [ms] Destexhe et al. 1998
    'tauRiseNMDA': 3,# [ms], Farinella et al., 2014
    'tauDecayNMDA': 70,# [ms], FITTED --- Destexhe et al.:. 25-125ms, Farinella et al., 2014: 70ms
    'DensityGlut_L23': 1.84, # synapses / 100um2
    'DensityGlut_L4': 1.42, # synapses / 100um2
    'DensityAMPA': 25, # synapses / 100um2
    'DensityNMDA': 25, # synapses / 100um2
    'DensityGABA_dend': 5, # synapses / 100um2
    'DensityGABA_soma': 15, # synapses / 100um2
    'DensityGABA': 15, # synapses / 100um2
    ###################################################
    # ---------- SIMULATION PARAMS  ----------------- #
    ###################################################
    'tstop':600, # [ms]
    'dt':0.025,# [ms]
    'seed':1, #
    'VC-gL-reduction-factor':5, # 
    'VC-discard-time':200, # [ms]
    'VC-gclamp':1,# [uS]
    'VC-cmd':20,# [mV]
    #################################################
    # ---------- MG-BLOCK PARAMS  ----------------- #
    #################################################
    'cMg': 1., # mM
    'etaMg': 0.33, # 1/mM
    'V0NMDA':1./0.08,# [mV]
    'Mg_NMDA':1.,# mM
    ########################################################
    # ---------- ZINC MODULATION PARAMS  ----------------- #
    ########################################################
    # 'alphaZn':0.19,# FITTED
    'alphaZn':0.35,# FITTED
    'tauRiseZn':150.,# [ms], # FITTED
    'tauDecayZn':638.,# [ms], # FITTED
    'Deltax0':1., # FITTED
    # 'x0':0.5, # FITTED
    # 'deltax':0.1, # FITTED
    #############################################################
    # ----------- MODEL CALIBRATION GRIDS  ----------------- ####
    #############################################################
    #### --- defining the grid extents for parameter searches ###
    'gL_min':0.02, 'gL_max':2., 'N_gL':30,
    'cm_min':0.5, 'cm_max':2., 'N_cm':30,
    'tauDecayNMDA_min':60, 'tauDecayNMDA_max':120, 'N_tauDecayNMDA':7,
    'Nsyn1_min':3, 'Nsyn1_max':11, 'N_Nsyn1':5,
    'Nsyn2_min':2, 'Nsyn2_max':10, 'N_Nsyn2':5,
    'Tnsyn20Hz_min':30, 'Tnsyn20Hz_max':70, 'N_Tnsyn20Hz':4,
    'Tnsyn3Hz_min':70, 'Tnsyn3Hz_max':1500, 'N_Tnsyn3Hz':4,
    'alphaZn_min':0.1, 'alphaZn_max':0.6, 'N_alphaZn':30,
    'tauDecayZn_min':100, 'tauDecayZn_max':1300, 'N_tauDecayZn':30,
    # 'tauRiseZn_min':5, 'tauRiseZn_max':300, 'N_tauRiseZn':7,
    # 'Deltax0_min':0.3, 'Deltax0_max':0.8, 'N_Deltax0':6,
    # 'x0_min':0.3, 'x0_max':0.8, 'N_x0':6,
    # 'deltax_min':0.05, 'deltax_max':0.15, 'N_deltax':3,
    #############################################################
    # ---------- SYNAPTIC STIMULATION PARAMS  ----------------- #
    #############################################################
    # evoked
    'stim_apic_compartment_index':1000, # type=int, default=1652)
    'Nsyn_synch_stim':5,# 
    'tsyn_stim':10., # [ms]
    # bg
    'Nsyn_Ebg':0, # syn.
    'Nsyn_Ibg':0, #
    'Fexc_bg':0.,# [Hz]
    'Finh_bg':0., # [Hz]
    # Increasing Synaptic Input on Branches
    'ISIB_Nsyn1':1,
    'ISIB_Nsyn2':80,
    'ISIB_Nsyn_N':10,
    'ISIB_log_Nsyn':True,
    'ISIB_delay':350, # [ms]
    'ISIB_window':500, # [ms]
    'ISIB_before':100, # [ms]
    'branch_index':0,
}    

def double_exp_normalization(T1, T2):
    return T1/(T2-T1)*((T2/T1)**(T2/(T2-T1)))

Model['nAMPA'] = double_exp_normalization(Model['tauRiseAMPA'],Model['tauDecayAMPA'])    
Model['nGABA'] = double_exp_normalization(Model['tauRiseGABA'],Model['tauDecayGABA'])    
Model['nNMDA'] = double_exp_normalization(Model['tauRiseNMDA'],Model['tauDecayNMDA'])

if __name__=='__main__':
    np.savez('study.npz', **Model)
