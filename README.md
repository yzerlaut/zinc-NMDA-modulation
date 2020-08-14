# Modeling Zinc-mediated modulation of synaptic transmission through NMDA receptors in cortical neurons

Source code for the theoretical model of zinc modulation of NMDA-dependent synaptic transmission in neocortical neurons. Implementing stochastic synaptic activity emulating _in-vivo_-like ongoing dynamics in neocortical networks. Analyzing the impact on cellular computation under 

## Model description

| ![description]('./figures/fig-model-description.png) |
|:---------------------------------------------------------------------------------:|
| **Modeling Zinc modulation of NMDAR signaling in Layer 2/3 pyramidal cells.**. **(a)** Biophysical model of NMDAR Zinc modulation at the single synapse level. Glutamate release is translated into a state-dependent integration variable $x$ (the $x$-dependency of the $\Delta x$ increments is shown in the inset, grey line). The $x$ variable is then non-linearly transformed and low-pass filtered to give the Zinc-binding variable $b_{Zn}$ (see the sigmoid in the inset, black line). Zinc binding reduces the NMDA synaptic conductance ($g_{NMDA}$, top plot), at full-binding $b_{Zn}=1$, the conductance is reduced by a factor $\alpha_{Zn}$. The increment parameter of the model $\Delta x^0$ is set to 0 when there is no available Zinc ("chelated-Zinc", green lines). **(b)** Morphological reconstruction of a Layer 2/3 pyramidal in the primary sensory cortex of the mouse (Jiang et al. (2015)). We highlight the soma (red circle) and the 5 glutamergic synapses on the basal dendite (orange dots) of panel *c*. **(c)** Voltage-clamp recordings in the model (see Methods) at four holding potentials following the synchronous stimulation of 5 glutamatergic synapses at 20Hz. We show the membrane potential (top plot, at the soma in red and at one synapse location in orange), the evoked conductances in one synapse (middle plot, AMPA and NMDA) and the recorded current at the soma (bottom). The green curves show the "chelated-Zinc" case ($\Delta_x^0$=0). |



## Requirements

- `python` (use a python distribution for scientific computing, e.g. Anaconda)

- `brian2`: a simulator of single cell computation and network dynamics, get it with `pip install brian2`

## Cellular morphology dataset

We make use of the publicly available dataset of cellular morphologies taken from the following study:

Jiang et al., _Science_ (2015): Principles of connectivity among morphologically defined cell types in adult neocortex https://science.sciencemag.org/content/350/6264/aac9462

<!-- Set of morphologies (dendritic arborization in red, axonal projections in green): -->
<!-- ![](figures/all_cells.png) -->

## Model capabilities

### - Branco et al., *Neuron* 2011

Trying to reproduce Figure 5 of [Branco et al. (2011)](https://www.sciencedirect.com/science/article/pii/S0896627311001036) in our model with the considered morphology:

![](figures/Branco_et_al_2011.png)

see the [branco-et-al-2011.py](./branco-et-al-2011.py) script.


## cellular biophysics: implementation

Analogous to Farinella et al. _PLoS Comp Biol_:

```
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
```

## NMDA-dependent excitatory synaptic transmission

Post-synaptic potentials at -70mV (5 synchronous synaptic events):

![](figures/PSP_at_rest.png)

Post-synaptic potentials at -50mV (5 synchronous synaptic events):

![](figures/PSP_at_depol_level.png)

## Background synaptic activity

Adding stochastic background synaptic activity on top of the evoked (synchronous) synaptic event(s):

N.B. We restrict the synaptic activity to the apical tuft.

No background:

![](figures/no_bg.png)

With excitatoy background only (10 synapses at Fe=15Hz):

![](figures/with_exc_bg_15.png)

With excitatoy background only (10 synapses at Fe=25Hz):

![](figures/with_exc_bg_25.png)

With excitatoy-inhibitory background activity (10 synapses each, Fe=25Hz, Fi=10Hz):

![](figures/with_bg_25_10.png)

With excitatoy-inhibitory background activity (10 synapses each, Fe=25Hz, Fi=15Hz):

![](figures/with_bg_25_15.png)

With excitatoy-inhibitory background activity (10 synapses each, Fe=60Hz, Fi=150Hz):

![](figures/with_bg_60_150.png)


## Acknowledgments

[...]

contact: yann.zerlaut@cnrs.fr
