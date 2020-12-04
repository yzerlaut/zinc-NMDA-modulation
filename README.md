# Modeling Zinc modulation NMDA receptor signalling in neocortical neurons

> Source code for the theoretical model of zinc modulation of synaptic transmission through NMDA receptors (NMDAR) in neocortical neurons

- we derive a model for the Zinc modulation of NMDAR
- we embed the implementation into a morphologically-detailed model of a mouse layer 2/3 pyramidal cell
- we study the response to synaptic stimulation
- we analyze the import of *in-vivo*-like ongoing dynamics on the modulation of Zinc

## Scientific content

Available as a [pdf report](./paper.pdf) (source file available as a [text file](./paper.txt)).

## Software requirements

- A `python` distribution for scientific computing (e.g. Anaconda)

- `brian2`: a simulator of single cell and network dynamics, see the [Brian2 documentation](https://brian2.readthedocs.io/en/stable/), get it with `pip install brian2`.

All other dependencies are listed in [requirements.txt](./requirements.txt), install them with:
```
pip install -r requirements.txt
```

## Cellular morphology dataset

We make use of the publicly available dataset of cellular morphologies taken from the following study:

Jiang et al., _Science_ (2015): Principles of connectivity among morphologically defined cell types in adult neocortex https://science.sciencemag.org/content/350/6264/aac9462

<!-- Set of morphologies (dendritic arborization in red, axonal projections in green): -->
<!-- ![](figures/all_cells.png) -->

## Model capabilities

### - Branco et al., *Neuron* 2011

We reproduce below Figure 5 of [Branco et al. (2011)](https://www.sciencedirect.com/science/article/pii/S0896627311001036) in our model with the considered morphology:

![](figures/Branco_et_al_2011.png)

See the [branco-et-al-2011.py](./branco-et-al-2011.py) script for the implementation.

## Acknowledgments

Research funded by a  fellowship from the [Fondation pour la Recherche Médicale](https://www.frm.org/).

contact: (yann [dot] zerlaut [at] icm-institute [dot] org)[yann.zerlaut@icm-institute.org]
