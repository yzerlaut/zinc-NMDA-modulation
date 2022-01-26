<div><img src="./figures/summary.png" alt="FRM Logo" width="20%" align="right" style="margin-left: 10px"></div>

## Modeling Zinc modulation NMDAR signaling in neocortical neurons

> Source code for the theoretical model in the paper:

> "Activity-dependent modulation of NMDA receptors by endogenous zinc shapes dendritic function in cortical neurons".

> A Morabito, Y Zerlaut, B Serraz, R Sala, P Paoletti, N Rebola. Cell Reports (2022)


[BioRXiv version of the paper](https://www.biorxiv.org/content/10.1101/2021.09.17.460586v1)

## Results and implementation

- we derive a model for the Zinc modulation of NMDAR -> see [description notebook](./description.ipynb)
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

## Model capabilities

### Branco et al., *Neuron* 2011

As a starting point for this investigation, we reproduced Figure 5 of [Branco et al. (2011)](https://www.sciencedirect.com/science/article/pii/S0896627311001036) in our model with the considered morphology:

![](figures/Branco_et_al_2011.png)

See the [branco-et-al-2011.py](./branco-et-al-2011.py) script for the implementation.

## Getting help

In case of questions, you can email: yann.zerlaut [at] icm-institute.org.

If you find a bug, please open a ticket in the [issue tracker](https://github.com/yzerlaut/zinc-NMDA-modulation/issues).

## Acknowledgments

<div><img src="https://www.frm.org/bundles/app/images/logo-header-new2.png" alt="FRM Logo" height="23%" width="15%" align="right" style="margin-left: 10px"></div>

This open source software code was developed thanks to the funding from the [Fondation pour la Recherche Médicale](https://www.frm.org/) under the fellowship agreement ARF201909009117.

