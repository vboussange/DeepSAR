[![arXiv](https://img.shields.io/badge/arXiv-2507.06358-b31b1b.svg)](https://arxiv.org/abs/2507.06358)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vboussange/DeepSAR/blob/master/deepsar_demo.ipynb)

# DeepSAR: deep learning-based species-area relationship model
Official implementation of the paper

> **Deep learning-based species-area models reveal multi-scale patterns of species richness and turnover**  
> Victor Boussange, Philipp Brun, Johanna T. Malle, Gabriele Midolo, Jeanne Portier, Théophile Sanchez, Niklaus E. Zimmermann, Irena Axmanová, Helge Bruelheide, Milan Chytrý, Stephan Kambach, Zdeňka Lososová, Martin Večeřa, Idoia Biurrun, Klaus T. Ecker, Jonathan Lenoir, Jens-Christian Svenning, Dirk Nikolaus Karger. arXiv: [2507.06358](https://arxiv.org/abs/2507.06358) (2025)

<!-- TODO: place GIF animation -->

## Quick start
### Inference
We provide a self-contained tutorial to predict species richness maps from pretrained deep SAR model weights: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vboussange/DeepSAR/blob/master/deepsar_demo.ipynb)
<!-- TODO: host .nc CHELSA dataset -->

### Training
To retrain the deep SAR model, you'll need to follow these steps.

0. Make sure you have the (vegetation) plot data and predictor variables under `data/`, and install [the project environment](#environment).
1. Generate training data with `scripts/data_processing/compile_eva_chelsa.py` (you can generate test data with `scripts/data_processing/compile_gift_chelsa.py`)
2. Train an ensemble model from the training data with `train.py`. The current deep SAR model, `deep4pweibull`, is defined under `deepsar/deep4pweibull.py`. 
3. Make predictions with `project.py` (seel also [Inference](#inference)).


### Environment
To install the dependencies and load the environment, make sure you have conda (or mamba) and 

```
uv sync
uv pip install torch --torch-backend=auto
uv pip install -e .
```

### Data
#### European Vegetation Archive (EVA) dataset
Anonymised vegetation plot data is located at `data/processed/EVA/anonymised`. It consists of two `.parquet` files:
- `plot_data.parquet` contains the metadata associated with the vegatation plots
- `species_data.parquet` contains anoynimised species names associated with each plot.

To obtain the full dataset, make a request at https://euroveg.org/eva-database/.

#### GIFT database
As a test dataset, we provide data retrieved from the [GIFT database](https://gift.uni-goettingen.de/home) and harmonized with the EVA dataset, located under `data/processed/GIFT/anonymised`. 


#### Predictors
As predictors, we currently bioclimate variables from the CHELSA dataset. To download the same bioclimate variables, go to `data/CHELSA/` and type

```
wget --no-host-directories --force-directories --input-file=envidat.txt
```


# Citations
If you use the anonymised data, please cite the following references:

```
@misc{boussange2025,
      title={Deep learning-based species-area models reveal multi-scale patterns of species richness and turnover}, 
      author={Victor Boussange and Philipp Brun and Johanna T. Malle and Gabriele Midolo and Jeanne Portier and Théophile Sanchez and Niklaus E. Zimmermann and Irena Axmanová and Helge Bruelheide and Milan Chytrý and Stephan Kambach and Zdeňka Lososová and Martin Večeřa and Idoia Biurrun and Klaus T. Ecker and Jonathan Lenoir and Jens-Christian Svenning and Dirk Nikolaus Karger},
      year={2025},
      eprint={2507.06358},
      archivePrefix={arXiv},
      primaryClass={q-bio.PE},
      url={https://arxiv.org/abs/2507.06358}, 
}
```

```
@article{weigelt2020,
author = {Weigelt, Patrick and König, Christian and Kreft, Holger},
title = {GIFT – A Global Inventory of Floras and Traits for macroecology and biogeography},
journal = {Journal of Biogeography},
volume = {47},
number = {1},
pages = {16-43},
doi = {https://doi.org/10.1111/jbi.13623},
eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1111/jbi.13623},
year = {2020}
}
```
```
@article{chytry2016,
  title = {European {{Vegetation Archive}} ({{EVA}}): An Integrated Database of {{European}} Vegetation Plots},
  shorttitle = {European {{Vegetation Archive}} ({{EVA}})},
  author = {Chytr{\'y}, Milan and Hennekens, Stephan M. and {Jim{\'e}nez-Alfaro}, Borja and Knollov{\'a}, Ilona and Dengler, J{\"u}rgen and Jansen, Florian and Landucci, Flavia and Schamin{\'e}e, Joop H.J. and A{\'c}i{\'c}, Svetlana and Agrillo, Emiliano and Ambarl{\i}, Didem and Angelini, Pierangela and Apostolova, Iva and Attorre, Fabio and Berg, Christian and Bergmeier, Erwin and Biurrun, Idoia and {Botta-Duk{\'a}t}, Zolt{\'a}n and Brisse, Henry and Campos, Juan Antonio and Carl{\'o}n, Luis and {\v C}arni, Andra{\v z} and Casella, Laura and Csiky, J{\'a}nos and {\'C}u{\v s}terevska, Renata and Daji{\'c} Stevanovi{\'c}, Zora and Danihelka, Ji{\v r}{\'i} and De Bie, Els and {de Ruffray}, Patrice and De Sanctis, Michele and Dickor{\'e}, W. Bernhard and Dimopoulos, Panayotis and Dubyna, Dmytro and Dziuba, Tetiana and Ejrn{\ae}s, Rasmus and Ermakov, Nikolai and Ewald, J{\"o}rg and Fanelli, Giuliano and {Fern{\'a}ndez-Gonz{\'a}lez}, Federico and FitzPatrick, {\'U}na and Font, Xavier and {Garc{\'i}a-Mijangos}, Itziar and Gavil{\'a}n, Rosario G. and Golub, Valentin and Guarino, Riccardo and Haveman, Rense and Indreica, Adrian and I{\c s}{\i}k G{\"u}rsoy, Deniz and Jandt, Ute and Janssen, John A.M. and Jirou{\v s}ek, Martin and K{\k a}cki, Zygmunt and Kavgac{\i}, Ali and Kleikamp, Martin and Kolomiychuk, Vitaliy and Krstivojevi{\'c} {\'C}uk, Mirjana and Krstono{\v s}i{\'c}, Daniel and Kuzemko, Anna and Lenoir, Jonathan and Lysenko, Tatiana and Marcen{\`o}, Corrado and Martynenko, Vassiliy and Michalcov{\'a}, Dana and Moeslund, Jesper Erenskjold and Onyshchenko, Viktor and Pedashenko, Hristo and {P{\'e}rez-Haase}, Aaron and Peterka, Tom{\'a}{\v s} and Prokhorov, Vadim and Ra{\v s}omavi{\v c}ius, Valerijus and {Rodr{\'i}guez-Rojo}, Maria Pilar and Rodwell, John S. and Rogova, Tatiana and Ruprecht, Eszter and R{\=u}si{\c n}a, Solvita and Seidler, Gunnar and {\v S}ib{\'i}k, Jozef and {\v S}ilc, Urban and {\v S}kvorc, {\v Z}eljko and Sopotlieva, Desislava and Stan{\v c}i{\'c}, Zvjezdana and Svenning, Jens-Christian and Swacha, Grzegorz and Tsiripidis, Ioannis and Turtureanu, Pavel Dan and U{\u g}urlu, Emin and Uogintas, Domas and Valachovi{\v c}, Milan and Vashenyak, Yulia and Vassilev, Kiril and Venanzoni, Roberto and Virtanen, Risto and Weekes, Lynda and Willner, Wolfgang and Wohlgemuth, Thomas and Yamalov, Sergey},
  year = {2016},
  journal = {Applied Vegetation Science},
  volume = {19},
  number = {1},
  pages = {173--180},
  issn = {1654-109X},
  doi = {10.1111/avsc.12191},
  urldate = {2024-05-17},
}
```