# Deep-learning-based species area relationship model
Code and model weights for predicting vegetation species richness across Europe as a function of sampling area and climate.

## Installation

### Environment
Two environments are necessary to run the scripts, specified in `environment-cudf.yml` (for megaplot compilation) and in `environment-torch.yml` (for cross-validation and training).


To install and activate e.g. the `torch` environment, make sure you have conda (or mamba) and 

```
mamba env create --prefix ./.env-torch --file environment-torch.yml
mamba activate ./.env-torch
pip install -e .
```

### Data
Three datasets are required for generating the training dataset, located in `data`.

- `CHELSA/`: *Climate data.* Go to the folder and type

```
wget --no-host-directories --force-directories --input-file=envidat.txt
```

 - `NaturealEarth/` (*Used for filtering EVA plots.*) Go to the folder and type

```
wget -q https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip -O ne_10m_admin_0_countries.zip && unzip -q ne_10m_admin_0_countries.zip
```

- `EVA/` (*Vegetation plot data*). Make a request at https://euroveg.org/eva-database/. You should obtain a `hea_all.csv` and a `vpl_all.csv`, to be placed in this folder.


## Quick start
### Data pre-processing
1. Construct megaplots with `scripts/eva_chelsa_processing/compile_eva_chelsa_megaplots.py`
2. Preprocess megaplots with  `scripts/eva_chelsa_processing/preprocess_eva_chelsa_megaplots.py`
    - Creates a train / validation / test dataset based on outputs of `eva_chelsa_processing` for each habitat considered and saves it as a separate file 
    <!-- (TODO: not clean, could be avoided by refactoring `scripts/eva_chelsa_processing.py`) -->

### Cross-validation
`cross_validate.py` performs a k-folded spatial block cross validation.

### Training
`train.py` trains neural net SAR for each habitat considered.

## Model weights and SR maps
Located in `maps/`.
