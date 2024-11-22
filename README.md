# Neural species area relationship model

## Installation
Two environments are necessary to run the scripts, specified in `environment-cudf.yml` (for megaplot compilation) and in `environment-torch.yml` (for cross-validation and training).


To install and activate e.g. the `torch` environment, make sure you have conda (or mamba) and 

```
mamba env create --prefix ./.env-torch --file environment-torch.yml
mamba activate ./.env-torch
```


## Quick start
### Data pre-processing
1. Construct megaplots with `scripts/eva_chelsa_processing.py`
2. Preprocess megaplots with  `scripts/preprocess_eva_chelsa_megaplots.py`
    - Creates a train / validation / test dataset based on outputs of `eva_chelsa_processing` for each habitat considered and saves it as a separate file (TODO: not clean, could be avoided by refactoring `scripts/eva_chelsa_processing.py`)

### Cross-validation
`cross_validate.py` performs a k-folded spatial block cross validation.

### Training
`train.py` trains neural net SAR for each habitat considered.

### TODO

#### SI
- plot location of EVA plots