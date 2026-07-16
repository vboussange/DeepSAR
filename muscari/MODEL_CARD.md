---
license: mit
library_name: muscari
tags:
  - ecology
  - biodiversity
  - species-richness
  - species-distribution-modeling
  - vegetation
  - Europe
  - geospatial
pretty_name: MuScaRi
pipeline_tag: other
---

# Model Card for MuScaRi

**MuScaRi** (Multi-Scale species Richness estimation, also named after the *Muscari* genus of perennial bulbous plants) is a deep learning model that estimates vascular plant species richness at arbitrary spatial scales from ecological survey data and environmental covariates.

- **Repository:** https://github.com/vboussange/MuScaRi
- **Paper:** [Multi-scale species richness estimation with deep learning](https://arxiv.org/abs/2507.06358)
- **Data repository:** [vboussange/muscari-data](https://huggingface.co/datasets/vboussange/muscari-data)
- **EVA spatial cross-validation samples:** [`ceacce0`](https://huggingface.co/datasets/vboussange/muscari-data/tree/main/generated_samples/sbcv/ceacce0)
- **GIFT total-species-richness extrapolation samples:** [`418c563`](https://huggingface.co/datasets/vboussange/muscari-data/tree/main/generated_samples/GIFT/418c563)
- **Demo:** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vboussange/MuScaRi/blob/master/muscari_demo.ipynb)

## Model Description

MuScaRi composes a fully connected feedforward neural network with a four-parameter Weibull rarefaction model. Given summary statistics of environmental covariates within a spatial unit, the neural network predicts the parameters of the rarefaction curve, which in turn predicts expected species richness as a function of sampling effort. Evaluating the curve at infinite sampling effort yields total (asymptotic) species richness predictions.

The pretrained model is an **ensemble of 5 members**, one per spatial cross-validation fold, trained on ~350k European vegetation plots from the European Vegetation Archive (EVA). Ensemble predictions are aggregated by arithmetic mean; standard deviations quantify prediction uncertainty.

See the [paper](https://arxiv.org/abs/2507.06358) for full architecture details and benchmarks, and the [`muscari-data` dataset card](https://huggingface.co/datasets/vboussange/muscari-data) for the dataset used during training.

## Quick Start

```python
from muscari import MuScaRiEnsemble
from muscari.data_processing.utils_features import EnvironmentalFeatureDataset
import pandas as pd

model = MuScaRiEnsemble.from_pretrained("vboussange/muscari")
print(f"Ensemble with {model.n_models} members")
print("Required features:", model.feature_names)

# Predict total species richness for a spatial unit
# df must contain columns listed in model.feature_names
df = pd.DataFrame([...])  # one row per spatial unit; see Colab demo for how to build it
sr_mean = model.predict_mean_sr_tot(df)   # asymptotic richness
sr_std  = model.get_std_sr_tot(df)        # ensemble uncertainty
```

For an end-to-end walkthrough, see the [Colab demo](https://colab.research.google.com/github/vboussange/MuScaRi/blob/master/muscari_demo.ipynb).

## Inputs and Outputs

**Inputs:**
a `df: pandas.Dataframe` with the following columns (see [Colab demo](https://colab.research.google.com/github/vboussange/MuScaRi/blob/master/muscari_demo.ipynb) for more details)

| Feature group | Columns | Description |
|---|---|---|
| Sampling effort | `log_observed_area` | Log of sampling effort (m²); omit for asymptotic prediction |
| Mean environmental conditions | mean of `bio1`, `bio12`, `sfcWind`, `pet`, `elevation` | Mean of CHELSA/EU-DEM variables within the spatial unit |
| Environmental heterogeneity | std of `bio1`, `bio12`, `sfcWind`, `pet`, `elevation` | Std of CHELSA/EU-DEM variables within the spatial unit |

**Outputs:**

- `model.predict_mean_sr(df)`: expected species richness at a given sampling effort (interpolation mode)
- `model.predict_mean_sr_tot(df)`: total species richness under asymptotic sampling effort (extrapolation mode)
- `model.get_std_sr_tot(df)`: ensemble standard deviation of the above

## Training Data and Evaluation

NRMSE is RMSE divided by mean observed richness and reported as a percentage. Fold-level values are arithmetic mean ± sample standard deviation across the five spatial cross-validation members. The final row evaluates the uniformly averaged ensemble distributed in this repository.

| Evaluation | RMSE | NRMSE | R² | D² | Median relative bias |
|---|---:|---:|---:|---:|---:|
| 1 km SBCV held-out partitions, fold members | 48.244 ± 1.916 | 18.166 ± 1.028% | 0.987 ± 0.001 | 0.853 ± 0.007 | 0.031 ± 0.009 |
| GIFT asymptotic extrapolation, fold members | 461.880 ± 69.704 | 29.473 ± 4.448% | 0.721 ± 0.088 | 0.495 ± 0.070 | 0.052 ± 0.042 |
| GIFT asymptotic extrapolation, published uniform ensemble | 409.584 | 26.136% | 0.784 | 0.543 | 0.066 |

The spatial cross-validation evaluation uses the uploaded [`ceacce0` partitions](https://huggingface.co/datasets/vboussange/muscari-data/tree/main/generated_samples/sbcv/ceacce0). The GIFT evaluation uses the uploaded [`418c563` dataset](https://huggingface.co/datasets/vboussange/muscari-data/tree/main/generated_samples/GIFT/418c563); metrics are calculated on the 178 spatial units with complete model inputs. The fold-level GIFT row uses true asymptotic predictions, while the ensemble row evaluates `predict_mean_sr_tot` after uniform aggregation. Full comparison tables are available in the [paper](https://arxiv.org/abs/2507.06358).

## Limitations

- Trained on European vascular plants; performance outside Europe is untested.
- Environmental predictors use a 1981-2010 climatological baseline.
- Predictions are less reliable in data-sparse regions (e.g. parts of France, Spain, Scandinavia).

## Citation

```bibtex
@misc{boussange2025muscari,
  title         = {Multi-scale species richness estimation with deep learning},
  author        = {Victor Boussange and Bert Wuyts and Philipp Brun and
                   Johanna T. Malle and Gabriele Midolo and Jeanne Portier and
                   Théophile Sanchez and Niklaus E. Zimmermann and
                   Irena Axmanová and Helge Bruelheide and Milan Chytrý and
                   Stephan Kambach and Zdeňka Lososová and Martin Večeřa and
                   Idoia Biurrun and Klaus T. Ecker and Jonathan Lenoir and
                   Jens-Christian Svenning and Dirk Nikolaus Karger},
  year          = {2025},
  eprint        = {2507.06358},
  archivePrefix = {arXiv},
  primaryClass  = {q-bio.PE},
  url           = {https://arxiv.org/abs/2507.06358},
}
```
