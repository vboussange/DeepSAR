---
license: mit
task_categories:
  - other
tags:
  - ecology
  - biodiversity
  - species-richness
  - species-distribution-modeling
  - vegetation
  - Europe
  - geospatial
pretty_name: MuScaRi Data
---

# Dataset Card for `muscari-data`

`muscari-data` is the companion dataset for **MuScaRi** (Multi-Scale species Richness estimation, also named after the *Muscari* genus of perennial bulbous plants), as proposed in the paper [Multi-scale species richness estimation with deep learning](https://arxiv.org/abs/2507.06358) by Boussange et al.

It bundles ~350k anonymized European vegetation plots (EVA), 184 independent regional plant inventories (GIFT), and gridded environmental predictors, enabling development and evaluation of methods that estimate species richness and related diversity metrics from opportunistic small-scale ecological surveys.

Species names are fully anonymized; it is not possible to reconstruct the original taxonomy from the released files alone.

- **Curated by:** Victor Boussange (WSL / ETH Zürich)
- **Repository:** https://github.com/vboussange/MuScaRi
- **Paper:** [Multi-scale species richness estimation with deep learning](https://arxiv.org/abs/2507.06358)
- **Pretrained model:** [vboussange/muscari](https://huggingface.co/vboussange/muscari)
- **Demo:** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vboussange/MuScaRi/blob/master/muscari_demo.ipynb)

## Dataset Structure

The dataset bundles four interoperable components:

| Component | Description | Format |
|-----------|-------------|--------|
| **EVA** | Anonymized presence-absence matrix derived from the European Vegetation Archive (EVA). Each row is a vegetation plot; each column is an anonymized plant species. | Parquet |
| **GIFT** | Anonymized presence-absence matrix derived from the Global Inventory of Floras and Traits (GIFT). Each row is a regional flora (polygon); each column is an anonymized plant species. Used as an independent out-of-distribution evaluation set. | Parquet |
| **Environmental Features** | Gridded environmental predictors over Europe: 19 CHELSA v2 bioclimatic variables (1981–2010), EEA Digital Elevation Model (1 km), and Corine Land Cover 2018. | NetCDF |
| **Generated Samples** | Model-ready spatial cross-validation partitions and the compiled GIFT evaluation table used for the published pretrained ensemble. | GeoParquet / JSON |

Together these components provide the inputs and ground-truth supervision to train and evaluate MuScaRi, and to generate continent-scale species-richness projections.

## Dataset creation

### Sources

#### EVA: European Vegetation Archive

The EVA component is derived from the **European Vegetation Archive** (Chytrý et al., 2016; version 2023-02-04, project 172), a coordinated repository of vegetation-plot records contributed by national databases. The raw dataset contains 502,724 vegetation plots; after filtering for plots with available area information, coordinate uncertainty < 1 km, land location, recording year ≥ 1972, and removing non-vascular taxa, approximately **352,000 plots** covering **~8,500 distinct vascular plant species** across 44 European countries are retained. 
The original dataset has restricted access; it is archived at DOI [10.58060/d1bp-fp47](https://doi.org/10.58060/d1bp-fp47) and is available upon request at the [EVA database](https://euroveg.org/eva-database). The version released here has been:

1. **Sanitized**: plots with unreliable coordinates, missing metadata, or implausible species counts removed; non-vascular taxa and pre-1972 records excluded.
2. **Anonymized**: original taxonomic names replaced with consistent anonymized tokens shared between EVA and GIFT, so that the dataset can be distributed freely without violating EVA data-sharing restrictions.

Geographic coverage: continental Europe and Iceland (~10°W – 40°E, 34°N – 72°N).

#### GIFT: Global Inventory of Floras and Traits

The GIFT component is derived from **GIFT** (Weigelt et al., 2020), a global database of regional plant checklists and trait information. Inventories were filtered to those falling within the geographic range of the EVA plots, yielding **184 exhaustive species surveys**. GIFT polygons mostly correspond to countries or administrative regions (median area ~11,700 km²) and serve as an independent out-of-distribution benchmark for evaluating total species richness under asymptotic sampling effort. Species are aligned to the EVA taxonomic namespace via the same anonymization procedure, so that anonymized species tokens are consistent between the two matrices. Scripts for downloading the raw GIFT dataset are provided in the [MuScaRi GitHub repository](https://github.com/vboussange/MuScaRi) under `data/raw/GIFT`.

#### Environmental Features

| Source | Variables | Resolution | CRS |
|--------|-----------|------------|-----|
| [CHELSA v2](https://chelsa-climate.org/) | All available bioclimatic variables (incl. BIO1–BIO19, sfcWind, pet), 1981–2010 climatology | ~1 km | EPSG:3035 |
| [EEA Digital Elevation Model (EU-DEM)](https://ec.europa.eu/eurostat/web/gisco/geodata/digital-elevation-model/eu-dem) | Elevation | 30m → resampled to 1 km | EPSG:3035 |
| [Corine Land Cover 2018](https://land.copernicus.eu/pan-european/corine-land-cover) | Land-cover class (remapped to consecutive integers) | 100 m → resampled to 1 km | EPSG:3035 |

All rasters are reprojected and resampled to a common 1 km grid in the ETRS89-LAEA projection (EPSG:3035) before upload. The MuScaRi model uses the mean and standard deviation of four variables computed within each spatial unit (mean annual temperature `bio1`, annual precipitation `bio12`, near-surface wind speed `sfcWind`, and potential evapotranspiration `pet`), together with elevation, as its environmental features. Scripts for downloading the original datasets are provided in the [MuScaRi GitHub repository](https://github.com/vboussange/MuScaRi) under `data/raw`.

### Curation

Preprocessing and anonymization scripts are available in the [MuScaRi GitHub repository](https://github.com/vboussange/MuScaRi) under `scripts/data_processing/`:

- `eva_preprocessing.py`: sanitizes raw EVA data (coordinate checks, area filters, taxon exclusions).
- `gift_preprocessing.py`: sanitizes raw GIFT data.
- `anonymise_gift_eva.py`: harmonizes species names between EVA and GIFT, then replaces them with consistent anonymized tokens.
- Environmental feature rasters are compiled and resampled to the 1 km EPSG:3035 grid by `muscari/data_processing/utils_features.py`.

## Data Fields

#### EVA / GIFT `species_matrix.parquet`

| Column | Type | Description |
|--------|------|-------------|
| `geometry` | `POINT` / `POLYGON` (WKB) | Plot centroid (EVA, EPSG:3035) or region polygon (GIFT, EPSG:3035) |
| `area_m2` | `float32` | Plot or region area in m² |
| `<anonymised_species_name>` | `bool` | Presence (`True`) / absence (`False`) for each species |

Species column names are anonymized tokens (e.g. `sp_00042`) that are consistent across the EVA and GIFT matrices; the original taxonomy is not exposed.

#### `chelsa_dem_cache.nc`

| Variable | Unit | Description |
|----------|------|-------------|
| `bio01` – `bio19` | varies | CHELSA v2 bioclimatic variables (1981–2010 climatology) |
| `elevation` | m | EEA Digital Elevation Model |

#### `landcover_cache.nc`

| Variable | Unit | Description |
|----------|------|-------------|
| `landcover` | categorical (int16) | Corine Land Cover 2018 class code |

The legacy cache retains source background codes `0` and `999` as remapped
categories. The reported MuScaRi benchmarks and released ensemble do not use
land-cover fractions as predictors.

#### Generated samples

| Path | Description |
|------|-------------|
| `generated_samples/sbcv/ceacce0/` | Five spatial cross-validation folds, each containing training, validation, and test GeoParquet partitions plus summary metadata. Spatial block groups use a 1 km block size. |
| `generated_samples/GIFT/418c563/compiled_data.parquet` | GIFT total-species-richness extrapolation samples with richness targets, geometry, and environmental features. |
| `generated_samples/GIFT/418c563/dataset_summary.json` | Schema, sample count, environmental variables, and value ranges for the compiled GIFT table. |

### Data Splits

The EVA species matrix is converted into model-ready richness samples by the procedure described in the [paper](https://arxiv.org/abs/2507.06358) (Methods, Section "Sample generation and spatial block cross-evaluation procedure") and implemented in the [MuScaRi GitHub repository](https://github.com/vboussange/MuScaRi) (`scripts/data_processing/compile_sbcv_eva_samples.py`). Dataset [`ceacce0`](https://huggingface.co/datasets/vboussange/muscari-data/tree/main/generated_samples/sbcv/ceacce0) provides the resulting five-fold spatial cross-validation partitions used to train and evaluate the pretrained ensemble.

The GIFT dataset, consisting of exhaustive regional plant inventories, serves as an independent out-of-distribution benchmark: the total species richness recorded for each entry — corresponding to a country or administrative region — is treated as a ground-truth estimate of total species richness under asymptotic sampling effort. Dataset [`418c563`](https://huggingface.co/datasets/vboussange/muscari-data/tree/main/generated_samples/GIFT/418c563) provides the total-species-richness extrapolation samples.


## Quick Start

```python
from muscari.data_processing.utils_eva import EVADataset
from muscari.data_processing.utils_gift import GIFTDataset
from muscari.data_processing.utils_features import EnvironmentalFeatureDataset
from huggingface_hub import hf_hub_download

# Load vegetation-plot data (downloads automatically on first call)
eva_df  = EVADataset.from_hub()        # GeoDataFrame, EPSG:3035 point geometries
gift_df = GIFTDataset.from_hub()       # GeoDataFrame, EPSG:3035 polygon geometries

# Load environmental predictors
env_ds, lc_ds = EnvironmentalFeatureDataset.from_hub()
print(list(env_ds.data_vars))          # ['bio01', ..., 'bio19', 'elevation']

# Inspect the species matrix
species_list = eva_df.attrs['species_list']
print(f"{len(eva_df)} plots × {len(species_list)} species")

# Download one model-ready spatial cross-validation partition
test_partition = hf_hub_download(
    repo_id="vboussange/muscari-data",
    repo_type="dataset",
    filename="generated_samples/sbcv/ceacce0/fold_0_test.parquet",
)
```


## Citation

If you use this dataset, please cite both the MuScaRi paper and the underlying data sources:

```bibtex
@misc{boussange2025muscari,
  title   = {Multi-scale species richness estimation with deep learning},
  author  = {Victor Boussange and Bert Wuyts and Philipp Brun and
             Johanna T. Malle and Gabriele Midolo and Jeanne Portier and
             Théophile Sanchez and Niklaus E. Zimmermann and
             Irena Axmanová and Helge Bruelheide and Milan Chytrý and
             Stephan Kambach and Zdeňka Lososová and Martin Večeřa and
             Idoia Biurrun and Klaus T. Ecker and Jonathan Lenoir and
             Jens-Christian Svenning and Dirk Nikolaus Karger},
  year    = {2025},
  eprint  = {2507.06358},
  archivePrefix = {arXiv},
  primaryClass  = {q-bio.PE},
  url     = {https://arxiv.org/abs/2507.06358},
}

@article{chytry2016eva,
  title   = {European Vegetation Archive (EVA): An Integrated Database of
             European Vegetation Plots},
  author  = {Chytrý, Milan and others},
  journal = {Applied Vegetation Science},
  volume  = {19},
  number  = {1},
  pages   = {173--180},
  year    = {2016},
  doi     = {10.1111/avsc.12191},
}

@article{weigelt2020gift,
  title   = {{GIFT} -- A Global Inventory of Floras and Traits for
             macroecology and biogeography},
  author  = {Weigelt, Patrick and König, Christian and Kreft, Holger},
  journal = {Journal of Biogeography},
  volume  = {47},
  number  = {1},
  pages   = {16--43},
  year    = {2020},
  doi     = {10.1111/jbi.13623},
}
```
