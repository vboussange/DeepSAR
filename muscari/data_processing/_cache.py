"""
Unified cache directory for all MuScaRi datasets.

Sub-directories::

    MUSCARI_CACHE_DIR/
        EVA/species_matrix.parquet
        GIFT/species_matrix.parquet
        environmental_features/chelsa_dem_cache.nc
                               landcover_cache.nc
"""

from pathlib import Path

MUSCARI_CACHE_DIR: Path = Path(__file__).parents[2] / "data" / ".cache"
