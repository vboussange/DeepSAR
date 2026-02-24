import pandas as pd
import numpy as np
from pathlib import Path
import geopandas as gpd
from muscari.data_processing._cache import MUSCARI_CACHE_DIR

# Default base paths
GIFT_DATA_DIR = Path(__file__).parent / "../../data/processed/GIFT/anonymised/"
HF_DATASET_REPO = "vboussange/muscari-data"
GIFT_CACHE_DIR = MUSCARI_CACHE_DIR / "GIFT"

class GIFTDataset:
    """
    Loader and preprocessor for the GIFT dataset.

    Provides helpers to read anonymised parquet files and build a plot-level
    presence-absence matrix.
    """
    
    def __init__(self, data_dir=GIFT_DATA_DIR, cache_dir=GIFT_CACHE_DIR):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)

        # Cache path for preprocessed species/plot matrix
        self.preprocessed_cache = self.cache_dir / "species_matrix.parquet"

    @staticmethod
    def _set_species_list_attr(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        species_list = [col for col in df.columns if col not in ["geometry", "area_m2"]]
        df.attrs["species_list"] = species_list
        return df

    def read_species_data(self):
        """Load GIFT species data from the parquet file.

        Returns:
            pd.DataFrame: Species-level records with `record_id` and
                `anonymised_species_name` columns.
        """
        species_dataframe_path = self.data_dir / "species_data.parquet"
        if species_dataframe_path.exists():
            species_df = pd.read_parquet(species_dataframe_path)
            return species_df
        else:
            raise FileNotFoundError(f"Species data not found at {species_dataframe_path}. Did you preprocess the data?")
    
    def read_plot_data(self):
        """Load GIFT plot data from the parquet file.

        Returns:
            gpd.GeoDataFrame: Plot-level polygons and metadata.
        """
        plot_data_file = self.data_dir / "plot_data.parquet"
        if plot_data_file.exists():
            plot_data = gpd.read_parquet(plot_data_file)
            return plot_data
        else:
            raise FileNotFoundError(f"Plot data not found at {plot_data_file}. Did you preprocess the data?")

    def push_to_hub(self, repo_id: str, token: str = None):
        """Upload the GIFT species/plot matrix to the Hugging Face Hub.

        Builds the matrix first if the local cache does not exist yet, then
        uploads it as ``GIFT/species_matrix.parquet``.

        Args:
            repo_id: HF Hub repository id, e.g. ``"username/muscari-data"``.
            token: HF API token. Falls back to the cached login token when
                ``None``.
        """
        from huggingface_hub import HfApi

        # Always rebuild from source before upload
        print("Building matrix from source before upload…")
        GIFTDataset.from_source(data_dir=self.data_dir, use_cache=False)

        api = HfApi()
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, token=token)

        path_in_repo = "GIFT/species_matrix.parquet"
        print(f"Uploading {path_in_repo} …")
        api.upload_file(
            path_or_fileobj=str(self.preprocessed_cache),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )
        print(f"  ✓ {path_in_repo} uploaded")

    @classmethod
    def from_source(cls, data_dir=GIFT_DATA_DIR, cache_dir=GIFT_CACHE_DIR, use_cache: bool = True):
        """Build and return the GIFT species/plot matrix from source parquet files.

        Args:
            data_dir: Base GIFT data directory containing anonymised files.
            cache_dir: Cache directory where ``species_matrix.parquet`` is stored.
            use_cache: If ``True``, read/write ``species_matrix.parquet``.

        Returns:
            gpd.GeoDataFrame: Species presence/absence matrix with
                ``attrs['species_list']``.
        """
        instance = cls(data_dir=data_dir, cache_dir=cache_dir)

        if use_cache and instance.preprocessed_cache.exists():
            return cls._set_species_list_attr(gpd.read_parquet(instance.preprocessed_cache))

        print("Loading plot and species data...")
        plot_gdf = instance.read_plot_data()
        species_df = instance.read_species_data()

        if 'area_m2' not in plot_gdf.columns:
            plot_gdf['area_m2'] = plot_gdf.geometry.area

        print("Building species presence-absence matrix...")

        all_species = sorted(species_df['anonymised_species_name'].unique().tolist())
        species_to_idx = {sp: i for i, sp in enumerate(all_species)}

        record_ids = plot_gdf['record_id'].values
        record_id_to_row = {pid: i for i, pid in enumerate(record_ids)}

        n_plots = len(plot_gdf)
        n_species = len(all_species)

        species_df_filtered = species_df[species_df['record_id'].isin(record_id_to_row)]

        row_indices = species_df_filtered['record_id'].map(record_id_to_row).values
        col_indices = species_df_filtered['anonymised_species_name'].map(species_to_idx).values

        species_matrix = np.zeros((n_plots, n_species), dtype=np.bool_)
        species_matrix[row_indices, col_indices] = True

        print(f"  Matrix shape: {species_matrix.shape} ({n_plots} plots × {n_species} species)")
        print(f"  Sparsity: {100 * (1 - species_matrix.sum() / species_matrix.size):.2f}%")

        species_df_matrix = pd.DataFrame(species_matrix, columns=all_species, index=plot_gdf.index)
        df = pd.concat([plot_gdf, species_df_matrix], axis=1)

        df.attrs['species_list'] = all_species

        if use_cache:
            instance.preprocessed_cache.parent.mkdir(parents=True, exist_ok=True)
            print(f"Caching preprocessed data to {instance.preprocessed_cache}")
            df.to_parquet(instance.preprocessed_cache, index=False)
            print(f"✓ Cache saved ({instance.preprocessed_cache.stat().st_size / 1e6:.1f} MB)")

        return df

    @classmethod
    def from_hub(
        cls,
        repo_id: str = HF_DATASET_REPO,
        data_dir=GIFT_DATA_DIR,
        cache_dir=GIFT_CACHE_DIR,
        token: str = None,
        use_cache: bool = True,
    ):
        """Return the GIFT species/plot matrix, preferring Hugging Face cache."""
        import shutil
        from huggingface_hub import hf_hub_download

        instance = cls(data_dir=data_dir, cache_dir=cache_dir)
        dest = instance.preprocessed_cache

        if use_cache and dest.exists():
            return cls._set_species_list_attr(gpd.read_parquet(dest))

        path_in_repo = "GIFT/species_matrix.parquet"
        print(f"Downloading {path_in_repo} from {repo_id} …")
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=path_in_repo,
            repo_type="dataset",
            token=token,
        )
        if use_cache:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(downloaded, dest)
            print(f"  ✓ Saved to {dest}")
            return cls._set_species_list_attr(gpd.read_parquet(dest))

        return cls._set_species_list_attr(gpd.read_parquet(downloaded))

        
if __name__ == "__main__":
    dataset = GIFTDataset()
    plot_data = dataset.read_plot_data()
    species_data = dataset.read_species_data()
    df = GIFTDataset.from_hub(
        data_dir=dataset.data_dir,
        cache_dir=dataset.cache_dir,
        use_cache=True,
    )
    
    obs_areas = df['area_m2'].values
    species_list = df.attrs['species_list']
    species_matrix = df[species_list].values
    print(f"Loaded {len(df)} plots with {len(species_list)} species")
