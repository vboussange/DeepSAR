"""
This script attempts to re-implement
`scripts/data_processing/compile_eva_chelsa.py` by leveraging JAX to benefit
from JIT and GPU acceleration, bypassing the bottlenecks arising from -
`run_SR_compilation` (bottleneck is the loop over all `sp_units`; no
vectorization)
    - for this one, a tentative JAX implementation is provided below,
      `run_SR_compilation_jax`; it needs to be tested, possibly with unit tests
- `run_climate_compilation` (here again, the loop over all `sp_units` is the
  bottleneck)
    - for this one, we could consider using `xarray_jax` or `coordax` instead of
      plain `xarray`; no implementation yet
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import jax
import jax.numpy as jnp
from functools import partial

def preprocess_inputs(plot_gdf, species_dict):
    """
    Converts GeoPandas data and species dict into JAX-compatible arrays.
    
    Returns:
        coords: jnp.array (N_plots, 2) [x, y]
        obs_areas: jnp.array (N_plots,)
        species_matrix: jnp.array (N_plots, N_species) bool
        species_index_map: list mapping matrix column index to species ID
    """
    # 1. Extract Coordinates
    # Ensure we are working with points. 
    # If polygons, we take centroids (standard approximation for 'bags of plots')
    coords = np.column_stack((plot_gdf.geometry.x, plot_gdf.geometry.y))
    obs_areas = plot_gdf['observed_area'].values.astype(np.float32)

    # 2. Vectorize Species Data (One-Hot / Adjacency Matrix)
    # Get all unique species across the dataset
    all_species = sorted(list(set(sp for sublist in species_dict.values() for sp in sublist)))
    species_to_idx = {sp: i for i, sp in enumerate(all_species)}
    n_plots = len(plot_gdf)
    n_species = len(all_species)
    
    # Create binary matrix (N_plots x N_species)
    # Using int8 or bool to save memory
    species_matrix = np.zeros((n_plots, n_species), dtype=bool)
    
    # Fill matrix (this part is slow but runs only once)
    # Assuming plot_gdf index aligns with keys in species_dict
    # If species_dict keys are different, map them appropriately here
    plot_ids = plot_gdf.index.tolist()
    for row_idx, plot_id in enumerate(plot_ids):
        if plot_id in species_dict:
            sp_indices = [species_to_idx[sp] for sp in species_dict[plot_id]]
            species_matrix[row_idx, sp_indices] = True
            
    return (jnp.array(coords), 
            jnp.array(obs_areas), 
            jnp.array(species_matrix), 
            all_species)
    
@partial(jax.jit, static_argnames=['min_area', 'max_area'])
def _generate_random_square_jax(key, coords, min_area, max_area):
    """
    JAX-compatible random square generation.
    Returns bounding box: [min_x, min_y, max_x, max_y] and the generated area.
    """
    k1, k2, k3 = jax.random.split(key, 3)
    
    # 1. Select random center point from existing plots
    idx = jax.random.randint(k1, shape=(), minval=0, maxval=coords.shape[0])
    center = coords[idx] # (2,)
    
    # 2. Sample Area using inverse transform sampling for 1/x distribution
    # PDF: f(x) ~ 1/x -> CDF requires log sampling
    # Log-uniform sampling logic: exp(U(log(a), log(b)))
    log_min = jnp.log(min_area)
    log_max = jnp.log(max_area)
    u = jax.random.uniform(k2)
    current_log_area = log_min + u * (log_max - log_min)
    area = jnp.exp(current_log_area)
    
    # 3. Calculate Box geometry
    # box is a square, so side = sqrt(area)
    side = jnp.sqrt(area)
    half_side = side / 2.0
    
    min_x = center[0] - half_side
    min_y = center[1] - half_side
    max_x = center[0] + half_side
    max_y = center[1] + half_side
    
    return jnp.array([min_x, min_y, max_x, max_y]), area

@jax.jit
def _process_single_iteration(key, coords, obs_areas, species_matrix, area_range):
    """
    Runs the logic for a single 'sp_unit' iteration.
    """
    k_box, k_subsample = jax.random.split(key, 2)
    
    # 1. Generate Random Box
    bbox, box_area = _generate_random_square_jax(
        k_box, coords, area_range[0], area_range[1]
    )
    
    # 2. Identify Plots within Box
    # Vectorized bounds check: (x > xmin) & (x < xmax) & (y > ymin) & (y < ymax)
    # coords is (N, 2)
    x = coords[:, 0]
    y = coords[:, 1]
    
    in_box_mask = (
        (x >= bbox[0]) & (x <= bbox[2]) & 
        (y >= bbox[1]) & (y <= bbox[3])
    )
    
    count_in_box = jnp.sum(in_box_mask)
    
    # 3. Subsampling Logic
    # We need to handle the case where count_in_box == 0 to avoid NaNs
    # Logic: x = 10^(uniform(log10(1), log10(count))) -> This is log-uniform
    
    log_n = jnp.log10(jnp.maximum(count_in_box, 1.0)) # protect log(0)
    u_sample = jax.random.uniform(k_subsample)
    # Interpolate in log10 space
    log_target = u_sample * log_n # ranges from 0 (log10(1)) to log10(N)
    target_count = jnp.power(10.0, log_target).astype(jnp.int32)
    
    # Ensure target is at least 1 if plots exist, but 0 if empty
    target_count = jnp.where(count_in_box > 0, jnp.maximum(1, target_count), 0)
    
    # To sample exactly 'target_count' plots without replacement:
    # We assign random scores to all plots, mask out those outside the box (score=-inf),
    # and pick the top K.
    
    # Generate random scores for shuffling
    rand_scores = jax.random.uniform(k_subsample, shape=(coords.shape[0],))
    
    # If not in box, set score to -1 (impossible to be picked over meaningful scores)
    # effective_scores = rand_scores * in_box_mask # This sets outside to 0. 
    # But we want to distinguish between "in box but low score" and "outside box".
    # Better: effective_scores = rand_scores + (in_box_mask * 100.0) 
    # Plots inside get score > 100, plots outside < 1.
    effective_scores = rand_scores + (in_box_mask * 2.0)
    
    # Get indices of top k plots. 
    # Note: efficient top_k in JAX requires static k, but our k is dynamic.
    # Approach: Create a boolean mask of the selected plots.
    
    # Rank plots by score (descending)
    # We can't use dynamic slice. We generate a cutoff rank.
    # However, argsort is expensive on large N.
    # Alternative for JAX: Compare scores against a threshold? Hard to find threshold.
    # Standard way: argsort, but that sorts N items. 
    
    # Optimization: Since we just need a boolean mask for aggregation:
    # Let's use `argsort` but keep in mind this is O(N log N).
    sorted_indices = jnp.argsort(effective_scores)[::-1] # Descending
    
    # Create a mask for the top 'target_count' elements
    rank_mask = jnp.arange(coords.shape[0]) < target_count
    
    # Map back to original indices
    # We want a mask 'is_selected' of shape (N,)
    # scattered_indices = sorted_indices where rank_mask is True
    # We construct the binary mask:
    selected_indices = jnp.where(rank_mask, sorted_indices, -1) # -1 is dummy
    
    # Create final boolean mask
    final_mask = jnp.zeros(coords.shape[0], dtype=bool)
    final_mask = final_mask.at[selected_indices].set(True)
    # Remove the dummy index -1 if it was set (it wraps to end in python, strict check needed)
    # Actually, simpler: just set valid ones.
    final_mask = jnp.zeros(coords.shape[0], dtype=bool)
    # Only update where index != -1
    final_mask = final_mask.at[selected_indices].set(
        jnp.where(selected_indices != -1, True, False)
    )
    
    # 4. Calculate Metrics
    
    # Species Richness: Union of species
    # Matrix multiply: (1, N_plots) @ (N_plots, N_species) -> (1, N_species) counts
    # Or simply: sum over axis 0 of masked matrix
    
    # subset: (N_plots, N_species) where unselected rows are False
    # But doing `species_matrix[final_mask]` is dynamic shape -> No Go in JIT.
    # Instead: Multiply by mask (broadcasting)
    
    # Logic: If a species is present in ANY selected plot, it counts.
    # sum( (species_matrix * final_mask[:, None]) > 0 ) is not quite right because we want Union.
    # Correct: max(species_matrix * mask, axis=0) -> 1 if present in at least one, 0 else.
    
    # Cast to int to allow math
    active_species = jnp.any(species_matrix & final_mask[:, None], axis=0)
    sr = jnp.sum(active_species)
    
    # Area metrics
    observed_area_sum = jnp.sum(obs_areas * final_mask)
    sp_unit_area = jnp.maximum(box_area, observed_area_sum)
    
    # If no plots found, SR is 0, etc. handled naturally by mask being all False.
    
    return bbox, observed_area_sum, sp_unit_area, sr, target_count, final_mask

# Batch processing
# We use vmap to run `sp_units` iterations in parallel
# in_axes: (0, None, None, None, None) -> Key is split, others broadcasted
_process_batch = jax.vmap(_process_single_iteration, in_axes=(0, None, None, None, None))

def run_SR_compilation_jax(plot_gdf, 
                           species_dict, 
                           sp_units, 
                           area_range, 
                           verbose=True,
                           batch_size=1000):
    
    if verbose: print("Preprocessing inputs for JAX...")
    
    # 1. Prepare Data
    coords, obs_areas, sp_matrix, all_species = preprocess_inputs(plot_gdf, species_dict)
    
    # Move to GPU/TPU if available, else CPU
    coords = jax.device_put(coords)
    obs_areas = jax.device_put(obs_areas)
    sp_matrix = jax.device_put(sp_matrix)
    area_range_arr = jnp.array(area_range)

    # 2. Setup Randomness
    nb_sp_units = sp_units if isinstance(sp_units, int) else len(sp_units)
    key = jax.random.PRNGKey(42) # Seed
    
    # 3. Execution (Chunked to prevent OOM)
    # If nb_sp_units is massive (e.g. 1M), we can't materialize all masks in one go.
    # We chunk the operations.
    
    results = {
        "bbox": [], "obs_area": [], "unit_area": [], "sr": [], "num_plots": [], "masks": []
    }
    
    # Calculate chunks
    n_batches = int(np.ceil(nb_sp_units / batch_size))
    
    if verbose: print(f"Compiling SR for {nb_sp_units} units in {n_batches} batches...")
    
    for i in range(n_batches):
        current_batch_size = min(batch_size, nb_sp_units - i * batch_size)
        key, subkey = jax.random.split(key)
        batch_keys = jax.random.split(subkey, current_batch_size)
        
        # Run JIT function
        # Note: block_until_ready() forces execution for accurate timing if benchmarking
        b_bbox, b_obs, b_unit, b_sr, b_num, b_mask = _process_batch(
            batch_keys, coords, obs_areas, sp_matrix, area_range_arr
        )
        
        # Transfer back to CPU (Host) to clear GPU memory
        results["bbox"].append(np.array(b_bbox))
        results["obs_area"].append(np.array(b_obs))
        results["unit_area"].append(np.array(b_unit))
        results["sr"].append(np.array(b_sr))
        results["num_plots"].append(np.array(b_num))
        
        # We process 'used_plots' mask here to save memory (convert to sparse indices)
        # Or store full mask if RAM permits. Here we assume we want indices later.
        # For strict compatibility with your output 'used_plots', we accumulate logic.
        # But returning a dense mask of (N_iters x N_plots) might be huge.
        # Let's aggregate unique used plots per batch.
        results["masks"].append(np.array(b_mask)) 
        
    # 4. Reconstruct Output
    if verbose: print("Reconstructing GeoDataFrame...")
    
    # Concatenate all batches
    bbox_all = np.concatenate(results["bbox"], axis=0) # (N_iter, 4)
    obs_all = np.concatenate(results["obs_area"])
    unit_all = np.concatenate(results["unit_area"])
    sr_all = np.concatenate(results["sr"])
    num_all = np.concatenate(results["num_plots"])
    
    # Create Geometries from Bboxes [minx, miny, maxx, maxy]
    from shapely.geometry import box
    geoms = [box(b[0], b[1], b[2], b[3]) for b in bbox_all]
    
    data = pd.DataFrame({
        "observed_area": obs_all,
        "sp_unit_area": unit_all,
        "sr": sr_all,
        "num_plots": num_all,
    })
    data = gpd.GeoDataFrame(data, geometry=geoms, crs=plot_gdf.crs)
    
    # Aggregate used plots
    # Collect all True indices from all masks
    # This is memory intensive if N_iter is huge.
    # Optimized way:
    all_masks = np.concatenate(results["masks"], axis=0)
    # Find any column (plot) that was used at least once
    used_plot_indices = np.where(all_masks.any(axis=0))[0]
    used_plots = plot_gdf.index[used_plot_indices].tolist()
    
    return data, set(used_plots) # TODO: used_plots is not necessarily wanted, if we do spatial block cross validation