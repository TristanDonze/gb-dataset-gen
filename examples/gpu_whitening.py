import cupy as cp
_ = cp.cuda.get_local_runtime_version()
sain_runtime = cp.cuda.runtime

import torch
_ = torch.cuda.is_available()

from gbgpu.gbgpu import GBGPU
cp.cuda.runtime = sain_runtime

from src.sampling import (
    sample_gb_parameters_uniform
)
from src.noise import AnalyticNoise

from config import dt, Tobs, f0_center
import numpy as np

def whitten_waveform_with_for(gb_object, nb_samples):
    N = gb_object.N 
    
    df = 1/Tobs
    f0_width = N * df

    f_min = f0_center - (f0_width / 2)
    f_max = f0_center + (f0_width / 2)

    k_min = int(np.round(f_min / df))
    freqs_grid = (cp.arange(N) + k_min) * df

    noise = AnalyticNoise(freqs_grid, "MRDv1")
    psd_A = noise.psd(option="A")
    psd_E = noise.psd(option="E")

    asd_A = cp.sqrt(psd_A)
    asd_E = cp.sqrt(psd_E)

    i_start = (gb.start_inds - k_min).astype(cp.int32)
    i_end = i_start + gb.N

    A_whitened = cp.zeros((nb_samples, N), dtype=cp.complex128)
    E_whitened = cp.zeros((nb_samples, N), dtype=cp.complex128)

    for i in range(nb_samples):
        start = max(0, int(i_start[i]))
        end = min(N, i_end[i])
        wave_start = start - i_start[i]
        wave_end = wave_start + (end - start)

        if start < end:
            A_whitened[i, start:end] = gb_object.A[i, wave_start:wave_end]
            E_whitened[i, start:end] = gb_object.E[i, wave_start:wave_end]
        else:
            print(f"Warning: Sample {i} has no overlap with the frequency grid. Skipping whitening for this sample.")
    
    A_whitened  *= cp.sqrt(4 * df) / asd_A
    E_whitened  *= cp.sqrt(4 * df) / asd_E
    return A_whitened, E_whitened

def whitten_waveform_fully_vectorized(gb_object, nb_samples):
    N = gb_object.N 
    
    df = 1/Tobs
    f0_width = N * df

    f_min = f0_center - (f0_width / 2)
    f_max = f0_center + (f0_width / 2)

    k_min = int(np.round(f_min / df))
    freqs_grid = (cp.arange(N) + k_min) * df

    noise = AnalyticNoise(freqs_grid, "MRDv1")
    psd_A = noise.psd(option="A")
    psd_E = noise.psd(option="E")

    asd_A = cp.sqrt(psd_A)
    asd_E = cp.sqrt(psd_E)

    i_start = (gb_object.start_inds - k_min).astype(cp.int32)

    A_whitened = cp.zeros((nb_samples, N), dtype=cp.complex128)
    E_whitened = cp.zeros((nb_samples, N), dtype=cp.complex128)

    # creation of a grid of destination columns (from 0 to 1279)
    cols = cp.arange(N)  # Shape: (1280,)

    # we compute the corresponding source indices for the 1000 signals using broadcasting: (1, 1280) - (1000, 1) => (1000, 1280)
    src_cols = cols[None, :] - i_start[:, None]

    #  we create the mask: we keep only the indices that actually exist in the source (between 0 and 1279)
    valid_mask = (src_cols >= 0) & (src_cols < N)  # Shape: (1000, 1280) of booleans

    # we create a grid to identify which row number (which sample) each cell belongs to
    rows = cp.arange(nb_samples)[:, None]
    rows_grid = cp.broadcast_to(rows, (nb_samples, N)) # Shape: (1000, 1280)

    # global copy in 1 GPU operation
    A_whitened[valid_mask] = gb_object.A[rows_grid[valid_mask], src_cols[valid_mask]]
    E_whitened[valid_mask] = gb_object.E[rows_grid[valid_mask], src_cols[valid_mask]]
        
    A_whitened  *= cp.sqrt(4 * df) / asd_A
    E_whitened  *= cp.sqrt(4 * df) / asd_E
    return A_whitened, E_whitened


if __name__ == "__main__":
    import time
    n_samples = 10000
    X, n_samples, N_final = sample_gb_parameters_uniform(n_samples, seed=42)
    print(f"Sampled parameters shape: {X.shape}, Number of samples: {n_samples}, N_final: {N_final}")
    gb = GBGPU(force_backend="cuda")
    gb.run_wave(*X, N=N_final, dt=dt, T=Tobs, oversample=None)

    start_t = time.time()
    A_whitened, E_whitened = whitten_waveform_with_for(gb, n_samples)
    end_t = time.time()

    start_t_vec = time.time()
    A_whitened_vec, E_whitened_vec = whitten_waveform_fully_vectorized(gb, n_samples)
    end_t_vec = time.time()

    if cp.allclose(A_whitened, A_whitened_vec) and cp.allclose(E_whitened, E_whitened_vec):
        print("Both methods yield the same results.")

    print(f"Execution time with for loop: {end_t - start_t:.4f} seconds\nExecution time with full vectorization: {end_t_vec - start_t_vec:.4f} seconds\nSpeedup: {(end_t - start_t) / (end_t_vec - start_t_vec):.2f}x")
