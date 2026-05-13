import cupy as cp
import numpy as np
from gbgpu.gbgpu import GBGPU
from src.noise import AnalyticNoise

from config import dt, Tobs, f0_center, snr_min, snr_max

def whiten_waveform(gb_object, nb_samples):
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

def reshape_waveform(A_white, E_white):
    num_bin = A_white.shape[0]
    N = A_white.shape[1]

    A_real = A_white.real
    A_imag = A_white.imag
    E_real = E_white.real
    E_imag = E_white.imag
    
    waveform = cp.stack([A_real, A_imag, E_real, E_imag], axis=1).astype(cp.float32)
    return waveform.get()

def generate_dataset(gb_object, params, N, nb_samples):
    gb_object.run_wave(*params, N=N, dt=dt, T=Tobs, oversample=None)
    A_white, E_white = whiten_waveform(gb_object, nb_samples)

    snr_squared = cp.sum(cp.abs(A_white)**2, axis=1) + cp.sum(cp.abs(E_white)**2, axis=1)
    snr = cp.sqrt(snr_squared)

    waveform = reshape_waveform(A_white, E_white)

    return waveform, snr.get()