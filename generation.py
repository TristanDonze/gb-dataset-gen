# Below imports are needed to initialize the GPU runtime and ensure that the GBGPU library can be used without issues. They should be placed at the very beginning of the script before any other imports or code execution.
import cupy
_ = cupy.cuda.get_local_runtime_version()
sain_runtime = cupy.cuda.runtime

import torch
_ = torch.cuda.is_available()

from gbgpu.gbgpu import GBGPU
cupy.cuda.runtime = sain_runtime

## After the above initialization, we can proceed with the rest of the imports and code for dataset generation.

import os
import tqdm
import h5py
import logging
import numpy as np

from config import (
    dataset_path,
    filename,
    train_dataset_size,
    val_dataset_size,
    dataset_block_size,
    dt,
    Tobs,
    fixed_params,
    difficulty_factor,
    f0_center,
    sobol_ranges,
    snr_min,
    snr_max,
)

from src.noise import AnalyticNoise

from gbgpu.utils.utility import get_N
from scipy.stats import qmc
from lisatools.utils.constants import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DatasetGenerator")

param_order = [
        "amp",
        "f0",
        "fdot",
        "fddot",
        "phi0",
        "iota",
        "psi",
        "lam",
        "beta"
    ]

def sample_gb_parameters_sobol(n_samples: int, seed=None):
    """
    Samples parameters for Galactic Binaries using a Sobol sequence and combines them 
    with fixed parameters into a single structured array.

    The function dynamically maps parameters from the `sobol_ranges` and `fixed_params` 
    dictionaries. It automatically handles the conversion of 'beta_sin' to 'beta' if needed.

    Args:
        n_samples (int): The requested number of parameter sets to generate. If this is 
                         not a power of 2, it will be automatically adjusted to the nearest 
                         power of 2 to ensure optimal Sobol sequence coverage.
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        tuple:
            - X (np.ndarray): A 2D array of shape (9, n_samples) containing all parameters
                              ordered exactly as expected by the GBGPU `run_wave` method:
                              [amp, f0, fdot, fddot, phi0, iota, psi, lam, beta].
                              This allows unpacking directly using *X.
            - n_samples (int): The actual number of samples generated, which may be adjusted to the nearest power of 2.
            - N_final (int): The calculated number of frequency bins required for the 
                             given observation time and difficulty factor.
    """
    m = int(np.ceil(np.log2(n_samples)))
    closest_value = 2**m
    if n_samples != closest_value:
        logger.info(f"Warning: n_samples ({n_samples}) adjusted to the nearest power of 2: {closest_value}")
        n_samples = closest_value
    
    base_N = get_N(fixed_params["amp"], f0_center, Tobs, oversample=2)[0]
    N_final = int(base_N * difficulty_factor)

    df = 1.0 / Tobs
    f0_width = N_final * df

    f0_min = f0_center - (f0_width / 2.0)
    f0_max = f0_center + (f0_width / 2.0)
    
    ranges = sobol_ranges.copy()
    ranges["f0"] = [f0_min, f0_max]

    # logger.info(f"N : {N_final}, df : {df:.2e}, f0_width : {f0_width:.2e}")
    # logger.info(f"Sampling {n_samples} GB parameters using Sobol sequences with the following ranges:")
    # for param, (min_val, max_val) in ranges.items():
    #     logger.info(f"  {param}: [{min_val:.2e}, {max_val:.2e}]")

    sampler = qmc.Sobol(d=len(ranges), scramble=True, seed=seed)
    sampled_X = sampler.random(n=n_samples)

    scaled_samples = {}
    for i, (param_name, (min_val, max_val)) in enumerate(ranges.items()):
        scaled_samples[param_name] = min_val + sampled_X[:, i] * (max_val - min_val)

    X_list = []
    for param in param_order:
        if param in fixed_params:
            X_list.append(np.full(n_samples, fixed_params[param]))
        elif param in scaled_samples:
            X_list.append(scaled_samples[param])
        elif param == "beta" and "beta_sin" in scaled_samples:
            X_list.append(np.arcsin(scaled_samples["beta_sin"]))
        else:
            raise ValueError(f"Parameter '{param}' is missing from both fixed_params and sobol_ranges.")
             
    X = np.array(X_list)
    
    return X, n_samples, N_final

def whiten_waveform(gb_object, f0_array):
    A_complex = gb_object.A.get() 
    E_complex = gb_object.E.get() 

    df_val = 1.0 / Tobs
    noise = AnalyticNoise(f0_array, "MRDv1")
    
    # Calculate PSD and reshape for broadcasting
    psd_A = noise.psd(option="A").get()[:, np.newaxis]
    psd_E = noise.psd(option="E").get()[:, np.newaxis]

    # Whitening factors
    whith_A = np.sqrt(4.0 * df_val) / np.sqrt(psd_A)
    whith_E = np.sqrt(4.0 * df_val) / np.sqrt(psd_E)

    # Apply whitening
    A_white = A_complex * whith_A
    E_white = E_complex * whith_E

    return A_white, E_white

def reshape_waveform(A_white, E_white):
    num_bin = A_white.shape[0]
    N = A_white.shape[1]

    A_real = A_white.real.reshape(num_bin, N)
    A_imag = A_white.imag.reshape(num_bin, N)
    E_real = E_white.real.reshape(num_bin, N)
    E_imag = E_white.imag.reshape(num_bin, N)
    
    waveform = np.stack([A_real, A_imag, E_real, E_imag], axis=1).astype(np.float32)
    
    return waveform

def generate_dataset(params, N):
    gb = GBGPU(force_backend="cuda")
    gb.run_wave(*params, N=N, dt=dt, T=Tobs, oversample=None)
    A_white, E_white = whiten_waveform(gb, params[1])  # Pass the f0_array as the second argument
    return reshape_waveform(A_white, E_white)

def generate_and_save_dataset_in_blocks(n_samples_requested, block_size, seed=None):
    params_gbgpu, actual_samples, N_final = sample_gb_parameters_sobol(n_samples_requested, seed=seed)
    hdf5_path = os.path.join(dataset_path, filename)
    
    with h5py.File(hdf5_path, 'w') as f:
        dset_wave = f.create_dataset(
            name='waveforms', 
            shape=(actual_samples, 4, N_final), 
            dtype=np.float32,
            compression="gzip",
            compression_opts=9
        )
        
        params_group = f.create_group("params")
        dset_params_dict = {}
        for param in param_order:
            dset_params_dict[param] = params_group.create_dataset(
                name=param, 
                shape=(actual_samples,), 
                dtype=np.float32,
                compression="gzip",
                compression_opts=9
            )
            
        logger.info(f"Starting generation for {actual_samples} waveforms in blocks of {block_size}...")
        
        for i in tqdm.tqdm(range(0, actual_samples, block_size)):
            end = min(i + block_size, actual_samples)
            
            params_block = params_gbgpu[:, i:end]
            waveforms_block = generate_dataset(params_block, N_final)
            
            dset_wave[i:end, :, :] = waveforms_block
            
            for idx, param in enumerate(param_order):
                dset_params_dict[param][i:end] = params_block[idx, :]

    logger.info(f"Dataset fully generated and saved to {hdf5_path}")

if __name__ == "__main__":
    generate_and_save_dataset_in_blocks(train_dataset_size, dataset_block_size, seed=42)