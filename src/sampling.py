import logging
import numpy as np

from src.noise import AnalyticNoise
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

from gbgpu.utils.utility import get_N
from scipy.stats import qmc
from lisatools.utils.constants import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Sampling")


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


def sample_gb_parameters_sobol(n_samples: int, force_difficulty_factor=None, seed=None):
    """
    Samples parameters for Galactic Binaries using a Sobol sequence and combines them 
    with fixed parameters into a single structured array.

    The function dynamically maps parameters from the `sobol_ranges` and `fixed_params` 
    dictionaries. It automatically handles the conversion of 'beta_sin' to 'beta' if needed.

    Args:
        n_samples (int): The requested number of parameter sets to generate. If this is 
                         not a power of 2, it will be automatically adjusted to the nearest 
                         power of 2 to ensure optimal Sobol sequence coverage.
        force_difficulty_factor (float, optional): If provided, this will be used as the difficulty factor instead of the default value.
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        tuple:
            - X (np.ndarray): A 2D array of shape (9, n_samples) containing all parameters ordered exactly as expected by the GBGPU `run_wave` method: [amp, f0, fdot, fddot, phi0, iota, psi, lam, beta]. This allows unpacking directly using *X.
            - n_samples (int): The actual number of samples generated, which may be adjusted to the nearest power of 2.
            - N_final (int): The calculated number of frequency bins required for the given observation time and difficulty factor.
    """
    m = int(np.ceil(np.log2(n_samples)))
    closest_value = 2**m
    if n_samples != closest_value:
        logger.info(f"Warning: n_samples ({n_samples}) adjusted to the nearest power of 2: {closest_value}")
        n_samples = closest_value
        
    effective_difficulty_factor = (
        force_difficulty_factor
        if force_difficulty_factor is not None
        else difficulty_factor
    )

    base_N = get_N(fixed_params["amp"], f0_center, Tobs, oversample=2)[0]
    N_final = int(base_N * effective_difficulty_factor)

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

def sample_gb_parameters_lhs(n_samples: int, force_difficulty_factor=None, seed=None):
    """
    Samples parameters for Galactic Binaries using a Latin Hypercube Sampling approach and combines them 
    with fixed parameters into a single structured array.

    The function dynamically maps parameters from the `sobol_ranges` and `fixed_params` 
    dictionaries. It automatically handles the conversion of 'beta_sin' to 'beta' if needed.

    Args:
        n_samples (int): The requested number of parameter sets to generate. If this is 
                         not a power of 2, it will be automatically adjusted to the nearest 
                         power of 2 to ensure optimal Latin Hypercube Sampling coverage.
        force_difficulty_factor (float, optional): If provided, this will be used as the difficulty factor instead of the default value.
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        tuple:
            - X (np.ndarray): A 2D array of shape (9, n_samples) containing all parameters ordered exactly as expected by the GBGPU `run_wave` method: [amp, f0, fdot, fddot, phi0, iota, psi, lam, beta]. This allows unpacking directly using *X.
            - n_samples (int): The actual number of samples generated, which may be adjusted to the nearest power of 2.
            - N_final (int): The calculated number of frequency bins required for the given observation time and difficulty factor.
    """
    m = int(np.ceil(np.log2(n_samples)))
    closest_value = 2**m
    if n_samples != closest_value:
        logger.info(f"Warning: n_samples ({n_samples}) adjusted to the nearest power of 2: {closest_value}")
        n_samples = closest_value
    
    effective_difficulty_factor = (
        force_difficulty_factor
        if force_difficulty_factor is not None
        else difficulty_factor
    )

    base_N = get_N(fixed_params["amp"], f0_center, Tobs, oversample=2)[0]
    N_final = int(base_N * effective_difficulty_factor)

    df = 1.0 / Tobs
    f0_width = N_final * df

    f0_min = f0_center - (f0_width / 2.0)
    f0_max = f0_center + (f0_width / 2.0)
    
    ranges = sobol_ranges.copy()
    ranges["f0"] = [f0_min, f0_max]

    # logger.info(f"N : {N_final}, df : {df:.2e}, f0_width : {f0_width:.2e}")
    # logger.info(f"Sampling {n_samples} GB parameters using Latin Hypercube Sampling with the following ranges:")
    # for param, (min_val, max_val) in ranges.items():
    #     logger.info(f"  {param}: [{min_val:.2e}, {max_val:.2e}]")

    sampler = qmc.LatinHypercube(d=len(ranges), seed=seed)
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

def sample_gb_parameters_uniform(n_samples: int, force_difficulty_factor=None, seed=None):
    """
    Samples parameters for Galactic Binaries using uniform random sampling and combines them 
    with fixed parameters into a single structured array.

    The function dynamically maps parameters from the `sobol_ranges` and `fixed_params` 
    dictionaries. It automatically handles the conversion of 'beta_sin' to 'beta' if needed.

    Args:
        n_samples (int): The number of parameter sets to generate.
        force_difficulty_factor (float, optional): If provided, this will be used as the difficulty factor instead of the default value.
        seed (int, optional): Random seed for reproducibility. Defaults to None.
    Returns:
        tuple:
            - X (np.ndarray): A 2D array of shape (9, n_samples) containing all parameters ordered exactly as expected by the GBGPU `run_wave` method: [amp, f0, fdot, fddot, phi0, iota, psi, lam, beta]. This allows unpacking directly using *X.
            - n_samples (int): The number of samples generated.
            - N_final (int): The calculated number of frequency bins required for the given observation time and difficulty factor.
    """
    if seed is not None:
        np.random.seed(seed)
        
    effective_difficulty_factor = (
        force_difficulty_factor
        if force_difficulty_factor is not None
        else difficulty_factor
    )

    base_N = get_N(fixed_params["amp"], f0_center, Tobs, oversample=2)[0]
    N_final = int(base_N * effective_difficulty_factor)

    df = 1.0 / Tobs
    f0_width = N_final * df

    f0_min = f0_center - (f0_width / 2.0)
    f0_max = f0_center + (f0_width / 2.0)
    
    ranges = sobol_ranges.copy()
    ranges["f0"] = [f0_min, f0_max]

    scaled_samples = {}
    for param_name, (min_val, max_val) in ranges.items():
        scaled_samples[param_name] = np.random.uniform(min_val, max_val, n_samples)

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