import cupy
_ = cupy.cuda.get_local_runtime_version()
sain_runtime = cupy.cuda.runtime

import torch
_ = torch.cuda.is_available()

from gbgpu.gbgpu import GBGPU
cupy.cuda.runtime = sain_runtime

import os
import h5py
import tqdm
import logging
import numpy as np

from config import dataset_path, train_filename
from src.sampling import sample_gb_parameters_sobol, param_order
from src.waveform import generate_dataset

logger = logging.getLogger("DatasetGenerator")

def generate_and_save_dataset_in_blocks(n_samples_requested, block_size, seed=None):
    params_gbgpu, actual_samples, N_final = sample_gb_parameters_sobol(n_samples_requested, seed=seed)
    hdf5_path = os.path.join(dataset_path, train_filename)

    gb = GBGPU(force_backend="cuda")
    
    with h5py.File(hdf5_path, 'w') as f:
        dset_wave = f.create_dataset('waveforms', shape=(actual_samples, 4, N_final), dtype=np.float32, compression="gzip", compression_opts=9)
        params_group = f.create_group("params")
        dset_params_dict = {}
        for param in param_order:
            dset_params_dict[param] = params_group.create_dataset(param, shape=(actual_samples,), dtype=np.float32, compression="gzip", compression_opts=9)
        
        dset_snr = params_group.create_dataset("snr", shape=(actual_samples,), dtype=np.float32, compression="gzip", compression_opts=9)

        logger.info(f"Training : Starting generation for {actual_samples} waveforms in blocks of {block_size}...")
        
        for i in tqdm.tqdm(range(0, actual_samples, block_size)):
            end = min(i + block_size, actual_samples)
            params_block = params_gbgpu[:, i:end]
            waveforms_block, snr_block = generate_dataset(gb, params_block, N_final, nb_samples=end - i)
            dset_wave[i:end, :, :] = waveforms_block
            dset_snr[i:end] = snr_block
            for idx, param in enumerate(param_order):
                dset_params_dict[param][i:end] = params_block[idx, :]

    logger.info(f"Dataset fully generated and saved to {hdf5_path}")