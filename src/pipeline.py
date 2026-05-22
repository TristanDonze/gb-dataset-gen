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

from config import dataset_path, train_filename, snr_min, snr_max
from src.sampling import sample_gb_parameters_sobol, param_order
from src.waveform import generate_dataset

logger = logging.getLogger("DatasetGenerator")

def generate_and_save_dataset_in_blocks(n_samples_requested, block_size, filter_with_snr=False, seed=None):
    oversampled_request = n_samples_requested * 3
    params_gbgpu, actual_samples, N_final = sample_gb_parameters_sobol(oversampled_request, seed=seed)
    hdf5_path = os.path.join(dataset_path, train_filename)

    gb = GBGPU(force_backend="cuda")
    
    with h5py.File(hdf5_path, 'w') as f:
        dset_wave = f.create_dataset('waveforms', shape=(n_samples_requested, 4, N_final), dtype=np.float32, compression="gzip", compression_opts=9)
        params_group = f.create_group("params")
        dset_params_dict = {}
        for param in param_order:
            dset_params_dict[param] = params_group.create_dataset(param, shape=(n_samples_requested,), dtype=np.float32, compression="gzip", compression_opts=9)
        
        dset_snr = params_group.create_dataset("snr", shape=(n_samples_requested,), dtype=np.float32, compression="gzip", compression_opts=9)
            
        logger.info(f"Training : Starting generation to reach {n_samples_requested} valid samples with block size {block_size}. Oversampled to {actual_samples} parameters.")
        if filter_with_snr:
            logger.info(f"Waveforms will be filtered with SNR between {snr_min} and {snr_max}.")
        
        valid_samples_collected = 0
        param_index = 0
        
        pbar = tqdm.tqdm(total=n_samples_requested, desc="Valid Waveforms")
        
        while valid_samples_collected < n_samples_requested:
            if param_index >= params_gbgpu.shape[1]:
                raise ValueError("Not enough parameters sampled to reach the valid requested amount. Increase the oversampling factor.")
                
            end_index = min(param_index + block_size, params_gbgpu.shape[1])
            current_block_size = end_index - param_index
            params_block = params_gbgpu[:, param_index:end_index]
            
            waveforms_block, snr_block = generate_dataset(gb, params_block, N_final, nb_samples=current_block_size)
            if filter_with_snr:
                valid_mask = (snr_block >= snr_min) & (snr_block <= snr_max)
                
                valid_waveforms = waveforms_block[valid_mask]
                valid_params = params_block[:, valid_mask]
                valid_snr = snr_block[valid_mask]
                
                nb_valid_in_block = len(valid_waveforms)
                
                if nb_valid_in_block > 0:
                    space_left = n_samples_requested - valid_samples_collected
                    take = min(nb_valid_in_block, space_left)
                    
                    start_write = valid_samples_collected
                    end_write = valid_samples_collected + take
                    
                    dset_wave[start_write:end_write, :, :] = valid_waveforms[:take]
                    dset_snr[start_write:end_write] = valid_snr[:take]
                    for idx, param in enumerate(param_order):
                        dset_params_dict[param][start_write:end_write] = valid_params[idx, :take]
                    
                    valid_samples_collected += take
                    pbar.update(take)
            else:
                space_left = n_samples_requested - valid_samples_collected
                take = min(current_block_size, space_left)
                
                start_write = valid_samples_collected
                end_write = valid_samples_collected + take
                
                dset_wave[start_write:end_write, :, :] = waveforms_block[:take]
                dset_snr[start_write:end_write] = snr_block[:take]
                for idx, param in enumerate(param_order):
                    dset_params_dict[param][start_write:end_write] = params_block[idx, :take]
                
                valid_samples_collected += take
                pbar.update(take)
            
            param_index = end_index
        pbar.close()
    logger.info(f"Dataset fully generated and saved to {hdf5_path}")