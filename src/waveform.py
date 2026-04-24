import numpy as np
from gbgpu.gbgpu import GBGPU
from src.noise import AnalyticNoise

from config import dt, Tobs, snr_min, snr_max

def whiten_waveform(gb_object, f0_array):
    A_complex = gb_object.A.get() 
    E_complex = gb_object.E.get() 

    df_val = 1.0 / Tobs
    noise = AnalyticNoise(f0_array, "MRDv1")
    
    psd_A = noise.psd(option="A").get()[:, np.newaxis]
    psd_E = noise.psd(option="E").get()[:, np.newaxis]

    whith_A = np.sqrt(4.0 * df_val) / np.sqrt(psd_A)
    whith_E = np.sqrt(4.0 * df_val) / np.sqrt(psd_E)

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
    A_white, E_white = whiten_waveform(gb, params[1])
    return reshape_waveform(A_white, E_white)

def generate_filtered_block(params_block, N_final):
    gb = GBGPU(force_backend="cuda")
    gb.run_wave(*params_block, N=N_final, dt=dt, T=Tobs, oversample=None)

    A_complex = gb.A.get()
    E_complex = gb.E.get()
    
    f0_array = params_block[1, :]
    df_val = 1.0 / Tobs
    
    noise = AnalyticNoise(f0_array, "MRDv1")
    psd_A = noise.psd(option="A").get()[:, np.newaxis]
    psd_E = noise.psd(option="E").get()[:, np.newaxis]

    snr_A_sq = np.sum((np.abs(A_complex)**2) / psd_A, axis=1) * 4.0 * df_val
    snr_E_sq = np.sum((np.abs(E_complex)**2) / psd_E, axis=1) * 4.0 * df_val
    total_snr = np.sqrt(snr_A_sq + snr_E_sq)

    valid_mask = (total_snr >= snr_min) & (total_snr <= snr_max)
    valid_params = params_block[:, valid_mask]
    valid_snr = total_snr[valid_mask]

    if valid_params.shape[1] == 0:
        return valid_params, None, valid_snr

    valid_A = A_complex[valid_mask]
    valid_E = E_complex[valid_mask]
    valid_psd_A = psd_A[valid_mask]
    valid_psd_E = psd_E[valid_mask]

    whith_A = np.sqrt(4.0 * df_val) / np.sqrt(valid_psd_A)
    whith_E = np.sqrt(4.0 * df_val) / np.sqrt(valid_psd_E)

    A_white = valid_A * whith_A
    E_white = valid_E * whith_E

    valid_waveforms = reshape_waveform(A_white, E_white)
    return valid_params, valid_waveforms, valid_snr