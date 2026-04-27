import numpy as np
from gbgpu.gbgpu import GBGPU
from src.noise import AnalyticNoise

from config import dt, Tobs, snr_min, snr_max

def whiten_waveform(gb_object, f0_array):
    pass

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