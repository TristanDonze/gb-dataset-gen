import cupy as cp

_ = cp.cuda.get_local_runtime_version()
sain_runtime = cp.cuda.runtime

import torch
_ = torch.cuda.is_available()

from gbgpu.gbgpu import GBGPU
cp.cuda.runtime = sain_runtime

# print("Attempting to initialize GBGPU...")
# gb = GBGPU(force_backend="cuda")
# print("Success! GBGPU is initialized.")

import numpy as np
from config import dt, f0_center, Tobs

params = np.array([
    [9.92075373e-23,  9.92075373e-23,  9.92075373e-23,  9.92075373e-23],
    [ 4.82056997e-03,  4.82285182e-03,  4.82368206e-03,  4.82124082e-03],
    [ 1.89879377e-15,  2.86937213e-16 , 7.50382335e-16  ,1.35241339e-15],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
    [ 4.83060032e+00,  4.83060032e+00,  4.83060032e+00,  4.83060032e+00],
    [ 1.46869457e+00,  1.46869457e+00,  1.46869457e+00,  1.46869457e+00],
    [ 5.08820233e+00,  5.08820233e+00,  5.08820233e+00,  5.08820233e+00],
    [ 6.06105915e+00,  2.59702560e+00,  1.97377241e+00,  1.07075745e+00],
    [ 7.96832638e-01, -7.80882558e-01, -4.82098749e-01,  6.09839799e-01]
])

gb = GBGPU(force_backend="cuda")
gb.run_wave(*params, dt=dt, T=Tobs, oversample=2)

A = gb.A # shape (4, 128)
E = gb.E # shape (4, 128)

N = gb.N

df = 1/Tobs
f0_width = N * df

f_min = f0_center - (f0_width / 2)
f_max = f0_center + (f0_width / 2)

k_min = int(np.round(f_min / df))
freqs_grid = (cp.arange(N) + k_min) * df

from src.noise import AnalyticNoise

noise = AnalyticNoise(freqs_grid, "MRDv1")
psd_A = noise.psd(option="A")
psd_E = noise.psd(option="E")

asd_A = cp.sqrt(psd_A)
asd_E = cp.sqrt(psd_E)

print(f"A shape: {A.shape}, E shape: {E.shape}" )
print(f"ASD A shape: {asd_A.shape}, ASD E shape: {asd_E.shape}")


def align_on_common_grid(A, E, start_inds, N):
    nb_samples = A.shape[0]
    i_start = (start_inds - k_min).astype(cp.int32)
    i_end = i_start + A.shape[1]

    A_aligned = cp.zeros((nb_samples, N), dtype=cp.complex128)
    E_aligned = cp.zeros((nb_samples, N), dtype=cp.complex128)

    for i in range(nb_samples):
        start = max(0, int(i_start[i]))
        end = min(N, int(i_end[i]))
        wave_start = start - int(i_start[i])
        wave_end = wave_start + (end - start)

        if start < end:
            A_aligned[i, start:end] = A[i, wave_start:wave_end]
            E_aligned[i, start:end] = E[i, wave_start:wave_end]
        else:
            print(f"Warning: Sample {i} has no overlap with the frequency grid.")

    return A_aligned, E_aligned


def whiten_aligned(A_aligned, E_aligned):
    A_whitened = A_aligned.copy()
    E_whitened = E_aligned.copy()
    A_whitened *= cp.sqrt(4 * df) / asd_A
    E_whitened *= cp.sqrt(4 * df) / asd_E
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

# First: align the raw waveforms on the same frequency grid.
A_aligned, E_aligned = align_on_common_grid(A, E, gb.start_inds, N)

# h_tot = whiten(h1 + h2 + h3 + h4)
sum_A_before_whitening = cp.sum(A_aligned, axis=0, keepdims=True)
sum_E_before_whitening = cp.sum(E_aligned, axis=0, keepdims=True)

print(f"sum_A_before_whitening shape: {sum_A_before_whitening.shape}, sum_E_before_whitening shape: {sum_E_before_whitening.shape}")

A_sum_whitened, E_sum_whitened = whiten_aligned(sum_A_before_whitening, sum_E_before_whitening)

# h_tot = whiten(h1) + whiten(h2) + whiten(h3) + whiten(h4)
A_whitened_each, E_whitened_each = whiten_aligned(A_aligned, E_aligned)
A_sum_after_whitening = cp.sum(A_whitened_each, axis=0, keepdims=True)
E_sum_after_whitening = cp.sum(E_whitened_each, axis=0, keepdims=True)

# Optional: same shape as the dataset tensors, useful for visual inspection.
summed_before_whitening_waveform = reshape_waveform(A_sum_whitened, E_sum_whitened)
summed_after_whitening_waveform = reshape_waveform(A_sum_after_whitening, E_sum_after_whitening)
print(f"summed_before_whitening_waveform shape: {summed_before_whitening_waveform.shape}, summed_after_whitening_waveform shape: {summed_after_whitening_waveform.shape}")

# Compare the complex GPU arrays directly, before the float32 reshape.
A_diff = A_sum_whitened - A_sum_after_whitening
E_diff = E_sum_whitened - E_sum_after_whitening

max_abs_diff_A = cp.max(cp.abs(A_diff)).get()
max_abs_diff_E = cp.max(cp.abs(E_diff)).get()
relative_l2_A = (cp.linalg.norm(A_diff) / cp.linalg.norm(A_sum_whitened)).get()
relative_l2_E = (cp.linalg.norm(E_diff) / cp.linalg.norm(E_sum_whitened)).get()
allclose_A = cp.allclose(A_sum_whitened, A_sum_after_whitening, rtol=1e-10, atol=1e-10).get()
allclose_E = cp.allclose(E_sum_whitened, E_sum_after_whitening, rtol=1e-10, atol=1e-10).get()
exactly_equal_A = cp.array_equal(A_sum_whitened, A_sum_after_whitening).get()
exactly_equal_E = cp.array_equal(E_sum_whitened, E_sum_after_whitening).get()

print(f"A max_abs_diff = {max_abs_diff_A:.6e}")
print(f"A relative_l2  = {relative_l2_A:.6e}")
print(f"A allclose     = {bool(allclose_A)}")
print(f"A exactly_equal = {bool(exactly_equal_A)}")
print(f"E max_abs_diff = {max_abs_diff_E:.6e}")
print(f"E relative_l2  = {relative_l2_E:.6e}")
print(f"E allclose     = {bool(allclose_E)}")
print(f"E exactly_equal = {bool(exactly_equal_E)}")
