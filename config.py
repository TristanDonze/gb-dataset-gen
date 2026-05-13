import os 
from pathlib import Path
import numpy as np
from lisatools.utils.constants import YRSID_SI


# 1. Paths & Dataset Configuration
scratch_dir         = Path("./data")
dataset_name        = "synthetic_dataset"
dataset_path        = Path(os.path.join(scratch_dir, dataset_name))
train_filename      = "train_dataset_2M_logamp_diff_1.hdf5"
val_filename        = "val_dataset_100K_diff_1.hdf5"

os.makedirs(dataset_path, exist_ok=True)
print(f"Dataset will be saved to: {dataset_path} as {train_filename} and {val_filename}")

train_dataset_size  = 2_000_000
val_dataset_size    = 100_000
dataset_block_size  = 10_000  # Number of samples to generate in each block (adjust based on available memory and speed requirements)

# 2. Observation parameters
dt          = 15.0                  # Time sampling cadence (seconds)
Tobs_years  = 1.0           # Observation time in years
Tobs        = Tobs_years * YRSID_SI

# 3. Fixed extrinsic parameters (Constants)
fixed_params = {
    "fddot": 0.0,
    "phi0":  4.830600316082553,
    "iota":  1.4686945655532282,
    "psi":   5.088202331694798,
}

# 4. Intrinsic parameters (Sobol ranges)

# Control of the difficulty of the dataset by adjusting the width of the f0 range. 
# The wider the range, the more separated the signals will be, making it easier for models to learn. 
# Conversely, a narrower range will create more overlapping signals, increasing the difficulty.

# 1 = Difficult (All signals packed within ~N points, strong interference)
# 10 = Medium (Signals spread over ~N*10 points)
# 100 = Easy (Signals spread over ~N*100 points, well separated signals)

difficulty_factor = 1
f0_center = 0.004821699107149872

sobol_ranges = {
    "log10_amp": [-24.0, -21.0],
    "fdot":     [9.5e-17, 1.9e-15],
    "beta_sin": [-1.0, 1.0],
    "lam":      [0.0, 2.0 * np.pi],
}

# ==========================================
# 5. VALIDATION PARAMETERS
# ==========================================
snr_min = 0.0
snr_max = 100.0