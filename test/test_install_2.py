import cupy

_ = cupy.cuda.get_local_runtime_version()
sain_runtime = cupy.cuda.runtime

import torch
_ = torch.cuda.is_available()

from gbgpu.gbgpu import GBGPU
cupy.cuda.runtime = sain_runtime

print("Attempting to initialize GBGPU...")
gb = GBGPU(force_backend="cuda")
print("Success! GBGPU is initialized.")