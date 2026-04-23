import torch
import gbgpu

print(f"GPU available: {torch.cuda.is_available()}")
print(f"PyTorch CUDA version: {torch.version.cuda}")


for backend in ["gbgpu_cpu", "gbgpu_cuda11x", "gbgpu_cuda12x"]:
    print(f" - Backend '{backend}': {'available' if gbgpu.has_backend(backend) else 'unavailable'}")