import torch
import time

size = 5000

# Sanity checks
print(f"PyTorch: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

# CPU: warmup, then time
_ = a_cpu @ b_cpu
start = time.perf_counter()
c_cpu = a_cpu @ b_cpu
cpu_time = time.perf_counter() - start
print(f"CPU: {cpu_time:.3f}s")

if torch.backends.mps.is_available():
    a_gpu = a_cpu.to("mps")
    b_gpu = b_cpu.to("mps")

    # Warmup: first MPS call compiles Metal kernels
    _ = a_gpu @ b_gpu
    torch.mps.synchronize()

    start = time.perf_counter()
    c_gpu = a_gpu @ b_gpu
    torch.mps.synchronize()
    gpu_time = time.perf_counter() - start
    print(f"GPU (MPS): {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.1f}x")