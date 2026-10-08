"""Smoke test: prove PyTorch sees the GPU inside the container.

Usage (from repo root):
    docker compose --profile amd run dev-amd python docker/verify_gpu.py
    docker compose --profile nvidia run dev-nvidia python docker/verify_gpu.py

Exits non-zero if no GPU is visible — use it as a container health check.
"""
import sys
import time
from pathlib import Path

# Provided by the container image, not the host environment.
import torch  # type: ignore[import-unresolved]  (container-only dep)
from ultralytics import YOLO  # type: ignore[import-unresolved]  (container-only dep)

print(f"torch      : {torch.__version__}")
print(f"cuda avail : {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("\nERROR: no GPU visible to PyTorch.")
    print("  AMD     -> check /dev/kfd and /dev/dri exist on the host and are mapped (see docker-compose.yml)")
    print("  NVIDIA  -> check nvidia-container-toolkit is installed: `docker run --gpus all nvidia/cuda apt ls`")
    sys.exit(1)

name = torch.cuda.get_device_name(0)
mem = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"device     : {name} ({mem:.1f} GB)")

# Quick FP16 matmul throughput — sanity-check that the compute path actually works
a = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
b = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
for _ in range(3):  # warmup
    _ = a @ b
torch.cuda.synchronize()
t0 = time.perf_counter()
iters = 10
for _ in range(iters):
    _ = a @ b
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / iters
tflops = 2 * 8192**3 / dt / 1e12
print(f"matmul     : {dt*1000:.1f} ms/iter  ({tflops:.0f} TFLOPS FP16)")

# Optional: time one YOLO inference if weights exist
weights = Path("results/military_drone_model/weights/best.pt")
if not weights.exists():
    print(f"\n(no trained weights at {weights} — skipping inference test; run `python main.py` first)")
    sys.exit(0)

model = YOLO(str(weights))
dummy = torch.randint(0, 255, (640, 640, 3), dtype=torch.uint8).numpy()
for _ in range(3):
    model.predict(dummy, verbose=False)  # warmup
t0 = time.perf_counter()
n = 10
for _ in range(n):
    model.predict(dummy, verbose=False)
dt = (time.perf_counter() - t0) / n * 1000
print(f"yolo11     : {dt:.1f} ms/frame @ 640px  (~{1000/dt:.0f} FPS)")
