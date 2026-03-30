# Matmul Benchmarks

Small CPU and CUDA matrix-multiplication benchmarks.

This repo contains:

- `matrixMultiple_cpu.cpp`: CPU implementations and an OpenBLAS reference
- `matrixMultiple.cu`: CUDA kernels and a cuBLAS reference
- `build.sh`: builds and runs the CPU benchmark, then the GPU benchmark

## Implementations

### CPU

The CPU benchmark compares these single-thread implementations for `C = A * B`:

- `multiplicateMatrixIKJ`
- `multiplicateMatrixIKJTile`
- `multiplicateMatrixIKJTileRegBlock`
- `openblasSgemm`

Notes:

- matrix sizes are currently hard-coded as `M = K = N = 2048`
- `build.sh` forces `OPENBLAS_NUM_THREADS=1`, so the CPU comparison is single-thread only
- the OpenBLAS result is also used as the correctness reference

### GPU

The CUDA benchmark compares:

- `multiplicateMatrixOnDevice`: naive global-memory kernel
- `matrixMultiplyShared`: shared-memory tiled kernel
- `cublasSgemm`

The cuBLAS result is used as the GPU reference.

## Requirements

CPU:

- `g++`
- `pkg-config`
- OpenBLAS development package with `cblas.h`

GPU:

- `nvcc`
- CUDA toolkit
- `cublas`
- an NVIDIA GPU supported by your CUDA installation

## Build And Run

From the repo root:

```bash
./build.sh
```

This script:

1. builds `build/matrixMultiple_cpu`
2. runs the CPU benchmark with `OPENBLAS_NUM_THREADS=1`
3. builds `build/matrixMultiple_gpu`
4. runs the GPU benchmark

## Output

Typical CPU output looks like:

```text
Thread config: OpenBLAS=1
------------------------------------------------------------------------------------
Computing matrix product using single-thread CPU implementations
------------------------------------------------------------------------------------
multiplicateMatrixIKJ:               (2048×2048)  CPU运行时间为：...
multiplicateMatrixIKJTile:           (2048×2048)  CPU运行时间为：...
multiplicateMatrixIKJTileRegBlock:   (2048×2048)  CPU运行时间为：...
openblasSgemm:                       (2048×2048)  CPU运行时间为：...
```

If an implementation differs from the OpenBLAS reference, the program prints the maximum absolute difference and a few sample values.

## Tuning Notes

Current CPU tuning in `matrixMultiple_cpu.cpp`:

- `TILE_BLOCK = 128`
- `REGISTER_BLOCK_N = 8`

`multiplicateMatrixIKJTileRegBlock` uses an AVX2/FMA micro-kernel when available and falls back to the scalar tiled kernel otherwise.

## Files Generated

- `build/matrixMultiple_cpu`
- `build/matrixMultiple_gpu`

The `build/` directory is ignored by git.
