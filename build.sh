#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

mkdir -p "${BUILD_DIR}"

read -r -a OPENBLAS_CFLAGS <<< "$(pkg-config --cflags openblas)"
read -r -a OPENBLAS_LIBS <<< "$(pkg-config --libs openblas)"

g++ "${SCRIPT_DIR}/matrixMultiple_cpu.cpp" \
	-O3 \
	-march=native \
	"${OPENBLAS_CFLAGS[@]}" \
	-o "${BUILD_DIR}/matrixMultiple_cpu" \
	"${OPENBLAS_LIBS[@]}"

echo
echo "===== CPU benchmark: single-thread ====="
env \
	OPENBLAS_NUM_THREADS=1 \
	"${BUILD_DIR}/matrixMultiple_cpu"

nvcc "${SCRIPT_DIR}/matrixMultiple.cu" \
	-O3 \
	-o "${BUILD_DIR}/matrixMultiple_gpu" \
	-lcublas

"${BUILD_DIR}/matrixMultiple_gpu"
