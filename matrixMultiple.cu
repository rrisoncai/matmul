#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"

#define M 2048
#define K 2048
#define N 2048

#define BLOCK_SIZE 32

#define CHECK_CUDA(call)                                                                        \
	do                                                                                         \
	{                                                                                          \
		cudaError_t err = (call);                                                              \
		if (err != cudaSuccess)                                                                \
		{                                                                                      \
			fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
			exit(EXIT_FAILURE);                                                                \
		}                                                                                      \
	} while (0)

#define CHECK_CUBLAS(call)                                                                      \
	do                                                                                         \
	{                                                                                          \
		cublasStatus_t status = (call);                                                        \
		if (status != CUBLAS_STATUS_SUCCESS)                                                   \
		{                                                                                      \
			fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, (int)status);     \
			exit(EXIT_FAILURE);                                                                \
		}                                                                                      \
	} while (0)

float maxAbsDiff(const float *lhs, const float *rhs, int size)
{
	float max_diff = 0.0f;
	for (int i = 0; i < size; i++)
	{
		float diff = fabsf(lhs[i] - rhs[i]);
		if (diff > max_diff)
		{
			max_diff = diff;
		}
	}
	return max_diff;
}

void initial(float *array, int size)
{
	for (int i = 0; i < size; i++)
	{
		array[i] = (float)(rand() % 10 + 1);
	}
}

__global__ void multiplicateMatrixOnDevice(const float *array_A,
	const float *array_B,
	float *array_C,
	int M_p,
	int K_p,
	int N_p)
{
	int ix = threadIdx.x + blockDim.x * blockIdx.x;
	int iy = threadIdx.y + blockDim.y * blockIdx.y;

	if (ix < N_p && iy < M_p)
	{
		float sum = 0.0f;
		for (int k = 0; k < K_p; k++)
		{
			sum += array_A[iy * K_p + k] * array_B[k * N_p + ix];
		}
		array_C[iy * N_p + ix] = sum;
	}
}

__global__ void matrixMultiplyShared(const float *A,
	const float *B,
	float *C,
	int numARows,
	int numAColumns,
	int numBRows,
	int numBColumns,
	int numCRows,
	int numCColumns)
{
	__shared__ float sharedM[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float sharedN[BLOCK_SIZE][BLOCK_SIZE];

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int row = by * BLOCK_SIZE + ty;
	int col = bx * BLOCK_SIZE + tx;

	float c_sub = 0.0f;

	for (int tile = 0; tile < (numAColumns + BLOCK_SIZE - 1) / BLOCK_SIZE; tile++)
	{
		if (tile * BLOCK_SIZE + tx < numAColumns && row < numARows)
			sharedM[ty][tx] = A[row * numAColumns + tile * BLOCK_SIZE + tx];
		else
			sharedM[ty][tx] = 0.0f;

		if (tile * BLOCK_SIZE + ty < numBRows && col < numBColumns)
			sharedN[ty][tx] = B[(tile * BLOCK_SIZE + ty) * numBColumns + col];
		else
			sharedN[ty][tx] = 0.0f;

		__syncthreads();

		for (int j = 0; j < BLOCK_SIZE; j++)
			c_sub += sharedM[ty][j] * sharedN[j][tx];

		__syncthreads();
	}

	if (row < numCRows && col < numCColumns)
		C[row * numCColumns + col] = c_sub;
}

int main()
{
	srand(0);

	int Axy = M * K;
	int Bxy = K * N;
	int Cxy = M * N;

	float *h_A = (float *)malloc(Axy * sizeof(float));
	float *h_B = (float *)malloc(Bxy * sizeof(float));
	float *naiveRef = (float *)malloc(Cxy * sizeof(float));
	float *sharedRef = (float *)malloc(Cxy * sizeof(float));
	float *cublasRef = (float *)malloc(Cxy * sizeof(float));

	if (h_A == NULL || h_B == NULL || naiveRef == NULL || sharedRef == NULL || cublasRef == NULL)
	{
		fprintf(stderr, "host allocation failed\n");
		return EXIT_FAILURE;
	}

	initial(h_A, Axy);
	initial(h_B, Bxy);

	float *d_A = NULL;
	float *d_B = NULL;
	float *d_C = NULL;
	CHECK_CUDA(cudaMalloc((void **)&d_A, Axy * sizeof(float)));
	CHECK_CUDA(cudaMalloc((void **)&d_B, Bxy * sizeof(float)));
	CHECK_CUDA(cudaMalloc((void **)&d_C, Cxy * sizeof(float)));

	CHECK_CUDA(cudaMemcpy(d_A, h_A, Axy * sizeof(float), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(d_B, h_B, Bxy * sizeof(float), cudaMemcpyHostToDevice));

	printf("\n");
	printf("------------------------------------------------------------------------------------\n");
	printf("Computing matrix product using GPU kernels and cuBLAS \n");
	printf("------------------------------------------------------------------------------------\n");

	cudaEvent_t gpustart, gpustop;
	float elapsedTime = 0.0f;

	dim3 naive_block(16, 16);
	dim3 naive_grid((N + naive_block.x - 1) / naive_block.x, (M + naive_block.y - 1) / naive_block.y);
	CHECK_CUDA(cudaEventCreate(&gpustart));
	CHECK_CUDA(cudaEventCreate(&gpustop));
	CHECK_CUDA(cudaEventRecord(gpustart, 0));
	multiplicateMatrixOnDevice<<<naive_grid, naive_block>>>(d_A, d_B, d_C, M, K, N);
	CHECK_CUDA(cudaEventRecord(gpustop, 0));
	CHECK_CUDA(cudaEventSynchronize(gpustop));
	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaEventElapsedTime(&elapsedTime, gpustart, gpustop));
	CHECK_CUDA(cudaMemcpy(naiveRef, d_C, Cxy * sizeof(float), cudaMemcpyDeviceToHost));
	printf("multiplicateMatrixOnDevice: (%d×%d)  <<<(%d,%d),(%d,%d)>>>  GPU运行时间为：%fms\n",
		M, N, naive_grid.x, naive_grid.y, naive_block.x, naive_block.y, elapsedTime);
	CHECK_CUDA(cudaEventDestroy(gpustart));
	CHECK_CUDA(cudaEventDestroy(gpustop));

	dim3 shared_block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 shared_grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
	elapsedTime = 0.0f;
	CHECK_CUDA(cudaEventCreate(&gpustart));
	CHECK_CUDA(cudaEventCreate(&gpustop));
	CHECK_CUDA(cudaEventRecord(gpustart, 0));
	matrixMultiplyShared<<<shared_grid, shared_block>>>(d_A, d_B, d_C, M, K, K, N, M, N);
	CHECK_CUDA(cudaEventRecord(gpustop, 0));
	CHECK_CUDA(cudaEventSynchronize(gpustop));
	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaEventElapsedTime(&elapsedTime, gpustart, gpustop));
	CHECK_CUDA(cudaMemcpy(sharedRef, d_C, Cxy * sizeof(float), cudaMemcpyDeviceToHost));
	printf("matrixMultiplyShared:       (%d×%d)  <<<(%d,%d),(%d,%d)>>>  GPU运行时间为：%fms\n",
		M, N, shared_grid.x, shared_grid.y, shared_block.x, shared_block.y, elapsedTime);
	CHECK_CUDA(cudaEventDestroy(gpustart));
	CHECK_CUDA(cudaEventDestroy(gpustop));

	cublasHandle_t handle;
	CHECK_CUBLAS(cublasCreate(&handle));

	const float alpha = 1.0f;
	const float beta = 0.0f;
	elapsedTime = 0.0f;
	CHECK_CUDA(cudaEventCreate(&gpustart));
	CHECK_CUDA(cudaEventCreate(&gpustop));
	CHECK_CUDA(cudaEventRecord(gpustart, 0));
	CHECK_CUBLAS(cublasSgemm(handle,
		CUBLAS_OP_N,
		CUBLAS_OP_N,
		N,
		M,
		K,
		&alpha,
		d_B,
		N,
		d_A,
		K,
		&beta,
		d_C,
		N));
	CHECK_CUDA(cudaEventRecord(gpustop, 0));
	CHECK_CUDA(cudaEventSynchronize(gpustop));
	CHECK_CUDA(cudaEventElapsedTime(&elapsedTime, gpustart, gpustop));
	CHECK_CUDA(cudaMemcpy(cublasRef, d_C, Cxy * sizeof(float), cudaMemcpyDeviceToHost));
	printf("cublasSgemm:                (%d×%d)  GPU运行时间为：%fms\n", M, N, elapsedTime);
	CHECK_CUDA(cudaEventDestroy(gpustart));
	CHECK_CUDA(cudaEventDestroy(gpustop));

	printf("naive_vs_cublas max_diff=%e\n", maxAbsDiff(naiveRef, cublasRef, Cxy));
	printf("shared_vs_cublas max_diff=%e\n", maxAbsDiff(sharedRef, cublasRef, Cxy));

	CHECK_CUBLAS(cublasDestroy(handle));
	CHECK_CUDA(cudaFree(d_A));
	CHECK_CUDA(cudaFree(d_B));
	CHECK_CUDA(cudaFree(d_C));

	free(h_A);
	free(h_B);
	free(naiveRef);
	free(sharedRef);
	free(cublasRef);

	CHECK_CUDA(cudaDeviceReset());
	return 0;
}
