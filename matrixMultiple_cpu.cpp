#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cblas.h>
#include <immintrin.h>

#define M 2048
#define K 2048
#define N 2048

constexpr int TILE_BLOCK = 128;
constexpr int REGISTER_BLOCK_N = 8;
constexpr int BENCHMARK_NAME_WIDTH = 36;

typedef void (*MatrixMultiplyFn)(const float[M][K], const float[K][N], float[M][N]);

double getTimeMs()
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

void printThreadConfiguration()
{
	printf("Thread config: OpenBLAS=%d\n",
		openblas_get_num_threads());
}

void initialMatrixA(float array[M][K])
{
	for (int i = 0; i < M; i++)
	{
		for (int k = 0; k < K; k++)
		{
			array[i][k] = (float)(rand() % 10 + 1);
		}
	}
}

void initialMatrixB(float array[K][N])
{
	for (int k = 0; k < K; k++)
	{
		for (int j = 0; j < N; j++)
		{
			array[k][j] = (float)(rand() % 10 + 1);
		}
	}
}

void printComparisonLogIfNeeded(const char *impl_name, const float impl[M][N], const float openblas[M][N])
{
	int max_row = 0;
	int max_col = 0;
	float max_diff = 0.0f;
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			float diff = fabsf(impl[i][j] - openblas[i][j]);
			if (diff > max_diff)
			{
				max_diff = diff;
				max_row = i;
				max_col = j;
			}
		}
	}

	if (max_diff == 0.0f)
	{
		return;
	}

	printf("comparison(%s vs openblas): max_abs_diff=%e at C[%d][%d], impl=%f, openblas=%f\n",
		impl_name,
		max_diff,
		max_row,
		max_col,
		impl[max_row][max_col],
		openblas[max_row][max_col]);

	const int sample_rows[3] = {0, M / 2, M - 1};
	const int sample_cols[3] = {0, N / 2, N - 1};
	for (int i = 0; i < 3; i++)
	{
		const int row = sample_rows[i];
		const int col = sample_cols[i];
		printf("  sample C[%d][%d]: impl=%f, openblas=%f, diff=%e\n",
			row,
			col,
			impl[row][col],
			openblas[row][col],
			fabsf(impl[row][col] - openblas[row][col]));
	}
}

void multiplicateMatrixNaive(const float array_A[M][K],
	const float array_B[K][N],
	float array_C[M][N])
{
	memset(array_C, 0, sizeof(float[M][N]));

	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			for (int k = 0; k < K; k++)
			{
				array_C[i][j] += array_A[i][k] * array_B[k][j];
			}
		}
	}
}

void multiplicateMatrixIKJ(const float array_A[M][K],
	const float array_B[K][N],
	float array_C[M][N])
{
	memset(array_C, 0, sizeof(float[M][N]));

	for (int i = 0; i < M; i++)
	{
		for (int k = 0; k < K; k++)
		{
			for (int j = 0; j < N; j++)
			{
				array_C[i][j] += array_A[i][k] * array_B[k][j];
			}
		}
	}
}

void matmulTile(const float* __restrict__ a,
	const float* __restrict__ b,
	float* __restrict__ c,
	int n,
	int k,
	int block)
{
	for (int i = 0; i < block; ++i)
	{
		const float *a_row = a + i * k;
		float *c_row = c + i * n;
		const float *b_row = b;

		for (int kk = 0; kk < block; ++kk)
		{
			const float a_ik = a_row[kk];
			for (int j = 0; j < block; ++j)
			{
				c_row[j] += a_ik * b_row[j];
			}
			b_row += n;
		}
	}
}

void matmulTileRegisterBlock(const float* __restrict__ a,
	const float* __restrict__ b,
	float* __restrict__ c,
	int n,
	int k,
	int block)
{
#if defined(__AVX2__) && defined(__FMA__)
	if (block % 4 != 0 || block % REGISTER_BLOCK_N != 0)
	{
		matmulTile(a, b, c, n, k, block);
		return;
	}

	for (int i = 0; i < block; i += 4)
	{
		const float *a_row0 = a + (i + 0) * k;
		const float *a_row1 = a + (i + 1) * k;
		const float *a_row2 = a + (i + 2) * k;
		const float *a_row3 = a + (i + 3) * k;
		float *c_row0 = c + (i + 0) * n;
		float *c_row1 = c + (i + 1) * n;
		float *c_row2 = c + (i + 2) * n;
		float *c_row3 = c + (i + 3) * n;

		for (int j = 0; j < block; j += REGISTER_BLOCK_N)
		{
			__m256 acc0 = _mm256_loadu_ps(c_row0 + j);
			__m256 acc1 = _mm256_loadu_ps(c_row1 + j);
			__m256 acc2 = _mm256_loadu_ps(c_row2 + j);
			__m256 acc3 = _mm256_loadu_ps(c_row3 + j);
			const float *b_row = b + j;

			for (int kk = 0; kk < block; ++kk)
			{
				const __m256 b_vec = _mm256_loadu_ps(b_row);
				acc0 = _mm256_fmadd_ps(_mm256_set1_ps(a_row0[kk]), b_vec, acc0);
				acc1 = _mm256_fmadd_ps(_mm256_set1_ps(a_row1[kk]), b_vec, acc1);
				acc2 = _mm256_fmadd_ps(_mm256_set1_ps(a_row2[kk]), b_vec, acc2);
				acc3 = _mm256_fmadd_ps(_mm256_set1_ps(a_row3[kk]), b_vec, acc3);
				b_row += n;
			}

			_mm256_storeu_ps(c_row0 + j, acc0);
			_mm256_storeu_ps(c_row1 + j, acc1);
			_mm256_storeu_ps(c_row2 + j, acc2);
			_mm256_storeu_ps(c_row3 + j, acc3);
		}
	}
#else
	matmulTile(a, b, c, n, k, block);
#endif
}

void multiplicateMatrixIKJTile(const float array_A[M][K],
	const float array_B[K][N],
	float array_C[M][N])
{
	memset(array_C, 0, sizeof(float[M][N]));

	constexpr int tile_block = TILE_BLOCK;
	for (int i = 0; i < M; i += tile_block)
	{
		for (int k = 0; k < K; k += tile_block)
		{
			for (int j = 0; j < N; j += tile_block)
			{
				const float* tile_A = &array_A[i][k];
				const float* tile_B = &array_B[k][j];
				float* tile_C = &array_C[i][j];
				matmulTile(tile_A, tile_B, tile_C, N, K, tile_block);
			}
		}
	}
}

void multiplicateMatrixIKJTileRegBlock(const float array_A[M][K],
	const float array_B[K][N],
	float array_C[M][N])
{
	memset(array_C, 0, sizeof(float[M][N]));

	constexpr int tile_block = TILE_BLOCK;
	for (int i = 0; i < M; i += tile_block)
	{
		for (int k = 0; k < K; k += tile_block)
		{
			for (int j = 0; j < N; j += tile_block)
			{
				const float *tile_A = &array_A[i][k];
				const float *tile_B = &array_B[k][j];
				float *tile_C = &array_C[i][j];
				matmulTileRegisterBlock(tile_A, tile_B, tile_C, N, K, tile_block);
			}
		}
	}
}

void benchmarkImplementation(const char *name,
	MatrixMultiplyFn fn,
	const float array_A[M][K],
	const float array_B[K][N],
	float array_C[M][N],
	const float openblasRef[M][N])
{
	double start = getTimeMs();
	fn(array_A, array_B, array_C);
	double finish = getTimeMs();
	double time = finish - start;

	printf("%-*s (%d×%d)  CPU运行时间为：%lfms\n",
		BENCHMARK_NAME_WIDTH,
		name,
		M,
		N,
		time);
	printComparisonLogIfNeeded(name, array_C, openblasRef);
}

int main()
{
	srand(0);

	float (*h_A)[K] = (float (*)[K])malloc(sizeof(float[M][K]));
	float (*h_B)[N] = (float (*)[N])malloc(sizeof(float[K][N]));
	float (*result)[N] = (float (*)[N])malloc(sizeof(float[M][N]));
	float (*openblasRef)[N] = (float (*)[N])malloc(sizeof(float[M][N]));

	if (h_A == NULL || h_B == NULL || result == NULL || openblasRef == NULL)
	{
		fprintf(stderr, "host allocation failed\n");
		return EXIT_FAILURE;
	}

	initialMatrixA(h_A);
	initialMatrixB(h_B);
	printThreadConfiguration();

	double start = getTimeMs();
	cblas_sgemm(CblasRowMajor,
		CblasNoTrans,
		CblasNoTrans,
		M,
		N,
		K,
		1.0f,
		&h_A[0][0],
		K,
		&h_B[0][0],
		N,
		0.0f,
		&openblasRef[0][0],
		N);
	double finish = getTimeMs();
	double openblasTime = finish - start;

	printf("\n");
	printf("------------------------------------------------------------------------------------\n");
	printf("Computing matrix product using single-thread CPU implementations \n");
	printf("------------------------------------------------------------------------------------\n");
	// benchmarkImplementation("multiplicateMatrixNaive:",
	// 	multiplicateMatrixNaive,
	// 	h_A,
	// 	h_B,
	// 	result,
	// 	openblasRef);
	benchmarkImplementation("multiplicateMatrixIKJ:",
		multiplicateMatrixIKJ,
		h_A,
		h_B,
		result,
		openblasRef);
	benchmarkImplementation("multiplicateMatrixIKJTile:",
		multiplicateMatrixIKJTile,
		h_A,
		h_B,
		result,
		openblasRef);
	benchmarkImplementation("multiplicateMatrixIKJTileRegBlock:",
		multiplicateMatrixIKJTileRegBlock,
		h_A,
		h_B,
		result,
		openblasRef);

	printf("%-*s (%d×%d)  CPU运行时间为：%lfms\n",
		BENCHMARK_NAME_WIDTH,
		"openblasSgemm:",
		M,
		N,
		openblasTime);

	free(h_A);
	free(h_B);
	free(result);
	free(openblasRef);
	return 0;
}
