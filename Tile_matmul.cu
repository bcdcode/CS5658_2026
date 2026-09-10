#include <stdio.h>
#include <cuda_runtime.h>

#define TILE 16

__global__ void tiledMatMul(float *A, float *B, float *C, int N)
{
    // Shared-memory tiles
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    // Which element of C this thread computes
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    // Move through A and B one tile at a time
    for (int t = 0; t < N / TILE; t++)
    {
        // Load one element into shared memory
        tileA[threadIdx.y][threadIdx.x] =
            A[row * N + t * TILE + threadIdx.x];

        tileB[threadIdx.y][threadIdx.x] =
            B[(t * TILE + threadIdx.y) * N + col];

        // Wait until the entire tile is loaded
        __syncthreads();

        // Multiply the two tiles
        for (int k = 0; k < TILE; k++)
        {
            sum += tileA[threadIdx.y][k] *
                   tileB[k][threadIdx.x];
        }

        // Make sure nobody overwrites the tile too early
        __syncthreads();
    }

    // Store result
    C[row * N + col] = sum;
}

int main()
{
    int N = 1024;

    size_t size = N * N * sizeof(float);

    // Host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize matrices
    for (int i = 0; i < N * N; i++)
    {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Device memory
    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy A and B to GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 2D grid of 16x16 thread blocks
    dim3 threads(TILE, TILE);
    dim3 blocks(N / TILE, N / TILE);

    // Launch kernel
    tiledMatMul<<<blocks, threads>>>(d_A, d_B, d_C, N);

    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("C[0][0] = %f\n", h_C[0]);
    printf("C[N-1][N-1] = %f\n", h_C[N*N - 1]);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}