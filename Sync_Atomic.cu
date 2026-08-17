#include <iostream>
#include <cuda_runtime.h>

__device__ int counter = 0;

// Global barrier
__device__ void barrier(int numBlocks)
{
    // Synchronize threads within the block
    __syncthreads();

    // One thread from each block increments counter
    if (threadIdx.x == 0)
    {
        atomicAdd(&counter, 1);
    }

    // Wait until all blocks reach the barrier
    if (threadIdx.x == 0)
    {
        while (atomicAdd(&counter, 0) < numBlocks)
        {
            // Busy wait
        }
    }

    // Release all threads in the block
    __syncthreads();
}

__global__ void syncKernel()
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    printf("Before barrier: Block %d, Thread %d\n", bid, tid);

    barrier(gridDim.x);

    printf("After barrier:  Block %d, Thread %d\n", bid, tid);
}

int main()
{
    int numBlocks = 4;
    int threadsPerBlock = 4;

    // Launch kernel
    syncKernel<<<numBlocks, threadsPerBlock>>>();

    // Wait for GPU to finish
    cudaError_t error = cudaDeviceSynchronize();

    if (error != cudaSuccess)
    {
        std::cerr << "CUDA Error: "
                  << cudaGetErrorString(error)
                  << std::endl;

        return 1;
    }

    return 0;
}