#include <stdio.h>
#include <cuda_runtime.h>
#include <limits.h>

#define INF 999999

// One thread handles one vertex
__global__ void SSSP(int V, const int *src, const int *dst,
                     const int *weight, int E, int *dist)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;

    if (u < V) {
        for (int e = 0; e < E; e++) {
            if (src[e] == u && dist[u] != INF) {
                int v = dst[e];
                int newDist = dist[u] + weight[e];

                atomicMin(&dist[v], newDist);
            }
        }
    }
}

int main()
{

    int V = 4, E = 5;

    int h_src[]    = {0, 0, 2, 1, 2};
    int h_dst[]    = {1, 2, 1, 3, 3};
    int h_weight[] = {4, 1, 2, 1, 5};

    int *d_src, *d_dst, *d_weight, *d_dist;

    cudaMalloc(&d_src, E * sizeof(int));
    cudaMalloc(&d_dst, E * sizeof(int));
    cudaMalloc(&d_weight, E * sizeof(int));
    cudaMalloc(&d_dist, V * sizeof(int));

    cudaMemcpy(d_src, h_src, E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst, h_dst, E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight, E * sizeof(int), cudaMemcpyHostToDevice);

    int h_dist[] = {0, INF, INF, INF};
    cudaMemcpy(d_dist, h_dist, V * sizeof(int),
               cudaMemcpyHostToDevice);

    // Run V-1 times
    for (int i = 0; i < V - 1; i++) {
        SSSP<<<1, V>>>(V, d_src, d_dst, d_weight, E, d_dist);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(h_dist, d_dist, V * sizeof(int),
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < V; i++)
        printf("Distance from 0 to %d = %d\n", i, h_dist[i]);

    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_weight);
    cudaFree(d_dist);

    return 0;
}