#include <stdio.h>
#include <limits.h>

#define N 5
#define INF INT_MAX

// CUDA kernel
__global__ void BFS(int graph[N][N], int dist[N], int level, int *changed)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;

    if (v >= N)
        return;

    // Process only vertices at current level
    if (dist[v] == level)
    {
        // Check all possible neighbors
        for (int u = 0; u < N; u++)
        {
            // There is an edge v -> u
            if (graph[v][u] == 1)
            {
                // If u has not been visited
                if (dist[u] > level + 1)
                {
                    dist[u] = level + 1;

                    // Some vertex was changed
                    atomicExch(changed, 1);
                }
            }
        }
    }
}

int main()
{
    int h_graph[N][N] =
    {
        {0, 1, 1, 0, 0},
        {0, 0, 0, 1, 0},
        {0, 0, 0, 0, 1},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0}
    };

    int h_dist[N];

    // Initialize distances to infinity
    for (int i = 0; i < N; i++)
        h_dist[i] = INF;

    // Source vertex = 0
    h_dist[0] = 0;

    // Device variables
    int (*d_graph)[N];
    int *d_dist;
    int *d_changed;

    // Allocate GPU memory
    cudaMalloc(&d_graph, sizeof(h_graph));
    cudaMalloc(&d_dist, sizeof(h_dist));
    cudaMalloc(&d_changed, sizeof(int));

    // Copy data CPU -> GPU
    cudaMemcpy(d_graph, h_graph,
               sizeof(h_graph),
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_dist, h_dist,
               sizeof(h_dist),
               cudaMemcpyHostToDevice);

    int level = 0;

    // One block containing N threads
    int threads = N;

    while (1)
    {
        int changed = 0;

        // changed = 0
        cudaMemcpy(d_changed, &changed,
                   sizeof(int),
                   cudaMemcpyHostToDevice);

        // Run BFS
        BFS<<<1, threads>>>(
            d_graph,
            d_dist,
            level,
            d_changed
        );

        // Wait for GPU
        cudaDeviceSynchronize();

        // Copy changed back to CPU
        cudaMemcpy(&changed, d_changed,
                   sizeof(int),
                   cudaMemcpyDeviceToHost);

        // No more vertices discovered
        if (changed == 0)
            break;

        level++;
    }

    // Copy distances GPU -> CPU
    cudaMemcpy(h_dist, d_dist,
               sizeof(h_dist),
               cudaMemcpyDeviceToHost);

    // Print result
    printf("BFS distances from vertex 0:\n");

    for (int i = 0; i < N; i++)
    {
        if (h_dist[i] == INF)
            printf("Vertex %d : INF\n", i);
        else
            printf("Vertex %d : %d\n", i, h_dist[i]);
    }

    // Free GPU memory
    cudaFree(d_graph);
    cudaFree(d_dist);
    cudaFree(d_changed);

    return 0;
}
