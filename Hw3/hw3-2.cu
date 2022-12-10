#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define INF (1 << 30) - 1
#define B 32

//======================
#define DEV_NO 0
cudaDeviceProp prop;

int *Dist;
int V, E, V_padding, rounds;

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&V, sizeof(int), 1, file);
    fread(&E, sizeof(int), 1, file);

    V_padding = (V % B == 0) ? V : (V / B + 1) * B;
    Dist = (int *)malloc(sizeof(int) * V_padding * V_padding);

    for (int i = 0; i < V_padding; ++i)
        for (int j = 0; j < V_padding; ++j)
            Dist[i * V_padding + j] = (i == j) ? 0 : INF;

    int pair[3];
    for (int i = 0; i < E; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0] * V_padding + pair[1]] = pair[2];
    }
    fclose(file);
    rounds = ceil(V_padding / B);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j)
            if (Dist[i * V_padding + j] >= INF) Dist[i * V_padding + j] = INF;
        fwrite(&Dist[i * V_padding], sizeof(int), V, outfile);
    }
    fclose(outfile);
}

int ceil(int a, int b) { return (a + b - 1) / b; }

__global__ void phase1(int *d, int round, int v) {
    int i = threadIdx.y + round * B;
    int j = threadIdx.x + round * B;

    if (i < v && j < v) {
        __shared__ int d_shared[B * B];

        int sharedIndexY = threadIdx.y * B;

        d_shared[sharedIndexY + threadIdx.x] = d[i * v + j];
        __syncthreads();

        for (int k = 0; k < B; k++) {
            int newWeight = d_shared[sharedIndexY + k] + d_shared[k * B + threadIdx.x];
            if (newWeight < d_shared[sharedIndexY + threadIdx.x])
                d_shared[sharedIndexY + threadIdx.x] = newWeight;
            __syncthreads();
        }

        d[i * v + j] = d_shared[sharedIndexY + threadIdx.x];
    }
}

__global__ void phase2(int *d, int round, int v) {
    if (blockIdx.x == round) // pivot block
        return;

    int i = threadIdx.y + blockIdx.x * B;
    int j = threadIdx.x + blockIdx.x * B;
    int pivot_i = threadIdx.y + round * B;
    int pivot_j = threadIdx.x + round * B;

    if (blockIdx.y == 0) // pivot row
        i = pivot_i;
    else // pivot column
        j = pivot_j;

    if (i < v && j < v) {
        __shared__ int pivot_shared[B * B];
        __shared__ int d_shared[B * B];

        int sharedIndexY = threadIdx.y * B;

        pivot_shared[sharedIndexY + threadIdx.x] = d[pivot_i * v + pivot_j];
        d_shared[sharedIndexY + threadIdx.x] = d[i * v + j];
        __syncthreads();

        for (int k = 0; k < B; k++) {
            int newWeight;
            if (blockIdx.y == 0)
                newWeight = pivot_shared[sharedIndexY + k] + d_shared[k * B + threadIdx.x];
            else
                newWeight = d_shared[sharedIndexY + k] + pivot_shared[k * B + threadIdx.x];

            if (newWeight < d_shared[sharedIndexY + threadIdx.x])
                d_shared[sharedIndexY + threadIdx.x] = newWeight;
        }

        d[i * v + j] = d_shared[sharedIndexY + threadIdx.x];
    }
}

__global__ void phase3(int *d, int round, int v) {
    if (blockIdx.x == round || blockIdx.y == round) // calculated
        return;

    int i = threadIdx.y + blockIdx.y * B;
    int j = threadIdx.x + blockIdx.x * B;

    if (i < v && j < v) {
        __shared__ int row_shared[B * B];
        __shared__ int col_shared[B * B];

        int sharedIndexY = threadIdx.y * B;

        row_shared[sharedIndexY + threadIdx.x] = d[i * v + (threadIdx.x + round * B)];
        col_shared[sharedIndexY + threadIdx.x] = d[(threadIdx.y + round * B) * v + j];
        __syncthreads();

        int weight = d[i * v + j];
        for (int k = 0; k < B; k++) {
            int newWeight = row_shared[sharedIndexY + k] + col_shared[k * B + threadIdx.x];
            if (newWeight < weight)
                weight = newWeight;
        }

        d[i * v + j] = weight;
    }
}

void block_FW(void) {
    int *Dist_device;
    cudaMalloc(&Dist_device, sizeof(int) * V_padding * V_padding);
    cudaMemcpy(Dist_device, Dist, sizeof(int) * V_padding * V_padding, cudaMemcpyHostToDevice);

    dim3 num_threads(B, B), num_blocks_2(rounds, 2), num_blocks_3(rounds, rounds);

    for (int round = 0; round < rounds; round++) {
        phase1<<<1, num_threads>>>(Dist_device, round, V_padding);
        phase2<<<num_blocks_2, num_threads>>>(Dist_device, round, V_padding);
        phase3<<<num_blocks_3, num_threads>>>(Dist_device, round, V_padding);
    }

    cudaMemcpy(Dist, Dist_device, sizeof(int) * V_padding * V_padding, cudaMemcpyDeviceToHost);
}

int main(int argc, char* argv[]) {
    cudaGetDeviceProperties(&prop, DEV_NO);
    printf("maxThreasPerBlock = %d, sharedMemPerBlock = %d\n", prop.maxThreadsPerBlock, prop.sharedMemPerBlock);
    input(argv[1]);
    printf("rounds: %d\n", rounds);
    block_FW();
    output(argv[2]);
    return 0;
}