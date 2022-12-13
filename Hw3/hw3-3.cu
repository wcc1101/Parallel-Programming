#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <omp.h>

#define INF (1 << 30) - 1
#define B 32

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
    int *edges = (int *)malloc(sizeof(int) * 3 * E);
    fread(edges, sizeof(int), 3 * E, file);
    for (int i = 0; i < E; ++i)
        Dist[edges[i * 3] * V_padding + edges[i * 3 + 1]] = edges[i * 3 + 2];
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
    __shared__ int d_shared[B * B];
    int sharedIndexY = threadIdx.y * B;
    d_shared[sharedIndexY + threadIdx.x] = d[i * v + j];
    __syncthreads();
    #pragma unroll 32
    for (int k = 0; k < B; k++) {
        d_shared[sharedIndexY + threadIdx.x] = min(d_shared[sharedIndexY + threadIdx.x], d_shared[sharedIndexY + k] + d_shared[k * B + threadIdx.x]);
        __syncthreads();
    }
    d[i * v + j] = d_shared[sharedIndexY + threadIdx.x];
}

__global__ void phase2(int *d, int round, int v) {
    if (blockIdx.x == round) // pivot block
        return;
    int ori_i = threadIdx.y + blockIdx.x * B;
    int ori_j = threadIdx.x + blockIdx.x * B;
    int pivot_i = threadIdx.y + round * B;
    int pivot_j = threadIdx.x + round * B;
    __shared__ int pivot_shared[B * B];
    __shared__ int row_shared[B * B];
    __shared__ int col_shared[B * B];
    int sharedIndexY = threadIdx.y * B;
    pivot_shared[sharedIndexY + threadIdx.x] = d[pivot_i * v + pivot_j];
    int i = pivot_i;
    int j = ori_j;
    row_shared[sharedIndexY + threadIdx.x] = d[i * v + j];
    i = ori_i;
    j = pivot_j;
    col_shared[sharedIndexY + threadIdx.x] = d[i * v + j];
    __syncthreads();
    #pragma unroll 32
    for (int k = 0; k < B; k++) {
        row_shared[sharedIndexY + threadIdx.x] = min(row_shared[sharedIndexY + threadIdx.x], pivot_shared[sharedIndexY + k] + row_shared[k * B + threadIdx.x]);
        col_shared[sharedIndexY + threadIdx.x] = min(col_shared[sharedIndexY + threadIdx.x], col_shared[sharedIndexY + k] + pivot_shared[k * B + threadIdx.x]);
    }
    d[i * v + j] = col_shared[sharedIndexY + threadIdx.x];
    i = pivot_i;
    j = ori_j;
    d[i * v + j] = row_shared[sharedIndexY + threadIdx.x];
}

__global__ void phase3(int *d, int round, int v, int offset) {
    int x = blockIdx.x;
    int y = blockIdx.y + offset;
    if (x == round || y == round) // calculated
        return;
    int i = threadIdx.y + y * B;
    int j = threadIdx.x + x * B;
    __shared__ int row_shared[B * B];
    __shared__ int col_shared[B * B];
    int sharedIndexY = threadIdx.y * B;
    row_shared[sharedIndexY + threadIdx.x] = d[i * v + (threadIdx.x + round * B)];
    col_shared[sharedIndexY + threadIdx.x] = d[(threadIdx.y + round * B) * v + j];
    __syncthreads();
    int weight = d[i * v + j];
    #pragma unroll 32
    for (int k = 0; k < B; k++) {
        weight = min(weight, row_shared[sharedIndexY + k] + col_shared[k * B + threadIdx.x]);
    }
    d[i * v + j] = weight;
}

void block_FW(void) {
    int device_count;
    cudaGetDeviceCount(&device_count);
    int *Dist_device[device_count];
    size_t size = sizeof(int) * V_padding * V_padding;
    cudaHostRegister(Dist_device, size, cudaHostRegisterDefault);
#pragma omp parallel
    {
        int id = omp_get_thread_num();
        int round_split = (rounds & 1 && id == 1) ? rounds / 2 + 1 : rounds / 2;
        int offset = id * rounds / 2;
        int curPos = offset * B * V_padding;
        int otherid =1 - id;

        cudaSetDevice(id);
        cudaMalloc(&Dist_device[id], size);
        cudaMemcpy(Dist_device[id], Dist, size, cudaMemcpyHostToDevice);

        dim3 num_threads(B, B), num_blocks_3(rounds, round_split);

        for (int round = 0; round < rounds; round++) {
            int curRow = round * B * V_padding;
            if ((round >= offset) && (round < offset + round_split))
                cudaMemcpy(Dist_device[otherid] + curRow, Dist_device[id] + curRow, sizeof(int) * B * V_padding, cudaMemcpyDeviceToDevice);
#pragma omp barrier
            phase1<<<1, num_threads>>>(Dist_device[id], round, V_padding);
            phase2<<<rounds, num_threads>>>(Dist_device[id], round, V_padding);
            phase3<<<num_blocks_3, num_threads>>>(Dist_device[id], round, V_padding, offset);
        }
#pragma omp barrier
        cudaMemcpy(Dist + curPos, Dist_device[id] + curPos, sizeof(int) * B * round_split * V_padding, cudaMemcpyDeviceToHost);
    }
}

int main(int argc, char* argv[]) {
    input(argv[1]);
    block_FW();
    output(argv[2]);
    return 0;
}