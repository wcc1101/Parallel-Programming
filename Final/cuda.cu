#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <climits>
#include <cuda.h>

#define MAX_THREADS 1024
#define MAX_BLOCKS 30
#define MAX_PERMS 5041

using namespace std;
int size, *graphWeights, minCost;
__device__ __shared__ int shared_cost;


void input(char* num) {
	size = atoi(num);
	string path = "testcases/";
	string filename = path + to_string(size) + ".txt";
    ifstream input(filename);

    graphWeights = new int[size * size];

    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            input >> graphWeights[i * size + j];

	input.close();
}

__host__ unsigned long long factorial(int n) {
	int c;
	unsigned long long result = 1;

	for (c = 1; c <= n; c++)
		result = result * c;

	return result;
}

__device__ void swap(int *x, int *y) { int tmp = *x; *x = *y;	*y = tmp; }

__device__ void reverse(int *first, int *last) { while ((first != last) && (first != --last)) swap(first++, last); }


__host__ void h_swap(int *x, int *y) { int tmp = *x; *x = *y;	*y = tmp; }

__host__ void h_reverse(int *first, int *last) { while ((first != last) && (first != --last)) h_swap(first++, last); }

__device__ void calPath(int * path, int * shortestPath, int * tcost, int * weights, int length, int tid) {
	int sum = 0;
	for (int i = 0; i < length; i++) {
		int val = weights[path[i] * length + path[(i + 1) % length]];
		if (val == -1) return;
		sum += val;
	}
	if (sum == 0) return;
	atomicMin(&shared_cost, sum);
	if (shared_cost == sum) {
		*tcost = sum;
		memcpy(shortestPath, path, length * sizeof(int));
	}
}

__device__ bool next_permutation(int * first, int * last) {
	if (first == last) return false;
	int * i = first;
	++i;
	if (i == last) return false;
	i = last;
	--i;

	for (;;) {
		int * ii = i--;
		if (*i < *ii) {
			int * j = last;
			while (!(*i < *--j));
			swap(i, j);
			reverse(ii, last);
			return true;
		}
		if (i == first) {
			reverse(first, last);
			return false;
		}
	}
}

__host__ bool h_next_permutation(int * first, int * last) {
	if (first == last) return false;
	int * i = first;
	++i;
	if (i == last) return false;
	i = last;
	--i;

	for (;;) {
		int * ii = i--;
		if (*i < *ii) {
			int * j = last;
			while (!(*i < *--j));
			h_swap(i, j);
			h_reverse(ii, last);
			return true;
		}
		if (i == first) {
			h_reverse(first, last);
			return false;
		}
	}
}

__host__ void print_Graph(int * graphWeights, int size) {
	int i, j;
	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			printf("%d\t", graphWeights[i * size + j]);
		}
		printf("\n");
	}
}

__host__ void print_ShortestPath(int * shortestPath, int cost, int size) {
	int i;
	if (cost == (size * 100)) printf("no possible path found.\n");
	else {
		for (i = 0; i < size; i++) {
			printf("%d\t", shortestPath[i]);
		}
		printf("\nCost: %d\n", cost);
	}
}

__global__ void find_permutations_for_threads(int * city_ids, int * k, int * choices, int * size, unsigned long long * threads_per_kernel) {
	int length = *size;
	int index = 1;
	unsigned long long count = 0;
	for (count = 0; count < *threads_per_kernel; count++) {
		for (int i = 0; i < length; i++) {
			choices[i + count * length] = city_ids[i];
		}
		reverse(city_ids + *k + index, city_ids + length);
		next_permutation(city_ids + index, city_ids + length);
	}
}

__host__ void h_find_permutations_for_threads(int *city_ids, int k, int * choices, int size, unsigned long long threads_per_kernel) {
	int length = size;
	int index = 1;
	unsigned long long count = 0;
	for (count = 0; count < threads_per_kernel; count++) {
		for (int i = 0; i < length; i++) {
			choices[i + count * length] = city_ids[i];
            
		}
		h_reverse(city_ids + k + index, city_ids + length);
		h_next_permutation(city_ids + index, city_ids + length);
	}
    for (int i = 0; i < length * 3; i++)
        cout << choices[i]<<" ";
}

__global__ void TSPKernel(int * choices, int * k, int * shortestPath, int * graphWeights, int * cost, int * size) {
	uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	int length = *size;
	int index = 1;

	/* local variables */
	int * _path, *_shortestPath;
	int _tcost;

	_path = (int *)malloc(length * sizeof(int));
	_shortestPath = (int *)malloc(length * sizeof(int));
	// _tcost = length * 100;

	memcpy(_path, choices + tid * length, length * sizeof(int));
	memcpy(_shortestPath, shortestPath, length * sizeof(int));

	if (threadIdx.x == 0) {
		if (cost[blockIdx.x] == 0) cost[blockIdx.x] = length * 100;
		shared_cost = length * 100;
	}

	__syncthreads();

	do {
		calPath(_path, _shortestPath, &_tcost, graphWeights, length, tid);
	} while (next_permutation(_path + *k + index, _path + length));

	if (_tcost == shared_cost) {
		atomicMin(&cost[blockIdx.x], _tcost);
		if (cost[blockIdx.x] == _tcost) {
			memcpy(shortestPath + blockIdx.x * length, _shortestPath, length * sizeof(int));
		}
	}


	free(_path);
	free(_shortestPath);
}

int main(int argc, char** argv) {

    // Input
    input(argv[1]);

    // print_Graph(graphWeights, size);

	int * city_ids, *shortestPath, *choices, *cost;
	unsigned long long total_permutations, thread_perms, num_blocks = 1, num_threads, num_kernels = 1;
	int selected_K = 0;
	unsigned long long threads_per_kernel;
	/* device variables */
	int *dev_shortestPath, *dev_graphWeights, *dev_choices;
	int * dev_cost, *dev_size;
	int * dev_selected_K;
	unsigned long long * dev_threads_per_kernel;
    minCost = INT_MAX;

	total_permutations = factorial(size - 1);
	printf("# of permutations for %d: %llu\n", size - 1, total_permutations);

    for (selected_K = 1; selected_K < size - 2; selected_K++) {
		thread_perms = factorial(size - 1 - selected_K);
		if (thread_perms < MAX_PERMS) break;
	}
	num_threads = total_permutations / thread_perms;
	int k;
	while (num_threads > MAX_THREADS) {
		k = 2;
		while (num_threads % k != 0) k++;
		num_threads /= k;
		num_blocks *= k;
	}
	while (num_blocks > MAX_BLOCKS) {
		k = 2;
		while (num_blocks % k != 0) k++;
		num_blocks /= k;
		num_kernels *= k;
	}
	threads_per_kernel = num_blocks * num_threads;
	// printf("K selected: %d\n", selected_K);
	printf("num_threads %llu thread_perms %llu num_blocks %llu num_kernels %llu threads_per_kernel %llu\n", num_threads, thread_perms, num_blocks, num_kernels, threads_per_kernel);

	dim3 block_dim(num_threads, 1, 1);
	dim3 grid_dim(num_blocks, 1, 1);

    std::cout << "start malloc" << std::endl;
	city_ids = (int *)malloc(size * sizeof(int));
	shortestPath = (int *)calloc(num_blocks * size, sizeof(int));
	cost = (int *)calloc(num_blocks, sizeof(int));
	choices = (int *)malloc(threads_per_kernel * size * sizeof(int));
    // for (int i = 0; i < size; i++){
	// 	city_ids[i] = i;
    //     choices[i] = size - i;
    //     choices[i + size] = i;
    // }
    // for (int j = 0; j < size * 3; j++)
    //     cout <<choices[j]<<" ";
    // cout << endl;

    // std::cout << "start cudaMalloc" << std::endl;
    int * dev_city_ids;
	cudaMalloc((void **)&dev_city_ids, size * sizeof(int));
	cudaMalloc((void **)&dev_shortestPath, size * sizeof(int) * num_blocks);
	cudaMalloc((void **)&dev_graphWeights, size * sizeof(int) * size);
	cudaMalloc((void **)&dev_cost, num_blocks * sizeof(int));
	cudaMalloc((void **)&dev_size, sizeof(int));
	cudaMalloc((void **)&dev_selected_K, sizeof(int));
	cudaMalloc((void **)&dev_choices, threads_per_kernel * size * sizeof(int));
	cudaMalloc((void **)&dev_threads_per_kernel, sizeof(unsigned long long));


    auto Mstart = std::chrono::high_resolution_clock::now();

	cudaMemcpy(dev_city_ids, city_ids, size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_shortestPath, shortestPath, size * sizeof(int) * num_blocks, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_graphWeights, graphWeights, size * sizeof(int) * size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_size, &size, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_selected_K, &selected_K, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_choices, choices, threads_per_kernel * size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_threads_per_kernel, &threads_per_kernel, sizeof(unsigned long long), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_cost, cost, num_blocks * sizeof(int), cudaMemcpyHostToDevice);

    auto Mend = std::chrono::high_resolution_clock::now();
    auto Mduration = std::chrono::duration_cast<std::chrono::microseconds>(Mend - Mstart).count();

    // cudaMemcpy(city_ids, dev_city_ids, size * sizeof(int), cudaMemcpyDeviceToHost);
    // for (int j = 0; j < size; j++)
    //     cout <<city_ids[j]<<" ";
    // h_find_permutations_for_threads(city_ids, selected_K, choices, size, threads_per_kernel);
    // for (int j = 0; j < 3* size; j++)
    //     cout <<choices[j]<<" ";
    // cout << endl;
    // return;
	float percentage;

    auto start = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < num_kernels; i++) {
		find_permutations_for_threads << < 1, 1 >> >(dev_city_ids, dev_selected_K, dev_choices, dev_size, dev_threads_per_kernel);
	    // cudaMemcpy(choices, dev_choices, threads_per_kernel * size * sizeof(int), cudaMemcpyDeviceToHost);
        // cout << threads_per_kernel << endl;
        // for (int j = 0; j < size * threads_per_kernel; j++)
        //     cout <<choices[j]<<" ";
		cudaGetLastError();
		cudaDeviceSynchronize();
	    TSPKernel << < grid_dim, block_dim >> > (dev_choices, dev_selected_K, dev_shortestPath, dev_graphWeights, dev_cost, dev_size);
		cudaGetLastError();
		cudaDeviceSynchronize();
		percentage = (100. / (float) num_kernels * (float)(i + 1));
		printf("\rProgress : ");
		for (int j = 0; j < 10; j++) {
			if ((percentage / 10) / j > 1) printf("#");
			else printf(" ");
		}
		printf(" [%.2f%%]", percentage);
		fflush(stdout);
	}

    auto end = std::chrono::high_resolution_clock::now();
    cout << endl;

	cudaMemcpy(shortestPath, dev_shortestPath, num_blocks * size * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(cost, dev_cost, num_blocks * sizeof(int), cudaMemcpyDeviceToHost);

    // cout << "print graphWeights" << endl;
    print_Graph(graphWeights, size);

    int min = cost[0];
    int index = 0;
    for (int i = 1; i < num_blocks; i++) {
        if (cost[i] < min) {
            min = cost[i];
            index = i;
        }
    }
    printf("Shortest path found on block #%d:\n", index + 1);
    print_ShortestPath(&shortestPath[index * size], min, size);

    // TSPKernel<<<grid_dim, block_dim>>>(d_graph, d_numCity, d_minCost);

    // cudaMemcpy((void *)minCost, (const void *)d_minCost, sizeof(int), cudaMemcpyDeviceToHost);

    // cout << endl << minCost << endl;

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    double seconds = duration * 1.0 / 1000000;
    double Mseconds = Mduration * 1.0 / 1000000;

    std::cout << "Duration: " << std::setprecision(3) << seconds << " seconds" << std::endl;

    std::cout << "MDuration: " << std::setprecision(3) << Mseconds << " seconds" << std::endl;

	return 0;
}
