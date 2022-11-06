#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	int rank, size;
    MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	unsigned long long cut = r / size;
	unsigned long long r2 = r * r;
	unsigned long long local_pixels = 0;

#pragma omp parallel for reduction(+:local_pixels)
	for (unsigned long long i = cut * rank; i < (rank == size - 1 ? r : cut * (rank + 1)); i++)
		local_pixels += ceil(sqrtl(r2 - i * i));
	unsigned long long global_pixels;
	MPI_Reduce(&local_pixels, &global_pixels, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	if (rank == 0)
		printf("%llu\n", (4 * (global_pixels % k)) % k);
	MPI_Finalize();
}
