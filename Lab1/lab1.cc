#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	int id, num;
    MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &num);
	unsigned long long cut = r / num;
	unsigned long long r2 = r * r;
	unsigned long long local_pixels = 0;
	for (unsigned long long i = cut * id; i < (id == num - 1 ? r : cut * (id + 1)); i++)
		local_pixels += ceil(sqrtl(r2 - i * i));
	unsigned long long global_pixels;
	MPI_Reduce(&local_pixels, &global_pixels, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	if (id == 0)
		printf("%llu\n", (4 * (global_pixels % k)) % k);
	MPI_Finalize();
}
