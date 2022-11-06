#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;
	unsigned long long r2 = r * r;
#pragma omp parallel for reduction(+:pixels)
	for (unsigned long long x = 0; x < r; x++) {
		pixels += ceil(sqrtl(r2 - x*x));
	}
	printf("%llu\n", (4 * (pixels % k)) % k);
}
