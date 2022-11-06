#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>

void *Cal(void* arg) {
	unsigned long long *input = (unsigned long long *) arg;
	unsigned long long *result = (unsigned long long *) malloc(sizeof(unsigned long long) * 1);
	result[0] = 0;
    for (unsigned long long i = input[1]; i < input[2]; i++)
		result[0] += ceil(sqrtl(input[0] - i * i));
    pthread_exit((void *) result);
}

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);

	cpu_set_t cpuset;
	sched_getaffinity(0, sizeof(cpuset), &cpuset);
	unsigned long long ncpus = CPU_COUNT(&cpuset);

	unsigned long long cut = r / ncpus;
	unsigned long long r2 = r * r;

	pthread_t threads[ncpus];
	unsigned long long input[8][3];
    for (int t = 0; t < ncpus; t++) {
		input[t][0] = r2;
        input[t][1] = cut * t;
		input[t][2] = (t == ncpus - 1 ? r : cut * (t + 1));
        pthread_create(&threads[t], NULL, Cal, (void*)input[t]);
    }

	unsigned long long sum = 0;
	void *res;
	for (int t = 0; t < ncpus; t++) {
        pthread_join(threads[t], &res);
		unsigned long long *result = (unsigned long long *) res;
		sum += result[0];
    }

	printf("%llu\n", (4 * (sum % k)) % k);
}
