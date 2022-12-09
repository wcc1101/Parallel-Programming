#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define INF (1 << 30) - 1

int V, E, nThreads;
int **Dist;
pthread_barrier_t barrier;

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&V, sizeof(int), 1, file);
    fread(&E, sizeof(int), 1, file);

    Dist = (int **)malloc(sizeof(int *) * V);
    for (int i = 0; i < V; ++i) {
        Dist[i] = (int *)malloc(sizeof(int) * V);
        for (int j = 0; j < V; ++j)
            Dist[i][j] = (i == j) ? 0 : INF;
    }

    int pair[3];
    for (int i = 0; i < E; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0]][pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j)
            if (Dist[i][j] >= INF) Dist[i][j] = INF;
        fwrite(Dist[i], sizeof(int), V, outfile);
    }
    fclose(outfile);
}

void *calculate(void *arg) {
    int id = *(int *)arg;
    for (int k = 0; k < V; k ++) {
        for (int i = id; i < V; i += nThreads)
            for (int j = 0; j < V; j++)
                if (Dist[i][k] + Dist[k][j] < Dist[i][j])
                    Dist[i][j] = Dist[i][k] + Dist[k][j];
        pthread_barrier_wait(&barrier);
    }
    pthread_exit(NULL);
}

int main(int argc, char **argv) {
    cpu_set_t cpuset;
	sched_getaffinity(0, sizeof(cpuset), &cpuset);
	nThreads = CPU_COUNT(&cpuset);
	pthread_barrier_init(&barrier, NULL, nThreads);
    
    input(argv[1]);

    pthread_t threads[nThreads];
    int arg[nThreads];
    for (int i = 0; i < nThreads; i++) {
        arg[i] = i;
        pthread_create(&threads[i], NULL, calculate, (void *)&arg[i]);
    }
    for (int i = 0; i < nThreads; i++)
        pthread_join(threads[i], NULL);
    
	pthread_barrier_destroy(&barrier);

    output(argv[2]);
}