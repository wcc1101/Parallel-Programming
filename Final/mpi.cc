#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <climits>
#include <mpi.h>

using namespace std;
int numCity;
int Rank, Size;

int **input(char* num) {
	numCity = atoi(num);
	string path = "testcases/";
	string filename = path + to_string(numCity) + ".txt";
    ifstream input(filename);

    int **graph = new int*[numCity];
    for (int i = 0; i < numCity; i++)
        graph[i] = new int[numCity];

    for (int i = 0; i < numCity; i++)
        for (int j = 0; j < numCity; j++)
            input >> graph[i][j];

	input.close();

	return graph;
}

int TSP(int **graph) {
	vector<int> cities;
	for (int i = 1; i < numCity; i++)
		cities.push_back(i);
	int minCost = INT_MAX;
	for (int j = Rank; j < cities.size(); j += Size) {
		auto newCities = cities;
		std::rotate(newCities.begin(), newCities.begin() + j, newCities.begin() + j + 1);
		do {
			int currentCost = 0;
			int k = 0; // start from 0
			for (int i = 0; i < newCities.size(); i++) {
				currentCost += graph[k][newCities[i]];
				k = newCities[i];
			}
			currentCost += graph[k][0];
			minCost = min(minCost, currentCost);
		} while (std::next_permutation(newCities.begin() + 1, newCities.end()));
	}
	return minCost;
}

int main(int argc, char** argv) {
	// omp_set_num_threads(4);
	// cout << omp_get_num_threads() << "threads" << endl;

	// Input
    int **graph = input(argv[1]);

    MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &Rank);
	MPI_Comm_size(MPI_COMM_WORLD, &Size);

    auto start = std::chrono::high_resolution_clock::now();

    int local_minCost = TSP(graph);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    double seconds = duration * 1.0 / 1000000;
    std::cout << "computation Duration: " << std::setprecision(3) << seconds << " seconds" << std::endl;

	int global_minCost = INT_MAX;

    start = std::chrono::high_resolution_clock::now();
	MPI_Reduce(&local_minCost, &global_minCost, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    seconds = duration * 1.0 / 1000000;
    std::cout << "communicate Duration: " << std::setprecision(3) << seconds << " seconds" << std::endl;

    if (Rank == 0)
        cout << "answer: " << global_minCost << endl;

	MPI_Finalize();

	return 0;
}
