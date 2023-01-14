#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <climits>

using namespace std;
int numCity;

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
	do {
		int currentCost = 0;
		int k = 0; // start from 0
		for (int i = 0; i < cities.size(); i++) {
			currentCost += graph[k][cities[i]];
			k = cities[i];
		}
		currentCost += graph[k][0];
		minCost = min(minCost, currentCost);

	} while (next_permutation(cities.begin(), cities.end())); // try every path

	return minCost;
}

int main(int argc, char** argv) {

    // Input
    int **graph = input(argv[1]);

    auto start = std::chrono::high_resolution_clock::now();

	cout << TSP(graph) << endl;

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    double seconds = duration * 1.0 / 1000000;

    std::cout << "Duration: " << std::setprecision(3) << seconds << " seconds" << std::endl;

	return 0;
}
