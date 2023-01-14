#include <iostream>
#include <fstream>
#include <cstdlib>

int main() {
    // 定義二維陣列的大小.
    int numCity = 4;

    // 建立二維陣列.
    int **graph = new int*[numCity];
    for (int i = 0; i < numCity; i++)
        graph[i] = new int[numCity];

    // 填入亂數值.
    for (int i = 0; i < numCity; i++) {
        graph[i][i] = 0;
        for (int j = i + 1; j < numCity; j++) {
            graph[i][j] = graph[j][i] = rand() % 10 + 1;
        }
    }

    // 將二維陣列寫入到檔案中.
    std::ofstream outfile("4.txt");
    for (int i = 0; i < numCity; i++) {
        for (int j = 0; j < numCity; j++) {
            outfile << graph[i][j] << " ";
        }
        outfile << "\n";
    }
    outfile.close();

    return 0;
}
