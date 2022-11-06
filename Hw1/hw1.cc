#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <boost/sort/spreadsort/spreadsort.hpp>

#define ODD 1
#define EVEN 2

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

	MPI_File input_file, output_file;
	MPI_Status status;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

	int totalNum = atoi(argv[1]);
	// each process sorts 1 / n part.
	int partition = std::ceil(totalNum / (double)size);
	// last process sorts the remaining part.
	int partitionSize = (rank >= totalNum) ? 0 : (rank == size - 1) ? (totalNum - partition * rank) : partition;

	// input
	float *data = new float[partitionSize];
	MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
	if (partitionSize != 0)
		MPI_File_read_at(input_file, sizeof(float) * (rank * partition), data, partitionSize, MPI_FLOAT, MPI_STATUS_IGNORE);
	MPI_File_close(&input_file);

	// sort each part.
	boost::sort::spreadsort::spreadsort(data, data + partitionSize);

	// calculate partner's rank.
	int evenPartnerRank = (rank & 1) ? rank - 1 : rank + 1;
	int oddPartnerRank = (rank & 1) ? rank + 1 : rank - 1;
	// edges
	if (oddPartnerRank == size || oddPartnerRank >= totalNum)
		oddPartnerRank = -1;
	if (evenPartnerRank == size || evenPartnerRank >= totalNum)
		evenPartnerRank = -1;

	// calculate partner's partitionSize.
	int evenPartnerSize = (evenPartnerRank == size - 1) ? totalNum - partition * rank : partition;
	int oddPartnerSize = (oddPartnerRank == size - 1) ? totalNum - partition * rank : partition;

	// allocate memory
	float *partnerData = new float[partition];
	float *tempData = new float[partitionSize];

	if (partitionSize != 0) {
		for (int round = 0; round <= size; round++) {
			if ((round & 1 && evenPartnerRank == -1) || (!(round & 1) && oddPartnerRank == -1))
				continue;
			if (round & 1) { // even round.
				if (!(rank & 1)) { // stores smaller part.
                    MPI_Sendrecv(&data[partitionSize - 1], 1, MPI_FLOAT, evenPartnerRank, EVEN, &partnerData[0], 1, MPI_FLOAT, evenPartnerRank, EVEN, MPI_COMM_WORLD, &status);
					if (data[partitionSize - 1] > partnerData[0]) {
						MPI_Sendrecv(data, partitionSize, MPI_FLOAT, evenPartnerRank, EVEN, partnerData, evenPartnerSize, MPI_FLOAT, evenPartnerRank, EVEN, MPI_COMM_WORLD, &status);
						memcpy(tempData, data, sizeof(float) * partitionSize);  
						for (int i(0), j(0), k(0); k < partitionSize; k++) {
                            if (j == evenPartnerSize || (i < partitionSize && tempData[i] < partnerData[j]))
                                data[k] = tempData[i++];
                            else
                                data[k] = partnerData[j++];
                        }
                    }
				} else if (rank & 1){ // stores bigger part.
                    MPI_Sendrecv(&data[0], 1, MPI_FLOAT, evenPartnerRank, EVEN, &partnerData[evenPartnerSize - 1], 1, MPI_FLOAT, evenPartnerRank, EVEN, MPI_COMM_WORLD, &status);
					if (data[0] < partnerData[evenPartnerSize - 1]) {
                        MPI_Sendrecv(data, partitionSize, MPI_FLOAT, evenPartnerRank, EVEN, partnerData, evenPartnerSize, MPI_FLOAT, evenPartnerRank, EVEN, MPI_COMM_WORLD, &status);
                        memcpy(tempData, data, sizeof(float) * partitionSize); 
						for (int i(partitionSize - 1), j(evenPartnerSize - 1), k(partitionSize - 1); k >= 0; k--) {
                            if (j == -1 || (i != -1 && tempData[i] > partnerData[j]))
                                data[k] = tempData[i--];
                            else
                                data[k] = partnerData[j--];
                        }
                    }
				}
			} else { // odd round.
				if (rank & 1) { // stores smaller part.
					MPI_Sendrecv(&data[partitionSize - 1], 1, MPI_FLOAT, oddPartnerRank, ODD, &partnerData[0], 1, MPI_FLOAT, oddPartnerRank, ODD, MPI_COMM_WORLD, &status);
                    if (data[partitionSize - 1] > partnerData[0]) {
                        MPI_Sendrecv(data, partitionSize, MPI_FLOAT, oddPartnerRank, ODD, partnerData, oddPartnerSize, MPI_FLOAT, oddPartnerRank, ODD, MPI_COMM_WORLD, &status);
                        memcpy(tempData, data, sizeof(float) * partitionSize); 
						for (int i(0), j(0), k(0); k < partitionSize; k++) {
                            if (j == oddPartnerSize || (i < partitionSize && tempData[i] < partnerData[j]))
                                data[k] = tempData[i++];
                            else
                                data[k] = partnerData[j++];
                        }
                    }
				} else if (!(rank & 1)) { // stores bigger part.
					MPI_Sendrecv(&data[0], 1, MPI_FLOAT, oddPartnerRank, ODD, &partnerData[oddPartnerSize - 1], 1, MPI_FLOAT, oddPartnerRank, ODD, MPI_COMM_WORLD, &status);
                    if (data[0] < partnerData[oddPartnerSize - 1]) {
                        MPI_Sendrecv(data, partitionSize, MPI_FLOAT, oddPartnerRank, ODD, partnerData, oddPartnerSize, MPI_FLOAT, oddPartnerRank, ODD, MPI_COMM_WORLD, &status);
                        memcpy(tempData, data, sizeof(float) * partitionSize); 
						for (int i(partitionSize - 1), j(oddPartnerSize - 1), k(partitionSize - 1); k >= 0; k--) {
                            if (j == -1 || (i != -1 && tempData[i] > partnerData[j]))
                                data[k] = tempData[i--];
                            else
                                data[k] = partnerData[j--];
                        }
                    }
				}	
			}
		}
	}

	// output
	MPI_File_open(MPI_COMM_WORLD, argv[3], MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
	if (partitionSize != 0)
		MPI_File_write_at(output_file, sizeof(float) * (rank * partition), data, partitionSize, MPI_FLOAT, MPI_STATUS_IGNORE);
	MPI_File_close(&output_file);

	// release memory
	delete[] partnerData;
	delete[] tempData;
	delete[] data;

	MPI_Finalize();
    return 0;
}
