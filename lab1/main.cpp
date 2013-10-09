#pragma comment(lib, "mpi.lib")
#pragma comment(lib, "cxx.lib")

#define MASTER_NODE 0 
#define MAX_ABS_ELEMENT_VALUE 100

#include <iostream>
#include <fstream>

#include <random>
#include <string>
#include <algorithm>
#include <ctime>
#include <cmath>
#include <cstring>

#include <mpi.h>
#include <omp.h>

using std::cout;
using std::endl;

int** generateMatrix(int size, bool fillRandom = false) {
    int** matrix = (int**)std::malloc(size * sizeof(int*));
    for (int i = 0; i < size; ++i) {
        matrix[i] = (int*)std::malloc(size * sizeof(int));
    }    

    if (fillRandom) {
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                matrix[i][j] = rand() % (2 * MAX_ABS_ELEMENT_VALUE + 1) - MAX_ABS_ELEMENT_VALUE;
            }
        }
    }

    return matrix;
}

void destroyMatrix(int** matrix, int size) {
    for (int i = 0; i < size; ++i) {
        std::free(matrix[i]);
    }    
    free(matrix);
}

void printMatrix(int** matrix, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            cout << matrix[i][j] << " ";       
        }
        cout << endl;
    }
}

void multiplyMatrices(int** A, int** B, int**C, int size, int ompCount = 1) {
    int oldOmpCount = omp_get_num_threads();
    omp_set_num_threads(ompCount);
#pragma omp parallel for
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            C[i][j] = 0;    
            for (int k = 0; k < size; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        } 
    }
    omp_set_num_threads(oldOmpCount);
}

int main(int argc, char** argv) {
	double globalsum;
	int generateByNode;
	double startWtime = 0;
	double endWtime;
	int globalBinSizes[100];

	MPI::Init(argc, argv);

    int **matrixA, **matrixB, **matrixC;

    int nOMP = std::atoi(argv[1]);
	int mSize = std::atoi(argv[2]);
	bool check = std::atoi(argv[2]) == 1;

	int numProcs = MPI::COMM_WORLD.Get_size();
	int nodeId = MPI::COMM_WORLD.Get_rank();

	omp_set_dynamic(0);

#ifdef _DEBUG
	int namelen;
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	MPI::Get_processor_name(processor_name, namelen);
	cout << "Initialized MPI worker " << nodeId << " of " << numProcs << " on " <<
		processor_name << endl;
#endif

	

#ifdef _DEBUG
    if (nodeId == MASTER_NODE) {
		cout << "Master: Generating matrices " << mSize << "x" << mSize << endl;
	} else {
        cout << nodeId << ": I know matrix size: " << mSize << "x" << mSize << endl;
        cout << nodeId << ": I know nOMP: " << nOMP << endl;
    }    
#endif

	omp_set_num_threads(nOMP);

    if (nodeId == MASTER_NODE) {
        srand(time(0));
        matrixA = generateMatrix(mSize, true);
        matrixB = generateMatrix(mSize, true);
        matrixC = generateMatrix(mSize, false);
    }

#ifdef _DEBUG
    if (nodeId == MASTER_NODE) {
        cout << "Master: Generated matrices " << endl << "A:" << endl;
        printMatrix(matrixA, mSize);
        cout << "B:" << endl;
        printMatrix(matrixB, mSize);
    }
#endif
    if (nodeId == MASTER_NODE) {
		startWtime = MPI::Wtime();
	}

    



    if (nodeId == MASTER_NODE) {
        
        destroyMatrix(matrixA, mSize);
        destroyMatrix(matrixB, mSize);
        destroyMatrix(matrixC, mSize);
    }

	MPI::Finalize();
	return 0;
}