#pragma comment(lib, "mpi.lib")
#pragma comment(lib, "cxx.lib")

#define MASTER_NODE 0 
#define SEND_MATRIX_TAG 1
#define SEND_ANSWER_MATRIX_TAG 3

#define MAX_ABS_ELEMENT_VALUE 10

#include <iostream>
#include <fstream>

#include <random>
#include <string>
#include <algorithm>
#include <ctime>
#include <cmath>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>

using std::cout;
using std::endl;
using std::vector;

double fRand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

double* generateMatrix(int size, bool fillRandom = false) {
    double* matrix = (double*)std::malloc(size * size * sizeof(double)); 

    if (fillRandom) {
        for (int i = 0; i < size * size; ++i) {
            matrix[i] = fRand(-MAX_ABS_ELEMENT_VALUE, MAX_ABS_ELEMENT_VALUE);
        }
    }

    return matrix;
}

void destroyMatrix(double* matrix) {
    free(matrix);
}

void printMatrix(double* matrix, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            cout << matrix[i * size + j] << " ";       
        }
        cout << endl;
    }
}

void multiplyMatrices(double* A, double* B, double* C, int size, size_t ompCount = 1) {    
    size_t oldOmpCount = omp_get_num_threads();
    omp_set_num_threads(ompCount);    
#pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {            
            C[i * size + j] = 0;    
            for (int k = 0; k < size; ++k) {
                C[i * size + j] += A[i * size + k] * B[k * size + j];                
            }
        } 
    }
    omp_set_num_threads(oldOmpCount);
}

int main(int argc, char** argv) {
	double startWtime = 0;

	MPI::Init(argc, argv);

    double *matrixA = NULL, *matrixB = NULL, *matrixC = NULL;

    size_t nOMP = std::atoi(argv[1]);
	size_t mSize = std::atoi(argv[2]);
	bool check = std::atoi(argv[3]) == 1;

	size_t numProcs = MPI::COMM_WORLD.Get_size();
	size_t nodeId = MPI::COMM_WORLD.Get_rank();

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
    } else {
        matrixB = generateMatrix(mSize, false);
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

    if (numProcs == 1) {
        
        if (nodeId == MASTER_NODE) {
            
            multiplyMatrices(matrixA, matrixB, matrixC, mSize, nOMP);
        }
    } else {
        size_t numWorkingProcs = numProcs;
        if (numProcs > mSize + 1) {
            numWorkingProcs = mSize + 1;
        }
        size_t sliceSize = mSize / (numWorkingProcs - 1);
        MPI::COMM_WORLD.Bcast(matrixB, mSize * mSize, MPI::DOUBLE, MASTER_NODE);
        if (nodeId == MASTER_NODE) {             
            for (size_t i = 0; i + 1 < numWorkingProcs; ++i) {
                size_t from = i * sliceSize;
                size_t to = (i + 1) * sliceSize;
                if (i + 1 == numWorkingProcs - 1 && to != mSize) {
                    to = mSize;
                }
                size_t elementsCount = (to - from) * mSize;

                MPI::COMM_WORLD.Isend(matrixA + from * mSize, elementsCount, MPI::DOUBLE, i + 1, SEND_MATRIX_TAG);
            }
            for (size_t i = 0; i + 1 < numWorkingProcs; ++i) {
                size_t from = i * sliceSize;
                size_t to = (i + 1) * sliceSize;                
                if (i + 1 == numWorkingProcs - 1 && to != mSize) {
                    to = mSize;
                }
                size_t elementsCount = (to - from) * mSize;
                MPI::COMM_WORLD.Recv(matrixC + from * mSize, elementsCount, MPI::DOUBLE, i + 1, SEND_ANSWER_MATRIX_TAG);
            }
        } else if (nodeId < numWorkingProcs) {
            size_t from = (nodeId - 1) * sliceSize;
            size_t to = nodeId * sliceSize;            
            if (nodeId == numWorkingProcs - 1 && to != mSize) {
                to = mSize;
            }
            size_t elementsCount = (to - from) * mSize;
            double* matrixRes = (double*)malloc(elementsCount * sizeof(double));
            double* matrixAslice = (double*)malloc(elementsCount * sizeof(double));     
            MPI::COMM_WORLD.Recv(matrixAslice, elementsCount, MPI::DOUBLE, MASTER_NODE, SEND_MATRIX_TAG);
            for (int i = 0; i + from < to; ++i) {
                #pragma omp parallel for
                for (int j = 0; j < (int)mSize; ++j) {
                    matrixRes[i * mSize + j] = 0;
                    for (int k = 0; k < (int)mSize; ++k) {
                        matrixRes[i * mSize + j] += matrixAslice[i * mSize + k] * matrixB[k * mSize + j];
                    }
                }
            }

            MPI::Request req = MPI::COMM_WORLD.Isend(matrixRes, elementsCount, MPI::DOUBLE, 0, SEND_ANSWER_MATRIX_TAG);
            
            free(matrixAslice);
            req.Wait();
            free(matrixRes);            
        }
    }

#ifdef _DEBUG
    if (nodeId == MASTER_NODE) {       
        cout << "Master: multiplied matrices, got C = " << endl;
        printMatrix(matrixC, mSize);
    }
#endif

    if (check) {
        if (nodeId == MASTER_NODE) {
            double* matrixC2 = generateMatrix(mSize, false);
            multiplyMatrices(matrixA, matrixB, matrixC2, mSize, 1);
#ifdef _DEBUG
            cout << "Master: checked multiplied matrices, got C = " << endl;
            printMatrix(matrixC2, mSize);
#endif
        }
    }

    if (nodeId == MASTER_NODE) {
        destroyMatrix(matrixA);
        destroyMatrix(matrixB);
        destroyMatrix(matrixC);
    } else {
        destroyMatrix(matrixB);
    }

    if (nodeId == MASTER_NODE) {
        cout << MPI::Wtime() - startWtime << endl;
    }

	MPI::Finalize();
	return 0;
}