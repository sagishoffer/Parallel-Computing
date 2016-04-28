// Sagi Shoffer

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "mpi.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include "math.h"

#define _CRT_SECURE_NO_DEPRECATE
#define STATES 1000
#define OBSERVES 30000
#define EMISSION_COLS 2
#define INITIATE 0
#define TERMINATE -1
#define ACCEPT 1
#define CONTINUE 2
#define OBFILE "C:\\Users\\Sagi\\Parallel\\Observation.txt"
#define TRFILE "C:\\Users\\Sagi\\Parallel\\Transition.txt"
#define ABFILE "C:\\Users\\Sagi\\Parallel\\AB.txt"
#define REFILE "C:\\Users\\Sagi\\Parallel\\Results.txt"

struct Node
{
	float nodeProb;
	int nodeParent;
} typedef node;

// Signatures
void allocateUtilitiesMatrix(float **obArray, float ***transitionMat, float ***abMat, float ***emissionMat);
void allocatePathArray(int **pathArr, int size);
void allocateSubMatrix(node ***subMat, int size);
void initSubVal(node *subMat[], int size);
void testValues(float observe[], float *trans[], float *ab[]);
void testRandValues(float observe[], float *trans[], float *ab[]);
float emssionFunc(float aj, float bj, float oi);
void printArray(float arr[], int size);
void printMatrix(float *mat[], int rows, int cols);
void printSubMat(node *subMat[], int size);
void forwardViterbi(node *subMat[], float *transitionMat[], float *emissionMat[], float *obArray, int size, int start);
void buildPath(int *pathArr, node *subMat[], int size);
void printPathArray(int arr[], int size);
void communicate(int *pathArr, int start, int end);

int loadArrayFromFile(float arr[], int size, char fpath[]);
int loadMatrixFromFile(float *mat[], int rows, int cols, char fpath[]);
int loadABMatrixFromFile(float *mat[], int rows, int cols, char fpath[]);
int writePathToFile(int arr[], int size, char fpath[]);

void freeMasterAllocations(float *obArray, float **transitionMat, float **abMat, float **emissionMat);
void freeSlaveSubMat(node **subMat, int size, int *pathArr);
void freeSlaveAllocations(float *obArray, float **transitionMat, float **emissionMat);

cudaError_t calcEmissions(float* vecA, float* vecB, int statesSize, float* obs, int obsSize, float** emissions);