// Sagi Shoffer

#include "MainHeader.h"

int main(int args, char *argv[])
{
	int myid, numprocs, numOfSlaves, slaveNum;
	int i, j, size, index;
	int send, start, end, ans, mode;
	long double startTime1, endTime1, startTime2, endTime2, sumTime;

	float *obArray;				// Observation array
	float **transitionMat;		// Transition matrix
	float **abMat;				// Coffisients matrix
	float **emissionMat;		// Emssion matrix
	node **subMat = NULL;		// Sub matrix	
	int *pathArr = NULL;		// Path Array

	MPI_Status status;
	MPI_Init(&args, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

	allocateUtilitiesMatrix(&obArray, &transitionMat, &abMat, &emissionMat);

	/* ------------------------------------------------ MASTER ------------------------------------------------------------ */
	if (myid == 0) {
		//testRandValues(obArray, transitionMat, emissionMat);
		//testValues(obArray, transitionMat, emissionMat);
		loadArrayFromFile(obArray, OBSERVES, OBFILE);
		loadMatrixFromFile(transitionMat, STATES, STATES, TRFILE);
		loadABMatrixFromFile(abMat, EMISSION_COLS, STATES, ABFILE);

		startTime1 = MPI_Wtime();
		// Calculate log of transitions values
		for (i = 0; i < STATES; i++) {
#pragma omp parallel for private(j)
			for (j = 0; j < STATES; j++)
				transitionMat[i][j] = log(transitionMat[i][j]);
		}

		// calculate emissions values with cuda
		cudaError_t cudaStatus = calcEmissions(abMat[0], abMat[1], STATES, obArray, OBSERVES, emissionMat);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addwithcuda failed!");
			return 1;
		}
		endTime1 = MPI_Wtime();
		sumTime = endTime1 - startTime1;

		// cudaDeviceReset must be called before exiting in order for profiling and
		// tracing tools such as Nsight and Visual Profiler to show complete traces.
		cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceReset failed!");
			return 1;
		}

		/*printf("obArray:\n");
		printArray(obArray, OBSERVES);
		printf("\ntransition:\n");
		printMatrix(transitionMat, STATES, STATES);
		printf("\nemmition:\n");
		printMatrix(emissionMat, EMISSION_COLS, STATES);*/
	}

	// Brodcast all utilities to all proccess
	MPI_Bcast(obArray, OBSERVES, MPI_FLOAT, 0, MPI_COMM_WORLD);
	for (i = 0; i < STATES; i++)
		MPI_Bcast(&(transitionMat[i][0]), STATES, MPI_FLOAT, 0, MPI_COMM_WORLD);
	for (i = 0; i < OBSERVES; i++)
		MPI_Bcast(&(emissionMat[i][0]), STATES, MPI_INT, 0, MPI_COMM_WORLD);

	
	if (myid == 0) {	
		allocatePathArray(&pathArr, OBSERVES);
		
		// Initiate all process
		start = INITIATE; end = INITIATE;
		for (i = 1; i < numprocs; i++) {
			MPI_Send(&start, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			MPI_Send(&end, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
		}

		startTime2 = MPI_Wtime();
		// Look for zeros at the middle of observations array and send compatible mission
		for (i = 0; i < OBSERVES; i++) {
			// zero in the middle
			if (obArray[i] == 0) {
				end = i + 1;
				communicate(pathArr, start, end);
				start = i + 1;
			}
			// end of observations
			else if (i == OBSERVES - 1 && obArray[i] != 0) {
				end = i + 2;
				communicate(pathArr, start, end);
			}
		}

		// Gather all last results and abort all processes
		start = TERMINATE; end = TERMINATE;
		for (int i = 1; i < numprocs; i++) {
			communicate(pathArr, start, end);
		}
		endTime2 = MPI_Wtime();
		
		sumTime += endTime2 - startTime2;
		printf("\nExecution time is %lf\n", sumTime);
		ans = writePathToFile(pathArr, OBSERVES, REFILE);
		if (ans == 0)
			printf("File Problem!");

		freeMasterAllocations(obArray, transitionMat, abMat, emissionMat);

		/*printf("Master path array:\n");
		printPathArray(pathArr, OBSERVES);
		printf("\n");*/
	}

	/* ------------------------------------------------ SLAVE ------------------------------------------------------------ */
	else {
		while (1) {
			MPI_Recv(&start, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
			MPI_Recv(&end, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

			mode = CONTINUE;

			if (start == TERMINATE && end == TERMINATE) {
				freeSlaveAllocations(obArray, transitionMat, emissionMat);
				break;
			}

			else if (start != INITIATE || end != INITIATE) {
				size = end - start;
				allocateSubMatrix(&subMat, size);
				allocatePathArray(&pathArr, size);

				// Initiate values
				initSubVal(subMat, size);

				// Check initiated sub matrix
				/*printf("\n");
				printSubMat(subMat, size);*/

				// Run viterbi algorithm for sub matrix
				forwardViterbi(subMat, transitionMat, emissionMat, obArray, size, start);

				// Print sub matrix result
				/*printf("after:\n");
				printSubMat(subMat, size);*/

				// Build path
				buildPath(pathArr, subMat, size);

				// Print path of subMat
				/*printf("Path arr: \n");
				printPathArray(pathArr, size);
				printf("\n\n");*/

				mode = ACCEPT;
			}

			// Send to master slave id and mode (continue/accept)
			MPI_Send(&myid, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
			MPI_Send(&mode, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);


			if (mode == ACCEPT) {
				MPI_Send(&size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);		// How many elements
				MPI_Send(&start, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);		// Index of result
				MPI_Send(pathArr, size, MPI_INT, 0, 0, MPI_COMM_WORLD);	// Elements of slave path

				freeSlaveSubMat(subMat, size, pathArr);					// Free subMat & pathArr 
			}
		}
	}

	MPI_Finalize();
}

/* #########################################  Allocation Functions  ################################################### */

/* Allocate to every process, space for utilities matrixes and arrays */
void allocateUtilitiesMatrix(float **obArray, float ***transitionMat, float ***abMat, float ***emissionMat)
{
	int i;

	// Array of OBSERVES  
	*obArray = (float*)malloc(OBSERVES * sizeof(float));

	// Transition matrix STATES x STATES 
	*transitionMat = (float**)malloc(STATES * sizeof(float*));
	for (i = 0; i < STATES; i++)
		(*transitionMat)[i] = (float*)malloc(STATES * sizeof(float));

	// AB matrix (a,b) x STATES
	*abMat = (float**)malloc(EMISSION_COLS * sizeof(float*));
	for (i = 0; i < EMISSION_COLS; i++)
		(*abMat)[i] = (float*)malloc(STATES * sizeof(float));

	// Emission matrix OBSERVES x STATES
	*emissionMat = (float**)malloc(OBSERVES * sizeof(float*));
	for (i = 0; i < OBSERVES; i++)
		(*emissionMat)[i] = (float*)malloc(STATES * sizeof(float));
}

/* Allocate space for path array */
void allocatePathArray(int **pathArr, int size)
{
	*pathArr = (int*)malloc(size * sizeof(int));
}

/* Allocate space for sub matrix */
void allocateSubMatrix(node ***subMat, int size)
{
	int i;

	*subMat = (node**)malloc(size*sizeof(node*));
	for (i = 0; i < size; i++)
		(*subMat)[i] = (node*)malloc(STATES * sizeof(node));
}

/* ###########################################  Values Functions  ##################################################### */

/* Initiate values for sub matrix */
void initSubVal(node *subMat[], int size)
{
	int i, j;

	for (i = 0; i < size; i++) {
		for (j = 0; j < STATES; j++) {
			(subMat[i][j]).nodeProb = 0;
			(subMat[i][j]).nodeParent = 0;
		}
	}
}

/* A small example of values OBSERVES 5 STATES 3 */
void testValues(float observe[], float *trans[], float *ab[])
{
	trans[0][0] = 8.0; trans[0][1] = 5.0; trans[0][2] = 3.0;
	trans[1][0] = 4.0; trans[1][1] = 5.0; trans[1][2] = 9.0;
	trans[2][0] = 3.0; trans[2][1] = 8.0; trans[2][2] = 9.0;

	ab[0][0] = 2; ab[0][1] = 6; ab[0][2] = 5;
	ab[1][0] = 9; ab[1][1] = 5; ab[1][2] = 8;

	observe[0] = 5; observe[1] = 5; observe[2] = 0; observe[3] = 4; observe[4] = 5;
}

/* Random values for all utilities */
void testRandValues(float observe[], float *trans[], float *ab[])
{
	int i, j;
	srand(time(NULL));

	for (i = 0; i < OBSERVES; i++) {
		observe[i] = ((float)rand() / (RAND_MAX));
	}
	observe[2] = 0;
	//observe[94] = 0;
	//observe[1500] = 0;
	//observe[5400] = 0;
	//observe[7600] = 0;
	//observe[8900] = 0;


	for (i = 0; i < STATES; i++) {
		for (j = 0; j < STATES; j++) {
			trans[i][j] = ((float)rand() / (RAND_MAX));
		}
	}

	for (i = 0; i < EMISSION_COLS; i++) {
		for (j = 0; j < STATES; j++) {
			ab[i][j] = ((float)rand() / (RAND_MAX));
		}
	}
}

/* ##########################################  Utilities Functions  ################################################### */

/* Master's function to communicate with specific slave */
void communicate(int *pathArr, int start, int end)
{
	int slaveNum, mode, size, index;
	MPI_Status status;

	MPI_Recv(&slaveNum, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
	MPI_Recv(&mode, 1, MPI_INT, slaveNum, 0, MPI_COMM_WORLD, &status);
	if (mode == ACCEPT) {
		MPI_Recv(&size, 1, MPI_INT, slaveNum, 0, MPI_COMM_WORLD, &status);				// How many elements
		MPI_Recv(&index, 1, MPI_INT, slaveNum, 0, MPI_COMM_WORLD, &status);				// Index of result
		MPI_Recv(pathArr + index, size, MPI_INT, slaveNum, 0, MPI_COMM_WORLD, &status);	// Get elements to main path array			
	}
	MPI_Send(&start, 1, MPI_INT, slaveNum, 0, MPI_COMM_WORLD);
	MPI_Send(&end, 1, MPI_INT, slaveNum, 0, MPI_COMM_WORLD);
}

/* Calculation of viterbi algorithm on slave's sub matrix */
void forwardViterbi(node *subMat[], float *transitionMat[], float *emissionMat[], float *obArray, int size, int start)
{
	int curState, nextState, observe, curParent;
	float prob, curProb;

	// Calculation
	for (observe = 0; observe < size - 1; observe++)
	{
#pragma omp parallel for private(prob, nextState, curState, curProb, curParent)
		for (nextState = 0; nextState < STATES; nextState++)
		{
			// Assgin first prob with first arguments
			curProb = (subMat[observe][0]).nodeProb + emissionMat[observe + start][0] + transitionMat[0][nextState];
			curParent = 0;

			for (curState = 1; curState < STATES; curState++)
			{
				prob = (subMat[observe][curState]).nodeProb + emissionMat[observe + start][curState] + transitionMat[curState][nextState];

				if (prob > curProb) {
					curProb = prob;
					curParent = curState;
				}
			}

			(subMat[observe + 1][nextState]).nodeProb = curProb;
			(subMat[observe + 1][nextState]).nodeParent = curParent;
		}
	}
}

/* Builds path on slave's sub matrix - back propagation */
void buildPath(int *pathArr, node *subMat[], int size)
{
	int lastRow, i, parent, lastNode;
	float max;

	lastRow = size - 1;
	lastNode = 0;
	max = subMat[lastRow][0].nodeProb;
	parent = subMat[lastRow][0].nodeParent;
	// Look for max prob at last row and build path backward
	for (i = 1; i < STATES; i++) {
		if (subMat[lastRow][i].nodeProb > max) {
			lastNode = i;
			max = subMat[lastRow][i].nodeProb;
			parent = subMat[lastRow][i].nodeParent;
		}
	}
	//printf("max = %.2f, parent = %d\n", max, parent);

	pathArr[lastRow] = lastNode;
	for (i = lastRow - 1; i >= 0; i--) {
		pathArr[i] = parent;
		parent = subMat[i][parent].nodeParent;
	}
}

/* Evaluate emission function */
float emssionFunc(float aj, float bj, float oi)
{
	return aj * exp(-1 * (pow(oi - bj, 2)));
}

/* #############################################  Files Functions  ##################################################### */

/* Loading data from file to array */
int loadArrayFromFile(float arr[], int size, char fpath[])
{
	FILE* f = fopen(fpath, "r+");

	if (f == NULL)
	{
		printf("\nFailed opening the file..\n");
		return 0;
	}

	for (int i = 0; i < size; i++)
	{
		fscanf(f, "%f", &arr[i]);
	}

	fclose(f);
	return 1;
}

/* Loading data from file to matrix */
int loadMatrixFromFile(float *mat[], int rows, int cols, char fpath[])
{
	FILE* f = fopen(fpath, "r+");

	if (f == NULL)
	{
		printf("\nFailed opening the file..\n");
		return 0;
	}

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			fscanf(f, "%f", &mat[i][j]);
		}
	}

	fclose(f);
	return 1;
}

/* Loading data from file to transposed matrix */
int loadABMatrixFromFile(float *mat[], int rows, int cols, char fpath[])
{
	FILE* f = fopen(fpath, "r+");

	if (f == NULL)
	{
		printf("\nFailed opening the file..\n");
		return 0;
	}

	for (int i = 0; i < cols; i++)
	{
		for (int j = 0; j < rows; j++)
		{
			fscanf(f, "%f", &mat[j][i]);
		}
	}


	fclose(f);
	return 1;
}

/* Writing data from array to file */
int writePathToFile(int arr[], int size, char fpath[])
{
	FILE* f = fopen(fpath, "w");

	if (f == NULL) {
		printf("\nFailed opening the file..\n");
		return 0;
	}

	fprintf(f,"The output of State Transition Path:\n");
	for (int i = 0; i < size; i++) {
		if (i != size - 1)
			fprintf(f, "O[%d]-%d-> ", i, arr[i]);
		else
			fprintf(f, "O[%d]-%d", i, arr[i]);
	}

	fclose(f);
	return 1;
}

/* ###########################################  Free Functions  ##################################################### */

/* Free allocations of master */
void freeMasterAllocations(float *obArray, float **transitionMat, float **abMat, float **emissionMat) {
	int i;

	free(obArray);

	for (i = 0; i < STATES; i++)
		free(transitionMat[i]);
	free(transitionMat);

	for (i = 0; i < EMISSION_COLS; i++)
		free(abMat[i]);
	free(abMat);

	for (i = 0; i < OBSERVES; i++)
		free(emissionMat[i]);
	free(emissionMat);
}

/* Free allocations of sub matrix */
void freeSlaveSubMat(node **subMat, int size, int *pathArr) {
	int i;

	for (i = 0; i < size; i++)
		free(subMat[i]);
	free(subMat);
	free(pathArr);
}

/* Free allocations of slave */
void freeSlaveAllocations(float *obArray, float **transitionMat, float **emissionMat) {
	int i;

	free(obArray);

	for (i = 0; i < STATES; i++)
		free(transitionMat[i]);
	free(transitionMat);

	for (i = 0; i < OBSERVES; i++)
		free(emissionMat[i]);
	free(emissionMat);
}

/* ###########################################  Prints Functions  ##################################################### */

/* Print array */
void printArray(float arr[], int size)
{
	int i;

	for (i = 0; i < size; i++)
		printf("%.2f  ", arr[i]);
	printf("\n");
}

/* Print path array */
void printPathArray(int arr[], int size)
{
	int i;

	for (i = 0; i < size; i++)
		printf("%d  ", arr[i]);
	printf("\n");
}

/* Print matrix */
void printMatrix(float *mat[], int rows, int cols)
{
	int i, j;

	for (i = 0; i < rows; i++)
	{
		for (j = 0; j < cols; j++)
		{
			printf("%.2f  ", mat[i][j]);
		}
		printf("\n");
	}
}

/* Print matrix of nodes */
void printSubMat(node *subMat[], int size)
{
	int i, j;

	for (i = 0; i < size; i++)
	{
		for (j = 0; j < STATES; j++)
		{
			printf("(%f, %d)  ", subMat[i][j].nodeProb, subMat[i][j].nodeParent);
		}
		printf("\n");
	}
}