// Sagi Shoffer

# include "CudaHeader.h"

__global__ void calcValues(float* a, float* b, float* obs, int obsSize, int statesSize, float* emissions, int size)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int oIdx = (i / statesSize);

	if (i < size) {
		emissions[i] = log((a[i%statesSize])*exp(-1 * pow(obs[oIdx] - b[i%statesSize], 2)));
	}
}

float* dev_a = 0;
float* dev_b = 0;
float* dev_obs = 0;
float* dev_emissions = 0;

// Helper function for using CUDA to add vectors in parallel.
cudaError_t calcEmissions(float* vecA, float* vecB, int statesSize, float* obs, int obsSize, float** emissions)
{
	cudaError_t cudaStatus;

	int i;
	int total = statesSize*obsSize;
	//int threadsPerBlock = 1024;
	int blocks = (total + THREAD_BLOCK - 1) / THREAD_BLOCK;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate memory of 1-dimansion array for A vector 
	cudaStatus = cudaMalloc((void**)&dev_a, statesSize * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Allocate memory of 1-dimansion array for B vector
	cudaStatus = cudaMalloc((void**)&dev_b, statesSize * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Allocate memory for obs vector on device
	cudaStatus = cudaMalloc((void**)&dev_obs, obsSize * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Allocate 1-dimansion memory in device for the emission values
	cudaStatus = cudaMalloc((void**)&dev_emissions, statesSize * obsSize * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy obs vector from host to device
	cudaStatus = cudaMemcpy(dev_a, vecA, statesSize * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Copy obs vector from host to device
	cudaStatus = cudaMemcpy(dev_b, vecB, statesSize * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Copy obs vector from host to device
	cudaStatus = cudaMemcpy(dev_obs, obs, obsSize * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	calcValues <<< blocks, THREAD_BLOCK >>>(dev_a, dev_b, dev_obs, obsSize, statesSize, dev_emissions, total);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	for (i = 0; i < obsSize; i++)
	{
		cudaStatus = cudaMemcpy(emissions[i], &dev_emissions[i*statesSize], statesSize * sizeof(float), cudaMemcpyDeviceToHost);
	}
	
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_a);
	cudaFree(dev_a);
	cudaFree(dev_obs);
	cudaFree(dev_emissions);

	return cudaStatus;
}
