// Sagi Shoffer

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>

#define THREAD_BLOCK 1024

cudaError_t calcEmissions(float* vecA, float* vecB, int statesSize, float* obs, int obsSize, float** emissions);