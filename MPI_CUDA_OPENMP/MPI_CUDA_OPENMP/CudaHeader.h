// Sagi Shoffer - 300989241

#define THREAD_BLOCK 1024

cudaError_t calcEmissions(float* vecA, float* vecB, int statesSize, float* obs, int obsSize, float** emissions);