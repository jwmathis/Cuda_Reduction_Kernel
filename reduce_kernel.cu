#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Define the kernel
__global__ void reduceKernel(int* inputArray, int* outputArray, int arraySize) {
	extern __shared__ int shared_memory[];

	// Calculate index
	int tid = threadIdx.x + blockIdx.x * blockDim.x; // Global index; blockDim.x is the number of threads in a block and is a built-in variable
	int stride = blockDim.x * gridDim.x; // Stride: step size which a thread progresses through the data it processes

	// Intialize partial sum
	int partial_sum = 0;

	// Perform reduction across threads
	for (int i = tid; i < arraySize; i += stride) {
		partial_sum += inputArray[i];
	}

	// Store partial sum in shared memory
	shared_memory[threadIdx.x] = partial_sum;

	// Synchronize threads
	__syncthreads();

	// Redeuce within the block
	for (int i = blockDim.x / 2; i > 0; i /= 2) {
		if (threadIdx.x < i) {
			shared_memory[threadIdx.x] += shared_memory[threadIdx.x + i];
		}

		// Synchronize threads
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		outputArray[blockIdx.x] = shared_memory[0];
	}
}