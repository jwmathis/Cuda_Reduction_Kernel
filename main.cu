#include <iostream>
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>

using namespace std;
#define SHMEM_SIZE = 256 * 4; // Shared memory size
#define SIZE = 256;

// Function to check for CUDA errors and print error messages
void checkCudaError(cudaError_t err, const char* msg) {
	if (err != cudaSuccess) {
		cerr << "Error: " << msg << " - " << cudaGetErrorString(err) << endl;
		exit(EXIT_FAILURE);
	}
}


// xorshift128+ generator for random numbers
int xorshift128plus(uint64_t s[2]) {
	uint64_t x = s[0];
	uint64_t y = s[1];
	s[0] = y;
	x ^= x << 23;
	s[1] = x ^ y ^ (x >> 17) ^ (y >> 26);
	uint64_t result =  s[1] + y;

	// Cast to int (ensure it fits in the int range)
	return static_cast<int>(result & 0x7FFFFFFF); // Mask to ensure positive 32-bit int
}

// Generate random array using xorshift128+
vector<int> genereateRandomArray(uint64_t s[2], size_t size) {
	vector<int> random_array(size);

	for (size_t i = 0; i < size; i++) {
		random_array[i] = xorshift128plus(s);
	}

	return random_array;
}

// Define the kernel
__global__ void reduceKernel(int* inputArray, int* outputArray, int arraySize) {
	extern __shared__ int shared_sum[];

	// Calculate thread index
	int tid = blockIdx.x * blockDim.x + threadIdx.x; // Global index
	int stride = blockDim.x * gridDim.x; // Stride

	// Intialize partial sum
	int partial_sum = 0;

	// Perform reduction across threads
	for (int i = tid; i < arraySize; i += stride) {
		partial_sum += inputArray[i];
	}

	// Store partial sum in shared memory
	shared_sum[threadIdx.x] = partial_sum;

	// Synchronize threads
	__syncthreads();

	// Redeuce within the block
	for (int i = blockDim.x / 2; i > 0; i /= 2) {
		if (threadIdx.x < i) {
			shared_sum[threadIdx.x] += shared_sum[threadIdx.x + i];
		}

		// Synchronize threads
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		outputArray[blockIdx.x] = shared_sum[0];
	}
}

void intialize_vector(vector<int>& v, int n) {
	for (size_t i = 0; i < v.size(); i++) {
		v[i] = n;
	}
}

void launchReduceKernel(int* dev_array, int* dev_output, int array_size, int blockSize, int numBlocks) {

	reduceKernel <<<numBlocks, blockSize, blockSize * sizeof(int)>>> (dev_array, dev_output, array_size);
}

int main() {
	///-------------- Generate random array -------------///
	uint64_t state[2] = { 123456789, 987654321 };
	size_t array_size = 1024;
	vector<int> random_array = genereateRandomArray(state, array_size);
	//vector<int> random_array(array_size);
	//intialize_vector(random_array, 1);

	//	Print random array
	printf("Random Array: \n");
	for (const auto& num : random_array) {
		cout << num << "\n";
	}

	///-------------- Memory allocation -------------///
	int* dev_array; // Declare Device array
	size_t size = array_size * sizeof(int); // Size of array
	cudaError_t err = cudaMalloc(&dev_array, size); // Allocate memory for device variables
	checkCudaError(err, "cudaMalloc failed!");

	int* dev_result;
	err = cudaMalloc(&dev_result, sizeof(int) * ((array_size + 255) / 256));
	checkCudaError(err, "cudaMalloc failed!");

	///-------------- Copy data from host to device -------------///
	err = cudaMemcpy(dev_array, random_array.data(), size, cudaMemcpyHostToDevice); // Copy host data to device data
	checkCudaError(err, "cudaMemcpy failed!");


	///-------------- Launch kernel -------------///
	/// 
	///-------------- Fixed Block Size, Varying Grid Size -------------///
	int blockSize = 512; // Number of threads per block
	float elapsedTime = 0.0f;
	for (int numBlocks = 1; numBlocks <= (array_size + blockSize - 1) / blockSize; numBlocks *= 2) {// Number of blocks
		// Launch the kernel
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start); // Start timing
		launchReduceKernel(dev_array, dev_result, array_size, blockSize, numBlocks);
		cudaEventRecord(stop); // Stop timing

		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsedTime, start, stop);

		cout << "Grid Size: " << numBlocks << ", Block Size: " << blockSize << ", Execution Time: " << elapsedTime << " ms" << endl;
		cudaEventDestroy(start);
		cudaEventDestroy(stop);

		///------------- Retrieve and verify result -------------///
		vector<int> host_result(numBlocks);
		cudaMemcpy(host_result.data(), dev_result, sizeof(int)*numBlocks, cudaMemcpyDeviceToHost);

		// Perform final reduction on the CPU
		int gpu_result = 0;
		for (const auto& block_sum : host_result) {
			gpu_result += block_sum;
		}

		// Compute the result on the CPU
		int cpu_result = 0;
		auto cpu_start = chrono::high_resolution_clock::now();
		for (const auto& num : random_array) {
			cpu_result += num;
		}
		auto cpu_end = chrono::high_resolution_clock::now();
		chrono::duration<float, std::milli> cpu_elapsed = cpu_end - cpu_start;
		if (gpu_result == cpu_result) {
			cout << "Results match! Sum: " << gpu_result << endl;
		}
		else {
			cout << "Results do not match! GPU Sum: " << gpu_result << ", CPU Sum: " << cpu_result << endl;
		}
		cout << "CPU Execution Time: " << cpu_elapsed.count() << " ms" << endl;

	}

///-------------- Fixed Grid Size, Varying Block Size -------------///
	int numBlocks = (array_size + 1024 - 1) / 1024;
	for (int blockSize = 128; blockSize <= 1024; blockSize *= 2) {
		// Launch the kernel
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start); // Start timing
		launchReduceKernel(dev_array, dev_result, array_size, blockSize, numBlocks);
		cudaEventRecord(stop); // Stop timing

		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsedTime, start, stop);

		cout << "Grid Size: " << numBlocks << ", Block Size: " << blockSize << ", Execution Time: " << elapsedTime << " ms" << endl;
		cudaEventDestroy(start);
		cudaEventDestroy(stop);

		///------------- Retrieve and verify result -------------///
		vector<int> host_result(numBlocks);
		cudaMemcpy(host_result.data(), dev_result, sizeof(int) * numBlocks, cudaMemcpyDeviceToHost);

		// Perform final reduction on the CPU
		int gpu_result = 0;
		for (const auto& block_sum : host_result) {
			gpu_result += block_sum;
		}

		// Compute the result on the CPU
		int cpu_result = 0;
		auto cpu_start = chrono::high_resolution_clock::now();
		for (const auto& num : random_array) {
			cpu_result += num;
		}
		auto cpu_end = chrono::high_resolution_clock::now();
		chrono::duration<float, std::milli> cpu_elapsed = cpu_end - cpu_start;
		if (gpu_result == cpu_result) {
			cout << "Results match! Sum: " << gpu_result << endl;
		}
		else {
			cout << "Results do not match! GPU Sum: " << gpu_result << ", CPU Sum: " << cpu_result << endl;
		}
		cout << "CPU Execution Time: " << cpu_elapsed.count() << " ms" << endl;
	}

	///------------- Clean up -------------///
	cudaFree(dev_array);
	cudaFree(dev_result);
	return 0;

}