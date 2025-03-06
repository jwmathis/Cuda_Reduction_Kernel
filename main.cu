#include <iostream>
#include <vector>
#include <cstdint>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

__global__ void reduceKernel(int* inputArray, int* outputArray, int arraySize);

// Function to check for CUDA errors and print error messages
void checkCudaError(cudaError_t err, const char* msg) {
	if (err != cudaSuccess) {
		cerr << "Error: " << msg << " - " << cudaGetErrorString(err) << endl;
		exit(EXIT_FAILURE);
	}
}


// xorshift128+ generator for random numbers
//int xorshift128plus(uint64_t s[2]) {
//	uint64_t x = s[0];
//	uint64_t y = s[1];
//	s[0] = y;
//	x ^= x << 23;
//	s[1] = x ^ y ^ (x >> 17) ^ (y >> 26);
//	uint64_t result =  s[1] + y;
//
//	// Cast to int (ensure it fits in the int range)
//	return static_cast<int>(result & 0x7FFFFFFF); // Mask to ensure positive 32-bit int
//}

int
xoroshiro128plus(uint64_t s[2])
{
	uint64_t s0 = s[0];
	uint64_t s1 = s[1];
	uint64_t result = s0 + s1;
	s1 ^= s0;
	s[0] = ((s0 << 55) | (s0 >> 9)) ^ s1 ^ (s1 << 14);
	s[1] = (s1 << 36) | (s1 >> 28);
	return static_cast<int>(result & 0x7FFFFFFF);
}


// Generate random array using xorshift128+
vector<int> genereateRandomArray(uint64_t s[2], size_t size) {
	vector<int> random_array(size);

	for (size_t i = 0; i < size; i++) {
		random_array[i] = xoroshiro128plus(s);
	}

	return random_array;
}

void launchReduceKernel(int* dev_array, int* dev_output, int array_size, int blockSize, int numBlocks) {

	reduceKernel <<<numBlocks, blockSize, blockSize * sizeof(int)>>> (dev_array, dev_output, array_size);
}

int main() {
	///-------------- Generate random array -------------///
	uint64_t state[2] = { 123456789, 987654321 };
	size_t array_size = 8192;
	vector<int> random_array = genereateRandomArray(state, array_size);
	//vector<int> random_array(array_size);
	//intialize_vector(random_array, 1);

	//	Print random array
	/*printf("Random Array: \n");
	for (const auto& num : random_array) {
		cout << num << "\n";
	}*/

	///-------------- Memory allocation -------------///
	int* dev_array;
	int* dev_result;
	size_t size = array_size * sizeof(int); // Size of array
	checkCudaError((cudaMalloc(&dev_array, size)), "cudaMalloc failed!");
	checkCudaError((cudaMalloc(&dev_result, sizeof(int) * ((array_size + 255) / 256))), "cudaMalloc failed!");

	///-------------- Copy data from host to device -------------///
	checkCudaError((cudaMemcpy(dev_array, random_array.data(), size, cudaMemcpyHostToDevice)), "cudaMemcpy failed!");

	///-------------- Launch kernel -------------///
	/// 
	///-------------- Fixed Block Size, Varying Grid Size -------------///
	cout << "Fixed Block Size, Varying Grid Size" << endl;
	cout << "---------------------------------------" << endl;
	int blockSize = 256; // Number of threads per block
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
	cout << "\nFixed Grid Size, Varying Block Size" << endl;
	cout << "---------------------------------------" << endl;
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