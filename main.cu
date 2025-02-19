#include <iostream>
#include <vector>
#include <cstdint>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

uint64_t xorshift128plus(uint64_t s[2]) {
	uint64_t x = s[0];
	uint64_t y = s[1];
	s[0] = y;
	x ^= x << 23;
	s[1] = x ^ y ^ (x >> 17) ^ (y >> 26);
	return s[1] + y;
}

vector<uint64_t> genereateRandomArray(uint64_t s[2], size_t size) {
	vector<uint64_t> random_array(size);

	for (size_t i = 0; i < size; i++) {
		random_array[i] = xorshift128plus(s);
	}

	return random_array;
}

int main() {
	///-------------- Generate random array -------------///
	// Initialize random number generator
	uint64_t state[2] = { 123456789, 987654321 };
	// Set array size
	size_t array_size = 1024;
	// Generate random array
	vector<uint64_t> random_array = genereateRandomArray(state, array_size);
	//	Print random array
	printf("Random Array: \n");
	for (const auto& num : random_array) {
		cout << num << "\n";
	}


	///-------------- Memory allocation -------------///
	float* dev_array;
	size_t size = array_size * sizeof(float);
	cudaError_t err = cudaMalloc(&dev_array, size);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return 1;
	}

	///-------------- Copy data from host to device -------------///
	err = cudaMemcpy(dev_array, random_array.data(), size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return 1;
	}

	///-------------- CUDA Kernel -------------///

	return 0;
}