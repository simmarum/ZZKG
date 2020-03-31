#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <math.h>
#include <iostream>

const int N = 1024;
const int blocksize = 16;

__global__ void add_matrix(float *a, float *b, float *c, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int index = i + j*N;
	if (i < N && j < N) {
		c[index] = a[index] + b[index];
	}
}

int main() {
	float *a = new float[N*N];
	float *b = new float[N*N];
	float *c = new float[N*N];
	for (int i = 0; i < N*N; ++i) {
		a[i] = 1.0f;
		b[i] = 3.5f; 
	}
	float *ad, *bd, *cd;
	const int size = N*N * sizeof(float);

	cudaMalloc((void**)&ad, size);
	cudaMalloc((void**)&bd, size);
	cudaMalloc((void**)&cd, size);

	cudaMemcpy(ad, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(bd, b, size, cudaMemcpyHostToDevice);

	dim3 dimBlock(blocksize, blocksize);
	dim3 dimGrid(ceil((float)N / (float)blocksize), ceil((float)N / blocksize));

	add_matrix <<< dimGrid, dimBlock >>>(ad, bd, cd, N);

	cudaMemcpy(c, cd, size, cudaMemcpyDeviceToHost);

	for (int i = 0; i<10; i++) {
		std::cout << c[i] << "\n";
	}

	cudaFree(ad); 
	cudaFree(bd); 
	cudaFree(cd);
	delete[] a; 
	delete[] b;
	delete[] c;

	return 0;
}