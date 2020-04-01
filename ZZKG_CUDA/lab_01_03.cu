#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <math.h>
#include <iostream>

const int N = 512;
const int blocksize = 32;

__global__ void solve_stm(unsigned int *a, unsigned int n) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int index = i + j*n;
	if (i < n && j < n) {
		a[index] = i * j;
	}
}


int main() {
	thrust::host_vector<unsigned int> a_tab;
	thrust::device_vector<unsigned int> ad_tab;

	for (int i = 1; i <= N*N; i++) {
		a_tab.push_back(0);
	}

	ad_tab = a_tab;

	dim3 dimBlock(blocksize, blocksize);
	unsigned int g = ceil((float)N / (float)blocksize);
	dim3 dimGrid(g,g);

	solve_stm << < dimGrid, dimBlock >> >(ad_tab.data().get(), N);

	a_tab = ad_tab;

	for (int i = 0; i < (int)(N*N*0.01); i++) {
		std::cout << i << " => " << i / N << ":" << i % N << " = " << a_tab[i] << "\n";
	}

	return 0;
}