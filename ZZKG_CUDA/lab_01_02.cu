#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <math.h>
#include <iostream>

const int N = 200;
const int M = 50;
const int blocksize = 1024;

__global__ void set_row_in_tab(unsigned int *a, unsigned int n, unsigned int m) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n*m) {
		a[i] = i/m;
	}
}


int main() {
	thrust::host_vector<unsigned int> a_tab;
	thrust::device_vector<unsigned int> ad_tab;

	for (int i = 1; i <= N*M; i++) {
		a_tab.push_back(0);
	}

	ad_tab = a_tab;

	dim3 dimBlock(blocksize);
	unsigned int g = ceil((float)N*M / (float)blocksize);
	dim3 dimGrid(g);

	set_row_in_tab << < dimGrid, dimBlock >> >(ad_tab.data().get(), N, M);

	a_tab = ad_tab;

	for (int i = 0; i < M*N; i++) {
		std::cout << i << " : " << a_tab[i] << "\n";
		//std::cout << N - 1 - i << " : " << a_tab[N - 1 - i] << "\n";
	}

	return 0;
}