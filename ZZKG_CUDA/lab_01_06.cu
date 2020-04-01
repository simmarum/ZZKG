#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <math.h>
#include <iostream>

const int N = 10000;
const int blocksize = 1024;

__global__ void shift_r(unsigned int *a, unsigned int *b, unsigned int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		b[i] = a[n-1-i];
	}
}


int main() {
	thrust::host_vector<unsigned int> a_tab;
	thrust::device_vector<unsigned int> ad_tab;
	thrust::host_vector<unsigned int> b_tab;
	thrust::device_vector<unsigned int> bd_tab;

	for (int i = 1; i <= N; i++) {
		a_tab.push_back(i);
		b_tab.push_back(0);
	}

	ad_tab = a_tab;
	bd_tab = b_tab;

	dim3 dimBlock(blocksize);
	unsigned int g = ceil((float)N / (float)blocksize);
	dim3 dimGrid(g);

	shift_r << < dimGrid, dimBlock >> >(ad_tab.data().get(), bd_tab.data().get(), N);

	b_tab = bd_tab;

	for (int i = 0; i < N; i++) {
		std::cout << a_tab[i] << " : " << b_tab[i] << "\n";
	}

	return 0;
}