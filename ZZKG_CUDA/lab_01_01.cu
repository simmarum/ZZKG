#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <math.h>
#include <iostream>

const int N = 1000000;
const int blocksize = 256;

__global__ void add_two_tab(unsigned int *a, unsigned int *b, unsigned int *c, unsigned int n) {
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid<n) {
		c[tid] = a[tid] + b[tid];
	}
}


int main() {
	thrust::host_vector<unsigned int> a_tab;
	thrust::device_vector<unsigned int> ad_tab;
	thrust::host_vector<unsigned int> b_tab;
	thrust::device_vector<unsigned int> bd_tab;
	thrust::host_vector<unsigned int> c_tab;
	thrust::device_vector<unsigned int> cd_tab;

	for (int i = 1; i <= N; i++) {
		a_tab.push_back(1);
		b_tab.push_back(10);
		c_tab.push_back(0);
	}

	ad_tab = a_tab;
	bd_tab = b_tab;
	cd_tab = c_tab;

	dim3 dimBlock(blocksize);
	dim3 dimGrid(ceil((float)N / (float)blocksize));
	
	add_two_tab <<< dimGrid,dimBlock >>>(ad_tab.data().get(), bd_tab.data().get(), cd_tab.data().get(), ad_tab.size());
	
	c_tab = cd_tab;

	for (int i = 0; i < 10; i++) {
		std::cout << i << " : " << c_tab[i] << "\n";
		std::cout << N-1-i << " : " << c_tab[N-1-i] << "\n";
	}

	return 0;
}