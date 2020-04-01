#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <math.h>
#include <iostream>

__global__ void multiplyBy2(unsigned int *data, unsigned int n) {
	unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid<n) {
		data[tid] = 2 * data[tid];
	}
}


int main() {
	thrust::host_vector<unsigned int> h_tab;
	thrust::device_vector<unsigned int> d_tab;

	for (int i = 1; i <= 10; i++) {
		h_tab.push_back(i);//dane wejsciowe
	}
	
	d_tab = h_tab; //Kopiowanie host->device
	
	multiplyBy2 << <1, 10 >> >(d_tab.data().get(), d_tab.size());
	
	h_tab = d_tab; //Kopiowanie device->host
	
	for (int i = 0; i < 10; i++) {
		std::cout << h_tab[i] << "\n";
	}

	return 0;
}