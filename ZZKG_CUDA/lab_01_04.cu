#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <math.h>
#include <iostream>

const int N = 10000;
const int M = 100;
const int blocksize = 1024;

__global__ void shift_r(int *a, float *b, unsigned int n) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n-2) {
		float delta = (float)(a[i + 1] * a[i + 1]) - (float)(4 * a[i] * a[i + 2]);
		float no_solve = 0;
		float x1 = 0.0;
		float x2 = 0.0;
		if (delta == 0) {
			no_solve = 1;
			x1 = -((float)a[i + 1] / (float)(2 * a[i]));
		}
		else if (delta > 0) {
			no_solve = 2;
			x1 = -(((float)a[i + 1] - (float)sqrt((float)delta)) / (float)(2 * a[i]));
			x2 = -(((float)a[i + 1] + (float)sqrt((float)delta)) / (float)(2 * a[i]));
		}
		float q = -((float)(delta) / (float)(4 * a[i]));

		b[4 * i + 0] = no_solve;
		b[4 * i + 1] = x1;
		b[4 * i + 2] = x2;
		b[4 * i + 3] = q;
	}
}


int main() {
	thrust::host_vector<int> a_tab;
	thrust::device_vector<int> ad_tab;
	thrust::host_vector<float> b_tab;
	thrust::device_vector<float> bd_tab;

	for (int i = 1; i <= N; i++) {
		a_tab.push_back((int)(rand() % M));
	}
	for (int i = 1; i <= 4*(N-2); i++) {
		b_tab.push_back(0);
	}

	ad_tab = a_tab;
	bd_tab = b_tab;

	dim3 dimBlock(blocksize);
	unsigned int g = ceil((float)N / (float)blocksize);
	dim3 dimGrid(g);

	shift_r << < dimGrid, dimBlock >> >(ad_tab.data().get(), bd_tab.data().get(), N);

	a_tab = ad_tab;
	b_tab = bd_tab;

	for (int i = 0; i < (int)(N-2); i++) {
		std::cout << i << " | ";
		std::cout << a_tab[i] << ", " << a_tab[i+1] << ", " << a_tab[i+2] << " => ";
		std::cout << b_tab[4 * i] << ", " << b_tab[4 * i + 1] << ", " << b_tab[4 * i + 2] << ", " << b_tab[4 * i + 3] << "\n";
	}

	return 0;
}