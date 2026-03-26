#include <cuda_runtime.h>
#include <cstdio>

__device__ void func(int *array, int size) {
	array[10] = 200; // NOTE: Should trigger an error
	//array[3999] = 200;
}

__device__ void func2(int *array, int size, int *val) {
  *val = array[3999];
}

__global__ void kernel(int *array, int size, int n) {
  //int tmp[1000];
	//printf("kernel: array[%d] %d\n", 0, array[0]);
	func(array, size);
  //func2(array, size, &tmp[n]);
	//printf("kernel: array %p size %d\n", array, size);
	//printf("kernel: array[%d] %d\n", 0, array[0]);
  //array[0] = tmp[0];
}

int main(int argc, char **argv) {
	const int size = 10;
  const int n = (argc == 1) ? 0 : 999;
	int *d_array;

	cudaError_t Err = cudaMalloc((void**)&d_array, size * sizeof(int));
	if (Err != cudaSuccess)
		fprintf(stderr, "error in cudaMalloc: %s\n", cudaGetErrorString(Err));

	kernel<<<1, 1>>>(d_array, size, n);

	Err = cudaFree(d_array);
	if (Err != cudaSuccess)
		fprintf(stderr, "error in cudaFree: %s\n", cudaGetErrorString(Err));
}
