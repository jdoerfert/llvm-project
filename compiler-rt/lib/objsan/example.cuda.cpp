#include <cuda_runtime.h>
#include <cstdio>

__device__ void func(int *array, int size) {
	array[1000] = 200;
}

__global__ void kernel(int *array, int size) {
	//printf("kernel: array[%d] %d\n", 0, array[0]);
	//printf("kernel: array %p %d\n", array, size);
	func(array, size);
	//printf("kernel: array %p size %d\n", array, size);
	//printf("kernel: array[%d] %d\n", 0, array[0]);
}

int main(int argc, char **argv) {
	const int size = 10;
	int *d_array;

	cudaError_t Err = cudaMalloc((void**)&d_array, size * sizeof(int));
	if (Err != cudaSuccess)
		fprintf(stderr, "error in cudaMalloc: %s\n", cudaGetErrorString(Err));

	kernel<<<1, 1>>>(d_array, size);

	Err = cudaFree(d_array);
	if (Err != cudaSuccess)
		fprintf(stderr, "error in cudaFree: %s\n", cudaGetErrorString(Err));
}
