#include <cuda_runtime.h>
#include <cstdio>

#include "objsan_interface_internal.h"

#define CUDA_CHECK(Call)                                                        \
    do {                                                                       \
        cudaError_t Error = (Call);                                             \
        if (Error != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Error at %s:%d: %s (%d)\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(Error), Error);      \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

__device__ int array_index;

__device__ void func(int *array, int size, int n) {
  array[n] = 200; // NOTE: Should trigger an error
}

__global__ void access_kernel(int *array, int size, int n) {
  func(array, size, array_index);
}

__global__ void index_kernel(int n) {
  array_index = n;
}

int main(int argc, char **argv) {
  __objsan_rt_init();

  const int size = 10;
  const int n = (argc > 1) ? std::atoi(argv[1]) : 0;

  int *d_array;
  CUDA_CHECK(cudaMalloc((void **)&d_array, size * sizeof(int)));

  index_kernel<<<1, 1>>>(n);
  CUDA_CHECK(cudaPeekAtLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  access_kernel<<<1, 1>>>(d_array, size, n);
  CUDA_CHECK(cudaPeekAtLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaFree(d_array));

  fprintf(stdout, "Execution completed successfully\n");

  __objsan_rt_deinit();
}
