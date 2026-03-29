// RUN: %clang_cuda_compile -c %s -o %t.o
// RUN: %clang_cuda_link %t.o %clang_cuda_post_link -o %t.a.out
// RUN: %t.a.out 2>&1 | FileCheck %s

// Checks that we do not break normal cuda compilation
// CHECK: Execution completed successfully

#include "common.cuda.h"

__device__ void func(int *array) {
  array[9] = 0;
}

__global__ void kernel(int *array) { func(array); }

int main(int argc, char **argv) {
  const int size = 10;
  int *d_array;

  CUDA_CHECK(cudaMalloc((void **)&d_array, size * sizeof(int)));

  kernel<<<1, 1>>>(d_array);
  CUDA_CHECK(cudaPeekAtLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaFree(d_array));

  fprintf(stdout, "%s", "Execution completed successfully\n");
}
