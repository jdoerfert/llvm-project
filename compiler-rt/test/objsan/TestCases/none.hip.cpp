// RUN: %clang_hip_compile -c %s -o %t.o
// RUN: %clang_hip_link %t.o %clang_hip_post_link -o %t.a.out
// RUN: %t.a.out 2>&1 | FileCheck %s

// Checks that we do not break normal hip compilation
// CHECK: Execution completed successfully

#include "common.hip.h"

__device__ void func(int *array) {
  array[9] = 0;
}

__global__ void kernel(int *array) { func(array); }

int main(int argc, char **argv) {
  const int size = 10;
  int *d_array;

  HIP_CHECK(hipMalloc((void **)&d_array, size * sizeof(int)));

  kernel<<<1, 1>>>(d_array);
  HIP_CHECK(hipPeekAtLastError());
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipFree(d_array));

  fprintf(stdout, "%s", "Execution completed successfully\n");
}
