// RUN: %clang_objsan_cuda_compile -DCONF1 -c %s -o %t.o
// RUN: %clang_objsan_cuda_link %t.o %clang_objsan_cuda_post_link -o %t.a.out
// RUN: not %t.a.out 2>&1 | FileCheck %s --check-prefix=CONF1
// CONF1: s bad

// RUN: %clang_objsan_cuda_compile -DCONF2 -c %s -o %t.o
// RUN: %clang_objsan_cuda_link %t.o %clang_objsan_cuda_post_link -o %t.a.out
// RUN: %t.a.out 2>&1 | FileCheck %s --check-prefix=CONF2
// CONF2: Execution completed successfully

// RUN: %clang_objsan_cuda_compile -DCONF3 -c %s -o %t.o
// RUN: %clang_objsan_cuda_link %t.o %clang_objsan_cuda_post_link -o %t.a.out
// RUN: not %t.a.out 2>&1 | FileCheck %s --check-prefix=CONF1
// CONF3: s bad

// RUN: %clang_objsan_cuda_compile -DCONF4 -c %s -o %t.o
// RUN: %clang_objsan_cuda_link %t.o %clang_objsan_cuda_post_link -o %t.a.out
// RUN: not %t.a.out 2>&1 | FileCheck %s --check-prefix=CONF4
// CONF4: l bad

// RUN: %clang_objsan_cuda_compile -DCONF5 -c %s -o %t.o
// RUN: %clang_objsan_cuda_link %t.o %clang_objsan_cuda_post_link -o %t.a.out
// RUN: %t.a.out 2>&1 | FileCheck %s --check-prefix=CONF5
// CONF5: Execution completed successfully

// RUN: %clang_objsan_cuda_compile -DCONF6 -c %s -o %t.o
// RUN: %clang_objsan_cuda_link %t.o %clang_objsan_cuda_post_link -o %t.a.out
// RUN: not %t.a.out 2>&1 | FileCheck %s --check-prefix=CONF6
// CONF6: l bad

#include "common.cuda.h"

__device__ void func(int *array) {
#ifdef CONF1
  array[10] = 0;
#endif
#ifdef CONF2
  array[9] = 0;
#endif
#ifdef CONF3
  array[-1] = 0;
#endif
#ifdef CONF4
  array[0] = array[10];
#endif
#ifdef CONF5
  array[0] = array[9];
#endif
#ifdef CONF6
  array[0] = array[-1];
#endif
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
