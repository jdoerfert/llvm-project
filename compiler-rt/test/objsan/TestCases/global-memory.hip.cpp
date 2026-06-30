// RUN: %clang_objsan_hip_compile -O2 -DCONF1 -c %s -o %t.o
// RUN: %clang_objsan_hip_link %t.o %clang_objsan_hip_post_link -o %t.a.out
// RUN: not %t.a.out 2>&1 | FileCheck %s --check-prefix=CONF1
// CONF1: s bad

// RUN: %clang_objsan_hip_compile -O2 -DCONF2 -c %s -o %t.o
// RUN: %clang_objsan_hip_link %t.o %clang_objsan_hip_post_link -o %t.a.out
// RUN: %t.a.out 2>&1 | FileCheck %s --check-prefix=CONF2
// CONF2: Execution completed successfully

// RUN: %clang_objsan_hip_compile -O2 -DCONF3 -c %s -o %t.o
// RUN: %clang_objsan_hip_link %t.o %clang_objsan_hip_post_link -o %t.a.out
// RUN: not %t.a.out 2>&1 | FileCheck %s --check-prefix=CONF1
// CONF3: s bad

// RUN: %clang_objsan_hip_compile -O2 -DCONF4 -c %s -o %t.o
// RUN: %clang_objsan_hip_link %t.o %clang_objsan_hip_post_link -o %t.a.out
// RUN: not %t.a.out 2>&1 | FileCheck %s --check-prefix=CONF4
// CONF4: l bad

// RUN: %clang_objsan_hip_compile -O2 -DCONF5 -c %s -o %t.o
// RUN: %clang_objsan_hip_link %t.o %clang_objsan_hip_post_link -o %t.a.out
// RUN: %t.a.out 2>&1 | FileCheck %s --check-prefix=CONF5
// CONF5: Execution completed successfully

// RUN: %clang_objsan_hip_compile -O2 -DCONF6 -c %s -o %t.o
// RUN: %clang_objsan_hip_link %t.o %clang_objsan_hip_post_link -o %t.a.out
// RUN: not %t.a.out 2>&1 | FileCheck %s --check-prefix=CONF6
// CONF6: l bad

// RUN: %clang_objsan_hip_compile -O2 -DCONF7 -c %s -o %t.o
// RUN: %clang_objsan_hip_link %t.o %clang_objsan_hip_post_link -o %t.a.out
// RUN: not %t.a.out 2>&1 | FileCheck %s --check-prefix=CONF7
// CONF7: s bad

// RUN: %clang_objsan_hip_compile -O2 -DCONF8 -c %s -o %t.o
// RUN: %clang_objsan_hip_link %t.o %clang_objsan_hip_post_link -o %t.a.out
// RUN: %t.a.out 2>&1 | FileCheck %s --check-prefix=CONF8
// CONF8: Execution completed successfully

#include "common.h"
#include "common.hip.h"

static __device__ int global[10];
static __device__ int array_index[2] = { 0, 0 };

__attribute__((noinline)) __device__ void get(int *array) {
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
#ifdef CONF7
  array[array_index[0]] = array[9];
#endif
#ifdef CONF8
  array[array_index[1]] = array[9];
#endif
}

__global__ void access_kernel(int *array, int size) {
  get(global);
  __syncthreads();
  array[0] = global[0];
}

__global__ void index_kernel(int index, int value) {
  array_index[index] = value;
}

OBJSAN_TEST_MAIN(run_test)

int run_test(int argc, char **argv) {
  const int size = 10;
  int *d_array;

  HIP_CHECK(hipMalloc((void **)&d_array, size * sizeof(int)));

  index_kernel<<<1, 1>>>(0, size);
  index_kernel<<<1, 1>>>(1, size-1);

  access_kernel<<<1, 1>>>(d_array, size);
  HIP_CHECK(hipPeekAtLastError());
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipFree(d_array));

  fprintf(stdout, "%s", "Execution completed successfully\n");
  return 0;
}
