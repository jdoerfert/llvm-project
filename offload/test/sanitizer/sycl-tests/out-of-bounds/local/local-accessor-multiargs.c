// RUN: %libomptarget-compileopt-generic -fsanitize=offload
// RUN: not %libomptarget-run-generic 2>&1 > %t.out
// RUN: %fcheck-generic --check-prefixes=CHECK < %t.out

// Port of https://github.com/intel/llvm/blob/sycl/sycl/test-e2e/
// AddressSanitizer/out-of-bounds/local/local_accessor_multiargs.cpp

#include <omp.h>
#include <stdio.h>

#define ITEM_COUNT 3

int main() {
  int N_SZ = sizeof(int) * ITEM_COUNT;

  int Device = omp_get_default_device();
  int *DevPtr1 = (int *)omp_target_alloc(N_SZ, Device);
  int *DevPtr2 = (int *)omp_target_alloc(N_SZ, Device);

#pragma omp target is_device_ptr(DevPtr1, DevPtr2)
  {
    int T1[ITEM_COUNT] = {0};
    int T2[ITEM_COUNT] = {0};
    int T3[ITEM_COUNT] = {0};
    for (int i = 0; i < ITEM_COUNT; i++) {
      DevPtr1[i] = T1[i] + T2[i] + T3[i];
      DevPtr2[i] = T1[i] + T2[i + 1] + T3[i];
    }
  }

  omp_target_free(DevPtr1, Device);
  omp_target_free(DevPtr2, Device);

  return 0;
}
