// RUN: %libomptarget-compileopt-generic -fsanitize=offload
// RUN: not %libomptarget-run-generic 2>&1 > %t.out
// RUN: %fcheck-generic --check-prefixes=CHECK < %t.out

// Port of https://github.com/intel/llvm/blob/sycl/sycl/test-e2e/
// AddressSanitizer/out-of-bounds/local/local_accessor_basic.cpp

#include <omp.h>
#include <stdio.h>

#define ITEM_COUNT 3

int main() {
  int N_SZ = sizeof(int) * ITEM_COUNT;

  int Device = omp_get_default_device();
  int *DevPtr = (int *)omp_target_alloc(N_SZ, Device);

#pragma omp target is_device_ptr(DevPtr1)
  {
    int T1[ITEM_COUNT] = {0};
    for (int i = 0; i < ITEM_COUNT; i++) {
      DevPtr[i] = T1[i + 1];
    }
  }

  omp_target_free(DevPtr, Device);

  return 0;
}
