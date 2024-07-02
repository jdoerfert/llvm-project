// RUN: %libomptarget-compileopt-generic -fsanitize=offload
// RUN: not %libomptarget-run-generic 2>&1 > %t.out
// RUN: %fcheck-generic --check-prefixes=CHECK < %t.out

// Port of
// https://github.com/intel/llvm/blob/sycl/sycl/test-e2e/AddressSanitizer
// /use-after-free/quarantine-no-free.cpp

#include <omp.h>
#include <stdio.h>

int main() {
  int N = 100;
  int N_SZ = sizeof(int) * N;

  int Device = omp_get_default_device();

  int *DevPtr = (int *)omp_target_alloc(N_SZ, Device);
  omp_target_free(DevPtr, Device);

  DevPtr = (int *)omp_target_alloc(N_SZ, Device);
  omp_target_free(DevPtr, Device);

  DevPtr = (int *)omp_target_alloc(N_SZ, Device);
  omp_target_free(DevPtr, Device);

#pragma omp target is_device_ptr(DevPtr)
  { DevPtr[0] = 0; }

  return 0;
}
