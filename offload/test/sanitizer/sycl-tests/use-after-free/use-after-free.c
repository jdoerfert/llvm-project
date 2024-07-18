// RUN: %libomptarget-compileopt-generic -fsanitize=offload
// RUN: not %libomptarget-run-generic 2>&1 > %t.out
// RUN: %fcheck-generic --check-prefixes=CHECK < %t.out

#include <omp.h>
#include <stdio.h>

int main() {
  int N = 100;
  int N_SZ = sizeof(int) * N;

  int Device = omp_get_default_device();

  int *DevPtr = (int *)omp_target_alloc(N_SZ, Device);
  omp_target_free(DevPtr, Device);

#pragma omp target is_device_ptr(DevPtr)
  for (int i = 0; i < N; i++) {
    DevPtr[i] = i;
  }

  return 0;
}
