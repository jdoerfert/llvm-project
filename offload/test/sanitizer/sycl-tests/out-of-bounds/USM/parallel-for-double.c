// RUN: %libomptarget-compileopt-generic -fsanitize=offload
// RUN: not %libomptarget-run-generic 2>&1 > %t.out
// RUN: %fcheck-generic --check-prefixes=CHECK < %t.out

#include <omp.h>
#include <stdio.h>

int main() {
  int N = 10;
  int N_SZ = sizeof(double) * N;

  int Device = omp_get_default_device();

  double *DevPtr = (double *)omp_target_alloc(N_SZ, Device);

#pragma omp target is_device_ptr(DevPtr)
  {
#pragma omp parallel for
    for (int i = 0; i < N; i++) {
      DevPtr[i] = 1.23;
    }
  }

  omp_target_free(DevPtr, Device);

  return 0;
}
