// RUN: %libomptarget-compileopt-generic -fsanitize=offload
// RUN: not %libomptarget-run-generic 2>&1 > %t.out
// RUN: %fcheck-generic --check-prefixes=CHECK < %t.out

#include <omp.h>
#include <stdio.h>

__attribute__((noinline)) void foo(int *array, int i) { array[i] = 1; }

int main() {
  int N = 10;
  int N_SZ = sizeof(int) * N;

  int Device = omp_get_default_device();

  int *DevPtr = (int *)omp_target_alloc(N_SZ, Device);

#pragma omp target is_device_ptr(DevPtr)
  {
#pragma omp parallel for
    for (int i = 0; i < N; i++) {
      foo(DevPtr, i);
    }
  }

  omp_target_free(DevPtr, Device);

  return 0;
}
