// RUN: %libomptarget-compileopt-generic -fsanitize=offload
// RUN: not %libomptarget-run-generic 2>&1 > %t.out
// RUN: %fcheck-generic --check-prefixes=CHECK < %t.out

#include <omp.h>
#include <stdio.h>

int main() {
  int FakePtr[3] = {1, 2, 3};
  int Device = omp_get_default_device();
  omp_target_free(FakePtr, Device);
  return 0;
}
