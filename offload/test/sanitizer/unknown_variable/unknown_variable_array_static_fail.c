// RUN: %libomptarget-compileopt-generic -g -fsanitize=offload \
// RUN:   -fopenmp-offload-mandatory
// RUN: not %libomptarget-run-generic 2> %t.out
// RUN: %fcheck-generic --check-prefixes=CHECK < %t.out

// REQUIRES: gpu

#include <omp.h>

#define __SHARED__ __attribute__((address_space(3)))
#define EL_TYPE int

__SHARED__ EL_TYPE shared_arr [[clang::loader_uninitialized]][3];
#pragma omp declare target to(shared_arr) device_type(nohost)

int main() {
#pragma omp target
  {
    EL_TYPE *unknown_arr = (EL_TYPE *)shared_arr + 2;
    // CHECK: 16 bytes inside of a 12-byte region
    unknown_arr[2] = 5238734;
  }
  return 0;
}
