// RUN: %libomptarget-compileopt-generic \
// RUN:   -fopenmp-offload-mandatory
// RUN: not %libomptarget-run-generic 2> %t.out
// RUN: %fcheck-generic --check-prefixes=CHECK < %t.out

// REQUIRES: gpu

#include <omp.h>
#include <stdio.h>

#define __SHARED__ __attribute__((address_space(3)))

__SHARED__ int shared_arr [[clang::loader_uninitialized]][3];
#pragma omp declare target to(shared_arr) device_type(nohost)

int main() {
#pragma omp target
  {
    // CHECK: 20 bytes inside of a 12-byte
    shared_arr[5] = 27;
  }
  return 0;
}
