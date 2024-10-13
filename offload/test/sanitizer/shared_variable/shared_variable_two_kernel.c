// RUN: %libomptarget-compileopt-generic \
// RUN:   -fopenmp-offload-mandatory
// RUN: %libomptarget-run-generic

// REQUIRES: gpu

#include <omp.h>

#define __SHARED__ __attribute__((address_space(3)))

__SHARED__ int shared_test [[clang::loader_uninitialized]];
#pragma omp declare target to(shared_test) device_type(nohost)

int main() {
#pragma omp target
  { shared_test = 192303; }
#pragma omp target
  { shared_test = 602934; }
  return 0;
}
