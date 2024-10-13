// RUN: %libomptarget-compileopt-generic -g \
// RUN:   -fopenmp-offload-mandatory
// RUN: %libomptarget-run-generic

// REQUIRES: gpu

#include <omp.h>

#define __SHARED__ __attribute__((address_space(3)))

__SHARED__ int shared_arr [[clang::loader_uninitialized]][3];
#pragma omp declare target to(shared_arr) device_type(nohost)

int main() {
#pragma omp target
  {
    for (int i = 0; i < 3; i++) {
      shared_arr[i] = i * 4;
    }
    int v = 1;
    shared_arr[v] = 203202;
  }
  return 0;
}
