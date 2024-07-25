// RUN: %libomptarget-compileopt-generic -g -fsanitize=offload
// RUN: %libomptarget-run-generic

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: x86_64-pc-linux-gnu
// UNSUPPORTED: x86_64-pc-linux-gnu-LTO
// UNSUPPORTED: s390x-ibm-linux-gnu
// UNSUPPORTED: s390x-ibm-linux-gnu-LTO

#include <omp.h>
#include <stdio.h>

int global_arr[3] = {1, 2, 3};
#pragma omp declare target(global_arr)

int main() {
#pragma omp target
  {
    for (int i = 0; i < 3; i++) {
      global_arr[i] *= 4;
    }
    global_arr[1] = 22;
  }
  return 0;
}
