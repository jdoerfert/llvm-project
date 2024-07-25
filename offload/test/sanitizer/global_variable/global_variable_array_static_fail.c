// RUN: %libomptarget-compileopt-generic -g -fsanitize=offload
// RUN: not %libomptarget-run-generic 2> %t.out
// RUN: %fcheck-generic --check-prefixes=CHECK < %t.out

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
    // CHECK: is located 20 bytes inside of a 12-byte region
    global_arr[5] = 27;
  }
  return 0;
}
