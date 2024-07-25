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

int global1 = 0;
int global2 = -1;
#pragma omp declare target(global1, global2)

int main() {
#pragma omp target
  {
    global1 = 72;
    global2 = global1 + 75;
  }
  return 0;
}
