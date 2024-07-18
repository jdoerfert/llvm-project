// RUN: %libomptarget-compileopt-generic -fsanitize=offload
// RUN: not %libomptarget-run-generic 2>&1 > %t.out
// RUN: %fcheck-generic --check-prefixes=CHECK < %t.out

// Port of https://github.com/intel/llvm/blob/sycl/sycl/test-e2e/
// AddressSanitizer/out-of-bounds/DeviceGlobal/device_global.cpp

#include <omp.h>
#include <stdio.h>

#define ITEM_COUNT 3

char dev_global[5];
#pragma omp declare target(dev_global)

int main() {
#pragma omp target
  { dev_global[8] = 42; }
  return 0;
}
