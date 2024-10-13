// RUN: %libomptarget-compileopt-generic \
// RUN:   -fopenmp-offload-mandatory
// RUN: %libomptarget-run-generic 1 1 > %t.ok.out
// RUN: %fcheck-generic --check-prefixes=OK < %t.ok.out
// RUN: not %libomptarget-run-generic 4 3 2> %t.err.out
// RUN: %fcheck-generic --check-prefixes=ERR < %t.err.out

// REQUIRES: gpu

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int global_arr[2][2] = {{1, 2}, {3, 4}};
#pragma omp declare target(global_arr)

int main(int argc, char **argv) {
  int first = atoi(argv[1]);
  int second = atoi(argv[2]);

  // OK: Accessing global_arr[1][1] on device
  printf("Accessing global_arr[%d][%d] on device\n", first, second);

#pragma omp target map(to : first, second)
  {
    // ERR: 44 bytes inside of a 16-byte
    global_arr[first][second] = 2302323;
  }
  return 0;
}
