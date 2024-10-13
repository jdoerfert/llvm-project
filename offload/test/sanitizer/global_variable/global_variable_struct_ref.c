// RUN: %libomptarget-compileopt-generic \
// RUN:   -fopenmp-offload-mandatory
// RUN: %libomptarget-run-generic 1 1 > %t.ok.out
// RUN: %fcheck-generic --check-prefixes=OK < %t.ok.out
// RUN: not %libomptarget-run-generic 1 4 2> %t.err.out
// RUN: %fcheck-generic --check-prefixes=ERR < %t.err.out

// REQUIRES: gpu

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

struct InnerData {
  float field1;
  int field2[3];
};

struct Data {
  int field1;
  struct InnerData field2;
};

struct Data global_arr[2] = {
    {1, {2.2, {1, 2, 3}}},
    {9, {4.3, {5, 6, 7}}},
};
#pragma omp declare target(global_arr)

int main(int argc, char **argv) {
  int first = atoi(argv[1]);
  int second = atoi(argv[2]);

  // OK: Accessing global_arr[1].field2.field2[1] on device
  printf("Accessing global_arr[%d].field2.field2[%d] on device\n", first,
         second);

#pragma omp target map(to : first, second)
  {
    // ERR: 44 bytes inside of a 40-byte
    global_arr[first].field2.field2[second] = 2302323;
  }
  return 0;
}
