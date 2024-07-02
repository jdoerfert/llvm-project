// RUN: %libomptarget-compileopt-generic -fsanitize=offload
// RUN: %libomptarget-run-generic 2>&1 > %t.out
// RUN: %fcheck-generic --check-prefixes=CHECK < %t.out

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define X 3

int main() {
  int A[X] = {1, 2, 3};

#pragma omp target map(tofrom : A)
  {
    for (int i = 0; i < X; i++) {
      A[i] = A[i] * 2;
    }
  }

  for (int i = 0; i < X; i++) {
    printf("%d\n", A[i]);
  }

  return 0;
}

// CHECK: 2
// CHECK: 4
// CHECK: 6
