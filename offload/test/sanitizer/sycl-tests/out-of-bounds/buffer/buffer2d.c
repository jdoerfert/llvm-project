// RUN: %libomptarget-compileopt-generic -fsanitize=offload
// RUN: %libomptarget-run-generic 2>&1 > %t.out
// RUN: %fcheck-generic --check-prefixes=CHECK < %t.out

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define X 2
#define Y 3

int main() {
  int A[X][Y] = {{1, 2, 3}, {4, 5, 6}};

#pragma omp target map(tofrom : A)
  {
    for (int i = 0; i < X; i++) {
      for (int j = 0; j < Y; j++) {
        A[i][j] = A[i][j] * 2;
      }
    }
  }

  for (int i = 0; i < X; i++) {
    for (int j = 0; j < Y; j++) {
      printf("%d", A[i][j]);
      if (j + 1 != Y) {
        printf(" ");
      }
    }
    printf("\n");
  }

  return 0;
}

// CHECK: 2 4 6
// CHECK-NEXT: 8 10 12
