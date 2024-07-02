// RUN: %libomptarget-compileopt-generic -fsanitize=offload
// RUN: %libomptarget-run-generic 2>&1 > %t.out
// RUN: %fcheck-generic --check-prefixes=CHECK < %t.out

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define X 4
#define Y 3
#define Z 2

int main() {
  int A[X][Y][Z] = {
      {{0, 1}, {2, 3}, {4, 5}},
      {{6, 7}, {8, 9}, {10, 11}},
      {{12, 13}, {14, 15}, {16, 17}},
      {{18, 19}, {20, 21}, {22, 23}},
  };

#pragma omp target map(tofrom : A)
  {
    for (int i = 0; i < X; i++) {
      for (int j = 0; j < Y; j++) {
        for (int k = 0; k < Z; k++) {
          A[i][j][k] = A[i][j][k] * 2;
        }
      }
    }
  }

  for (int i = 0; i < X; i++) {
    for (int j = 0; j < Y; j++) {
      printf("(");
      for (int k = 0; k < Z; k++) {
        printf("%d", A[i][j][k]);
        if (k + 1 != Z) {
          printf(",");
        }
      }
      printf(")");
      if (j + 1 != Y) {
        printf(",");
      }
    }
    printf("\n");
  }

  return 0;
}

// CHECK: (0,2),(4,6),(8,10)
// CHECK: (12,14),(16,18),(20,22)
// CHECK: (24,26),(28,30),(32,34)
// CHECK: (36,38),(40,42),(44,46)
