
#include <stdio.h>

void foo(int *q) {
  int V = q[0];
  if (V > 0) {
    printf("1: true\n");
    if (V < 100) {
      printf("2: true\n");
      int P = q[1];
      if (V == P) {
        printf("Hit 1: %i, %i\n", V, P);
      }
    } else {
      printf("2: false\n");
      int P = q[1];
      if (V * 2 == P) {
        printf("Hit 2: %i, %i\n", V, P);
      }
    }
  } else {
    printf("1: false\n");
  }
}
