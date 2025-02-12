
#include <stdio.h>

void foo(int *q) {
  int V = q[0];
  if (V > 0) {
    if (V < 100) {
      int P = q[1];
      if (V == P)
        printf("Hit 1a: %i, %i\n", V, P);
      else
        printf("Hit 1b: %i, %i\n", V, P);
    } else {
      int P = q[1];
      if (V * 2 == P)
        printf("Hit 2a: %i, %i\n", V, P);
      else
        printf("Hit 2b: %i, %i\n", V, P);
    }
  } else {
    printf("Hit 3: %i\n", V);
  }
}
