
#include <stdio.h>

__attribute__((inputgen_entry))
void foo(int *q) {
  int V = q[0];
  int P = q[1];
  if (V == P) {
    printf("Hit 1: %i : %i\n", V, P);
  } else {
    printf("Hit 2: %i : %i\n", V, P);
  }
  printf("Hit 3: %i : %i\n", V, P);
}
