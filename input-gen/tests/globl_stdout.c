#include <stdio.h>

FILE *G;

void bar(void) {
  G = stdout;
}
void foo(int *a) {
  fprintf(G, "foo: %p\n", a);
}
