#include <stdio.h>
#include <stdlib.h>

struct LL {
  int P;
  struct LL* next;
} L;

__attribute__((inputgen_entry))
int main(int argc, char **argv) {
  struct LL *p = &L;
  int sum = 0;
  while (p) {
    sum += p->P;
    p = &p->next[abs(p->P) % 100];
  }
  printf("Sum: %i\n", sum);
  return 0;
}
