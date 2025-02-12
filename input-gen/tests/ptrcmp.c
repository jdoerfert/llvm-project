#include <stdint.h>
#include <stdio.h>

void entry(void *p) {
  int **Q = *(int***)p;
  int E = 12;
  for (int i = 0; i < E; i++) {
    int *B = *((int**)p + 8 * i + 4);
    printf("Ptr before ptr to int %p\n", B);
    Q[i] = B;
    if (i < 8)
      B[i] = i;
    if (!B)  {
      printf("hit nullptr %i : %p\n", i, B);
      if (i == 11)
        E++;
    }
  }
  for (int i = 0; i < 10; i++) {
    printf("Ptr %i %p\n", i, Q[i]);
    Q[i][i] = i;
  }
}
