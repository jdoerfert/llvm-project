
#include <stdlib.h>
int main(int argc, char **argv) {
  int *A = calloc(argc, 4);
  for (int i = 0; i < atoi(argv[2]); ++i)
    A[i] = i;
  return A[atoi(argv[3])];
}
