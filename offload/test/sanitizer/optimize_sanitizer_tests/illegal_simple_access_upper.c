#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef int IntTy;

IntTy *foo(int Size) {

  IntTy *a;

  a = (IntTy *)malloc(sizeof(IntTy) * Size);

#pragma omp target teams map(from : a [0:Size])
  {
    for (IntTy I = 0; I < Size + 1; I++) {
      a[I] = I;
    }
  }

  return a;
}

void printArray(int *a, int Size) {

  for (IntTy I = 0; I < Size; I++) {
    printf("a: %d ", a[I]);
  }
}

int main() {

  int N = 1000;
  int *a = foo(N);
  printArray(a, N);
}
