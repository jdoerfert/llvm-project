#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef int IntTy;

IntTy *foo() {

  IntTy *a;
  IntTy *b;
  IntTy *c;

  int N = 1000;

  a = (IntTy *)malloc(sizeof(IntTy) * N * N);
  b = (IntTy *)malloc(sizeof(IntTy) * N * N);
  c = (IntTy *)malloc(sizeof(IntTy) * N * N);

  // I ranges from -N, -N + 1, ..., 0, 1, 2, ... N
  // Square ranges from N^2 - 1, (N-1)^2 ..., 0, 1, 4, 9, 16, ... N^2 - 1
#pragma omp target teams map(from : a [0:N * N])
  {
    int Square;
    for (IntTy I = -N; I <= N; I++) {
      Square = I * I;
      // legal boundary access.
      if (I == N || I == -N) {
        Square = Square - 1;
      }
      a[Square] = I;
    }
  }

  return a;
}

void printArray(int *a, int N) {

  for (IntTy I = -N; I <= N; I++) {
    int Index = I * I;
    if (I == N || I == -N) {
      Index = Index - 1;
    }
    printf("a: %d ", a[Index]);
  }
}

int main() {
  int N = 1000;
  int *a = foo();
  printArray(a, N);
}
