#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef int IntTy;

typedef struct {
  IntTy *a;
  IntTy *b;
  IntTy *c;
} Product;

Product *foo(IntTy N) {

  IntTy *a;
  IntTy *b;
  IntTy *c;

  a = (IntTy *)malloc(sizeof(IntTy) * N);
  b = (IntTy *)malloc(sizeof(IntTy) * N);
  c = (IntTy *)malloc(sizeof(IntTy) * N);

#pragma omp target teams map(from : a [0:N], b [0:N], c [0:N])
  {
    for (IntTy I = 0; I < N; I++) {
      a[I] = I;
      b[I] = I;
      c[I] = a[I] + b[I];
    }
  }

  Product *P = (Product *)malloc(sizeof(Product));
  P->a = a;
  P->b = b;
  P->c = c;
  return P;
}

void printProduct(Product *P, IntTy N) {

  IntTy *a = P->a;
  IntTy *b = P->b;
  IntTy *c = P->c;

  for (IntTy i = 0; i < N; i++) {
    printf("a: %d, b:%d, c:%d\n", a[i], b[i], c[i]);
  }
}

int main() {

  IntTy N = 1000;
  Product *P = foo(N);
  printProduct(P, N);
}
