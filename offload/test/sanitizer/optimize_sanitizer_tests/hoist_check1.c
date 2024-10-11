#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef int IntTy;

IntTy *foo(IntTy Size) {

  IntTy *a;
  a = (IntTy *)malloc(sizeof(IntTy) * Size);

  //Ideally the check for this pointer should be hoisted out of the loop.  
  IntTy *Hoistable = (IntTy *)malloc(sizeof(IntTy));
  
#pragma omp target teams map(from : a[0:Size])
  {
    for (IntTy I = 0; I < Size; I++) {
      Hoistable[0] = 1;
      a[I] = I;
    }
  }

  return a;
}

void printArray(IntTy *a, IntTy Size) {

  for (IntTy I = 0; I < Size; I++) {
    printf("a: %d ", a[I]);
  }
}

int main() {

  int N = 10000000;
  int *a = foo(N);
  //printArray(a, N);
}
