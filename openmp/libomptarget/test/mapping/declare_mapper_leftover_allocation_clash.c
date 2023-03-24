// RUN: %libomptarget-compile-generic 
// RUN: %libomptarget-run-generic | %fcheck-generic
// Taken from https://github.com/llvm/llvm-project/issues/61636
#include <stdio.h>

#define TRUE 1
#define FALSE 0

typedef struct {
  int i1, i2, i3;
} TY1;


#pragma omp declare mapper(TY1 t) map(to:t.i1) map(from: t.i3)


unsigned foo() {
  TY1 t1[2];
  for(int i = 0; i < 2; i++)
    t1[i].i1 = 1;

#pragma omp target map(tofrom:t1)
  for (int i = 0; i < 2; i++)
    t1[i].i3 = t1[i].i1;

  for (int i = 0; i < 2; i++) {
    if (t1[i].i3 != t1[i].i1) {
      printf("failed. t1[%d].i3 (%d) != t1[%d].i1 (%d)\n", i, t1[i].i3, i, t1[i].i1);
      return FALSE;
    }
  }

  return TRUE;
}

unsigned bar() {
  TY1 t2[5];

  for(int i = 0; i < 5; i++)
    t2[i].i1 = 2;

#pragma omp target map(tofrom:t2)
  for (int i = 0; i < 5; i++)
    t2[i].i3 = t2[i].i1;

  for (int i = 0; i < 5; i++) {
    if (t2[i].i3 != t2[i].i1) {
      printf("failed. t2[%d].i3 (%d) != t2[%d].i1 (%d)\n", i, t2[i].i3, i, t2[i].i1);
      return FALSE;
    }
  }

  return TRUE;
}

int main() {
  if (foo() != TRUE)
    return 1;
  if (bar() != TRUE)
    return 2;
  // CHECK: passed
  printf("passed\n");
  return 0;
}
