// RUN: %libomptarget-compileopt-generic -fsanitize=offload
// RUN: %libomptarget-run-generic 2>&1 > %t.out
// RUN: %fcheck-generic --check-prefixes=CHECK < %t.out

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
  int N = 10;
  int N_SZ = sizeof(int) * N;

  int Device = omp_get_default_device();

  int *Buffer = (int *)malloc(N_SZ);

#pragma omp target map(tofrom : Buffer[0 : N])
  {
    for (int i = 0; i < N; i++) {
      Buffer[i] = i;
    }
  }

  for (int i = 0; i < N; i++) {
    printf("%d\n", Buffer[i]);
  }
  free(Buffer);

  return 0;
}

// CHECK: 0
// CHECK-NEXT: 1
// CHECK-NEXT: 2
// CHECK-NEXT: 3
// CHECK-NEXT: 4
// CHECK-NEXT: 5
// CHECK-NEXT: 6
// CHECK-NEXT: 7
// CHECK-NEXT: 8
// CHECK-NEXT: 9
