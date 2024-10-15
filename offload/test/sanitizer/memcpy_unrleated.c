#include <omp.h>
#include <stdio.h>

int main() {

  int A[10] = {1};
  int D = omp_get_default_device();
  int H = omp_get_initial_device();
  int *DP = omp_target_alloc(8, D);
  printf("Copy 1 %p -> %p\n", DP, &A[0]);
  omp_target_memcpy(DP, &A[0], 8, 0, 0, D, H);
  printf("Copy 2 %p -> %p\n", DP, &A[0]);
  omp_target_memcpy(DP, &A[0], 4, 0, 0, D, H);
  int r;
#pragma omp target map(from : r)
  { r = *DP; }
  printf("R %i\n", r);
  return 0;
}
