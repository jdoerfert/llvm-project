// TODO: make this a test case
#include <omp.h>
#include <stdio.h>
#define N 12

void foo(int *A) {
  if (omp_get_thread_num() == N / 2)
  {
    for (int i = 0; i < 4; ++i)
      A[i] = omp_get_team_size(i);
    #pragma omp parallel
    {
      for (int i = 0; i < 4; ++i)
        A[i+4] = omp_get_team_size(i);
      #pragma omp parallel
      {
        for (int i = 0; i < 4; ++i)
          A[i+8] = omp_get_team_size(i);
      }
    }
  }
}

int main() {
  int A[N];
  for (int i = 0; i < N; ++i)
    A[i] = 42424242;
  #pragma omp target teams num_teams(1) thread_limit(N) map(tofrom:A[:N])
  {
    #pragma omp parallel
    foo(A);
  }
  for (int i = 0; i < N; ++i)
    printf("%i : %i\n", i, A[i]);
  return 0;
}
