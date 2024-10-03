// clang-format off
// RUN: %libomptarget-compile-generic -DN=5 -mllvm -amdgpu-enable-offload-sanitizer -fopenmp-cuda-mode -loffload.kernels
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic
// clang-format on

#include <stdio.h>

int main() {
  int idx = 3, res = 0;
#pragma omp target teams map(tofrom : res) ompx_bare num_teams(1)              \
    thread_limit(1)
  {
    int A[10];
    A[3] = 42;
    res = A[idx];
  }
  printf("Res: %d\n", res);
  // CHECK: Res: 42
}
