// clang-format off
// RUN: %libomptarget-compile-generic -DN=5 -mllvm -amdgpu-enable-offload-sanitizer -fopenmp-cuda-mode
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefixes=CHECK5
// RUN: %libomptarget-compileopt-generic -DN=5 -mllvm -amdgpu-enable-offload-sanitizer -fopenmp-cuda-mode
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefixes=CHECK5
// RUN: %libomptarget-compileopt-generic -DN=2 -mllvm -amdgpu-enable-offload-sanitizer -fopenmp-cuda-mode
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefixes=CHECK2
// clang-format on

#include <stdio.h>

int main() {
  int idx = 3, res;
#pragma omp target teams map(from : res) ompx_bare num_teams(1) thread_limit(1)
  {
    int A[10];
    A[3] = 42;
    res = A[idx];
  }
  printf("Res: %d\n", res);
  // CHECK5: Res: 4
  // CHECK2: Res: 3
}
