// clang-format off
// RUN: %libomptarget-compile-generic -DN=5 -mllvm -amdgpu-enable-offload-sanitizer -fopenmp-cuda-mode
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefixes=CHECK5
// RUN: %libomptarget-compileopt-generic -DN=2 -mllvm -amdgpu-enable-offload-sanitizer -fopenmp-cuda-mode
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefixes=CHECK2
// RUN: %libomptarget-compile-generic -DN=101 -mllvm -amdgpu-enable-offload-sanitizer -fopenmp-cuda-mode
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefixes=CRASH
// RUN: %libomptarget-compileopt-generic -DN=101 -mllvm -amdgpu-enable-offload-sanitizer -fopenmp-cuda-mode
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefixes=CRASH
// clang-format on

#include <stdio.h>

int main() {
  int idx = 3, res = -1, UB = N;
#pragma omp target map(from : res)
  {
    int A[100];
    for (int i = 0; i < 100; ++i)
      A[i] = i;
    for (int i = 0; i < UB; ++i)
      A[i]++;
    res = A[idx];
  }
  printf("Res: %d\n", res);
  // CHECK5: Res: 4
  // CHECK2: Res: 3
}
