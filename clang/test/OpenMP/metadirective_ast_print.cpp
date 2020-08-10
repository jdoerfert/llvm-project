// RUN: %clang_cc1 -verify -fopenmp -ast-print %s | FileCheck %s
// expected-no-diagnostics

int main() {
  int N = 15;
#pragma omp metadirective when(user = {condition(N > 10)} \
                               : parallel for) default()
  // CHECK: #pragma omp metadirective when(N > 10: parallel for) default()
  for (int i = 0; i < N; i++)
    ;

#pragma omp metadirective when(user = {condition(N < 10)} \
                               :) default(parallel for)
  // CHECK: #pragma omp metadirective when(N < 10: ) default(parallel for)
  for (int i = 0; i < N; i++)
    ;

  return 0;
}
