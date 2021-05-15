// Check that the CHECK lines are generated for clang-generated functions
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp %s -emit-llvm -o - | FileCheck --check-prefix=OMP %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -emit-llvm -o - | FileCheck --check-prefix=NOOMP %s

void t0() {
  #pragma omp parallel
  {
  }
  #pragma omp parallel
  {
  }
  #pragma omp parallel
  {
  }
}

void t1() {
  #pragma omp parallel
  {
  }
  #pragma omp parallel
  {
  }
  #pragma omp parallel
  {
  }
}

void t2() {
  #pragma omp parallel
  {
  }
  #pragma omp parallel
  {
  }
  #pragma omp parallel
  {
  }
}
