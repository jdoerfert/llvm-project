// RUN: %clang_cc1 -verify -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -x c++ -emit-llvm %s -triple %itanium_abi_triple -fexceptions -fcxx-exceptions -o - | FileCheck %s
// expected-no-diagnostics

#include <cmath>

int main() {
  double a(0);
  return (std::fpclassify(a) != FP_ZERO);
}
