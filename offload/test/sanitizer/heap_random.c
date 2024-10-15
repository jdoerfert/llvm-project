// clang-format off
// RUN: %libomptarget-compileopt-generic -fsanitize=offload
// RUN: not %libomptarget-run-generic 2> %t.out
// RUN: %fcheck-generic --check-prefixes=CHECK < %t.out
// clang-format on

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: x86_64-pc-linux-gnu
// UNSUPPORTED: x86_64-pc-linux-gnu-LTO
// UNSUPPORTED: s390x-ibm-linux-gnu
// UNSUPPORTED: s390x-ibm-linux-gnu-LTO

// Align lines.

int main(void) {

  int X = 0;
  int *Random = &X;
#pragma omp target
  { *Random = 99; }
  // CHECK: 0x{{[a-f0-9]*}} is located {{[0-9]*}} bytes inside
}
