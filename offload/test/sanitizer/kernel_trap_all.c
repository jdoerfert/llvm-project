
// clang-format off
// RUN: %libomptarget-compile-generic -g -mllvm -amdgpu-enable-offload-sanitizer
// RUN: %not --crash env -u LLVM_DISABLE_SYMBOLIZATION OFFLOAD_TRACK_NUM_KERNEL_LAUNCH_TRACES=1 %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefixes=CHECK,TRACE,DEBUG
// RUN: %not --crash %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefixes=CHECK
// clang-format on

// REQUIRES: amdgpu

#include <omp.h>

int main(void) {

#pragma omp target teams
  {
#pragma omp parallel
    __builtin_trap();
  }
}
// clang-format off
// CHECK: OFFLOAD ERROR: Kernel {{.*}} (__omp_offloading_{{.*}}_main_{{.*}})
// CHECK: OFFLOAD ERROR: execution interrupted by hardware trap instruction
// CHECK: Triggered by thread <{{[0-9]*}},0,0> block <{{[0-9]*}},0,0> PC 0x{{.*}}
// TRACE:     launchKernel
// NDEBG:     main
// DEBUG:     main {{.*}}kernel_trap_all.c:
// clang-format on
