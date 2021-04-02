// RUN: %clang++ -x cuda --offload-arch=sm_70 -nocudalib -nocudainc %s -o - | FileCheck %s
// REQUIRES: nvptx-registered-target
//
// CHECK:.global .align 1 .b8 llvm_$_embedded_$_module[

__device__ void foo(int mask) {
  __nvvm_bar_warp_sync(mask);
}
