// Test target codegen - host bc file has to be created first.
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -fopenmp-cuda-mode -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -fopenmp-cuda-mode -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -fopenmp-cuda-mode -x c++ -triple i386-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -fopenmp-cuda-mode -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -fopenmp-cuda-mode -fexceptions -fcxx-exceptions -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp -fopenmp-cuda-mode -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-cuda-mode -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-cuda-mode -x c++ -triple i386-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-cuda-mode -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-cuda-mode -fexceptions -fcxx-exceptions -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s

// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// CHECK-DAG: {{@__omp_offloading_.+l31}}_exec_mode = weak constant i8 0
// CHECK-DAG: {{@__omp_offloading_.+l36}}_exec_mode = weak constant i8 0
// CHECK-DAG: {{@__omp_offloading_.+l41}}_exec_mode = weak constant i8 0
// CHECK-DAG: {{@__omp_offloading_.+l46}}_exec_mode = weak constant i8 0

#define N 1000

template<typename tx>
tx ftemplate(int n) {
  tx a[N];
  short aa[N];
  tx b[10];

  #pragma omp target simd
  for(int i = 0; i < n; i++) {
    a[i] = 1;
  }

  #pragma omp target simd
  for (int i = 0; i < n; i++) {
    aa[i] += 1;
  }

  #pragma omp target simd
  for(int i = 0; i < 10; i++) {
    b[i] += 1;
  }

  #pragma omp target simd reduction(+:n)
  for(int i = 0; i < 10; i++) {
    b[i] += 1;
  }

  return a[0];
}

int bar(int n){
  int a = 0;

  a += ftemplate<int>(n);

  return a;
}

// CHECK: call i32 @__kmpc_target_init({{.+}}, i1 true, i1 false)
// CHECK: call void @__kmpc_target_deinit({{.+}}, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.+}}, i1 true, i1 false)
// CHECK: call void @__kmpc_target_deinit({{.+}}, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.+}}, i1 true, i1 false)
// CHECK: call void @__kmpc_target_deinit({{.+}}, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.+}}, i1 true, i1 false)
// CHECK: call void @__kmpc_target_deinit({{.+}}, i1 true)

#endif
