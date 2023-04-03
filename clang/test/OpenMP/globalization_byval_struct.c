// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp -x c -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp -x c -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s
// expected-no-diagnostics

extern int printf(const char *, ...);

struct S {
  int a;
  float b;
};

// CHECK: define{{.*}}void @test(%struct.S* noundef byval(%struct.S) align {{[0-9]+}} [[arg:%[0-9a-zA-Z]+]])
// CHECK: [[g:%[0-9a-zA-Z]+]] = call align {{[0-9]+}} i8* @__kmpc_alloc_shared
// CHECK: bitcast i8* [[g]] to %struct.S*
// CHECK: bitcast %struct.S* [[arg]] to i8**
// CHECK: call void [[cc:@__copy_constructor[_0-9a-zA-Z]+]]
// CHECK: void [[cc]]
void test(struct S s) {
#pragma omp parallel for
  for (int i = 0; i < s.a; ++i) {
    printf("%i : %i : %f\n", i, s.a, s.b);
  }
}

void foo() {
  #pragma omp target teams num_teams(1)
  {
    struct S s;
    s.a = 7;
    s.b = 11;
    test(s);
  }
}
