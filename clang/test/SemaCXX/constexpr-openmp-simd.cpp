// RUN: %clang_cc1 -verify -std=c++2a -fsyntax-only -fopenmp -triple x86_64-apple-macosx10.14.0 %s
// RUN: %clang_cc1 -verify -std=c++2a -fsyntax-only -fopenmp -triple x86_64-apple-macosx10.14.0 %s -fno-signed-char
// RUN: %clang_cc1 -verify -std=c++2a -fsyntax-only -fopenmp -triple aarch64_be-linux-gnu %s
// RUN: %clang_cc1 -emit-llvm -std=c++2a -fopenmp -triple aarch64_be-linux-gnu %s -DEMIT_IR -o - | FileCheck %s

constexpr int good(int r) {

  #pragma omp simd
  for (int i = 0; i != 10; ++i)
    r += 1;

  return r;
}

int test1() {
  if (good(10) == 20)
    return 42;
  return good(7);
// Make sure this is folded to 42 and not 17.
// CHECK: ret i32 42
}

#ifndef EMIT_IR
constexpr int bad(int r) {
  #pragma omp simd private(r) // expected-error {{OpenMP simd statement with clauses not allowed in constexpr function}}
  for (int i = 0; i != 10; ++i)
    r += 1;

  return r;
}

int test2() {
  return bad(10);
}
#endif
