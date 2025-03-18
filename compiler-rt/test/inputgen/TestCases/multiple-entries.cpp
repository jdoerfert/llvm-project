// RUN: %clangxx_inputgen_full_gen
// RUN: %clangxx_inputgen_full_replay_gen

// RUN: %inputgen_repl_gen 2> %t.avail.out || true
// RUN: cat %t.avail.out | FileCheck %s
// RUN: %inputgen_gen 2> %t.out || true
// RUN: cat %t.avail.out | FileCheck %s

// CHECK:   Num available functions: 2
// CHECK:   Available functions:
// CHECK:     0: _Z3fooi
// CHECK:     1: _Z3bari
// CHECK-NOT:     baz
// CHECK-NOT:     2:

#include <stdio.h>

__attribute__((inputgen_entry)) void foo(int d) { printf("d %d\n", d); }

__attribute__((inputgen_entry)) void bar(int d) { printf("d %d\n", d); }

void baz(int d) { printf("d %d\n", d); }
