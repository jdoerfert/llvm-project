// RUN: %clangxx_inputgen_full_gen
// RUN: %clangxx_inputgen_full_replay_gen

// RUN: %inputgen_repl_gen 2>&1 | FileCheck %s --check-prefix=REPL
// RUN: %inputgen_gen -1  2>&1 | FileCheck %s --check-prefix=GEN

// REPL:   Num available functions: 2
// REPL:   Available functions:
// REPL:     0: _Z3fooi
// REPL:     1: _Z3bari
// REPL-NOT:     baz
// REPL-NOT:     2:

// GEN:   Num available functions: 2
// GEN:   Available functions:
// GEN:     0: _Z3fooi
// GEN:     1: _Z3bari
// GEN-NOT:     baz
// GEN-NOT:     2:

#include <stdio.h>

__attribute__((inputgen_entry)) void foo(int d) { printf("d %d\n", d); }

__attribute__((inputgen_entry)) void bar(int d) { printf("d %d\n", d); }

void baz(int d) { printf("d %d\n", d); }
