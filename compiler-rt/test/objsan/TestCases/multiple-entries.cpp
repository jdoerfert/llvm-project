// RUN: %clangxx_inputgen_gen -c %s -o %t.gen.o
// RUN: %clangxx_inputgen_link_gen %t.gen.o -o %t.gen.exe

// RUN: %clangxx_inputgen_replay_gen -c %s -o %t.repl.o
// RUN: %clangxx_inputgen_link_replay_gen %t.repl.o -o %t.repl.exe

// RUN: %t.repl.exe 2>&1 | FileCheck %s --check-prefix=REPL
// RUN: %t.gen.exe -1  2>&1 | FileCheck %s --check-prefix=GEN

// REPL:   Num available functions: 2
// REPL:   Available functions:
// REPL:     0: _Z3fooi
// REPL:     1: _Z3bari
// REPL-NOT:     bar

// GEN:   Num available functions: 2
// GEN:   Available functions:
// GEN:     0: _Z3fooi
// GEN:     1: _Z3bari
// GEN-NOT:     bar

#include <stdio.h>

__attribute__((inputgen_entry)) void foo(int d) { printf("d %d\n", d); }

__attribute__((inputgen_entry)) void bar(int d) { printf("d %d\n", d); }

void baz(int d) { printf("d %d\n", d); }
