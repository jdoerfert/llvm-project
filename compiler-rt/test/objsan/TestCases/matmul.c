// RUN: %clangxx_inputgen_gen -c %s -o %t.gen.o
// RUN: %clangxx_inputgen_replay_gen -c %s -o %t.repl.o
// XFAIL: *

#include <stdio.h>

__attribute__((inputgen_entry)) void matmul(double *a, double *b, double *m,
                                            int n) {

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      double t = 0.0;
      for (int k = 0; k < n; ++k)
        t += a[i * n + k] * b[k * n + j];
      m[i * n + j] = t;
    }
  }
  printf("m[n/2] = m[%i] : %lf\n", n / 2, m[n / 2]);
}
