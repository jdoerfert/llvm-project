// RUN: %clangxx_inputgen_full_gen
// RUN: %clangxx_inputgen_full_replay_gen

// RUN: %inputgen_gen > %inputgen_gen.0.0.0.inp.gen.out
// RUN: %inputgen_repl_gen %inputgen_gen.0.0.0.42.inp > %inputgen_gen.0.0.0.inp.repl.out
// RUN: diff %inputgen_gen.0.0.0.inp.gen.out %inputgen_gen.0.0.0.inp.repl.out

extern "C" int printf(const char *__restrict __format, ...);

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
