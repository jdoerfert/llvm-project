// RUN: %clangxx_inputgen_full_gen
// RUN: %clangxx_inputgen_full_replay_gen

// RUN: %inputgen_gen > %inputgen_gen.0.0.0.inp.gen.out
// RUN: %inputgen_repl_gen %inputgen_gen.0.0.0.42.inp > %inputgen_gen.0.0.0.inp.repl.out
// RUN: diff %inputgen_gen.0.0.0.inp.gen.out %inputgen_gen.0.0.0.inp.repl.out

extern "C" int printf(const char *__restrict __format, ...);

__attribute__((inputgen_entry)) void vec_init(double *a, int n) {
  for (int i = 0; i < n; ++i) {
      a[i] = 30104;
  }
  printf("a[n/2] = a[%i] : %lf\n", n / 2, a[n / 2]);
}
