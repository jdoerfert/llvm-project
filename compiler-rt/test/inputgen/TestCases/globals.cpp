// RUN: %clangxx_inputgen_full_gen
// RUN: %clangxx_inputgen_full_replay_gen

// RUN: %inputgen_gen > %inputgen_gen.0.0.0.inp.gen.out
// RUN: %inputgen_repl_gen %inputgen_gen.0.0.0.42.inp > %inputgen_gen.0.0.0.inp.repl.out
// RUN: diff %inputgen_gen.0.0.0.inp.gen.out %inputgen_gen.0.0.0.inp.repl.out

extern "C" int printf(const char *__restrict __format, ...);

int global_int;
extern int external_global_int;
extern const int const_external_global_int;
static int static_global_int = 42;
int global_int_array[10];
char global_char_array[11];

__attribute__((inputgen_entry)) void foo(double *a, double *b, double *m,
                                         int n) {
    printf("GI %d\n", global_int);
    printf("EGI %d\n", external_global_int);
    printf("CEGI %d\n", const_external_global_int);
    printf("SGI %d\n", static_global_int);
    printf("GIA %d\n", global_int_array[2]);
    printf("GCA %d\n", (int)global_char_array[10]);
}
