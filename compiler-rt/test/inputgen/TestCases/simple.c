// RUN: %clangxx_inputgen_full_gen
// RUN: %clangxx_inputgen_full_replay_gen

// RUN: for i in $(seq 0 5); do %inputgen_gen $i > %inputgen_gen.$i.0.0.inp.gen.out; done
// RUN: for i in $(seq 0 5); do %inputgen_repl_gen %inputgen_gen.$i.0.0.inp $i > %inputgen_gen.$i.0.0.inp.repl.out; done
// RUN: for i in $(seq 0 5); do diff %inputgen_gen.$i.0.0.inp.gen.out %inputgen_gen.$i.0.0.inp.repl.out; done

extern "C" int printf(const char *__restrict __format, ...);

__attribute__((inputgen_entry)) void arg(int n) {
  printf("ARG %d\n", n);
}

__attribute__((inputgen_entry)) void mem1(int *n) {
  printf("MEM1 %d\n", *n);
}

__attribute__((inputgen_entry)) void mem2(int &n) {
  printf("MEM2 %d\n", n);
}

__attribute__((inputgen_entry)) void arg_mem_offset(int *m, int *n) {
  printf("MEM2 %d\n", *n);
}

__attribute__((inputgen_entry)) void mem_pos_offset(int *n) {
  printf("MEM2 %d\n", *(n + 1));
}

__attribute__((inputgen_entry)) void mem_neg_offset(int *n) {
  printf("MEM2 %d\n", *(n - 1));
}
