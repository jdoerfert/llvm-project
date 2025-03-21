// RUN: %clangxx_inputgen_full_gen
// RUN: %clangxx_inputgen_full_replay_gen

// RUN:  for i in $(seq 0 10); do %inputgen_gen $i > %inputgen_gen.$i.0.0.inp.gen.out; done
// RUN:  for i in $(seq 0 10); do %inputgen_repl_gen %inputgen_gen.$i.0.0.inp $i > %inputgen_gen.$i.0.0.inp.repl.out; done
// RUN:  for i in $(seq 0 10); do diff %inputgen_gen.$i.0.0.inp.gen.out %inputgen_gen.$i.0.0.inp.repl.out; done
// RUN: (for i in $(seq 0 10); do cat %inputgen_gen.*.0.0.inp.repl.out; done) | FileCheck %s

extern "C" int printf(const char *__restrict __format, ...);

// CHECK: ARG
__attribute__((inputgen_entry)) void arg(int n) {
  printf("ARG %d\n", n);
}
// CHECK: MEM1
__attribute__((inputgen_entry)) void mem1(int *n) {
  printf("MEM1 %d\n", *n);
}
// CHECK: MEM2
__attribute__((inputgen_entry)) void mem2(int &n) {
  printf("MEM2 %d\n", n);
}
// CHECK: MEM3
__attribute__((inputgen_entry)) void arg_mem_offset(int *m, int *n) {
  printf("MEM3 %d\n", *n);
}
// CHECK: MEM4
__attribute__((inputgen_entry)) void mem_pos_offset(int *n) {
  printf("MEM4 %d\n", *(n + 1));
}
// CHECK: MEM5
__attribute__((inputgen_entry)) void mem_neg_offset(int *n) {
  printf("MEM5 %d\n", *(n - 1));
}
// CHECK: MEM6
__attribute__((inputgen_entry)) void load_after_store(double *n) {
  *n = 30140;
  printf("MEM6 %f\n", *n);
}
// CHECK: MEM7
__attribute__((inputgen_entry)) void store_after_load(double *n) {
  printf("MEM7 %f\n", *n);
  *n = 30140;
}
// CHECK: MEM8
__attribute__((inputgen_entry)) void store_after_load_partial(char *n) {
  printf("MEM8 %f\n", *(float *)n);
  *(double *)n = 301400;
}
// CHECK: MEM9
// CHECK: MEM9
__attribute__((inputgen_entry)) void char_store_rem_0(char *n) {
  printf("MEM9 %d\n", (int)*n);
  *(int *)n = 39;
  printf("MEM9 %d\n", *(int *)n);
}
// CHECK: MEM10
// CHECK: MEM10
__attribute__((inputgen_entry)) void char_store_rem_1(char *n) {
  printf("MEM10 %d\n", (int)*(n + 1));
  *(int *)n = 39;
  printf("MEM10 %d\n", *(int *)n);
}
