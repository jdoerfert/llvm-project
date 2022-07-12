#pragma omp begin assumes ext_no_call_asm
void non_recursive_asm(void) {
  asm volatile("barrier.sync %0;" : : "r"(1) : "memory");
}
#pragma omp end assumes ext_no_call_asm
void recursive_asm(void) {
  asm volatile("barrier.sync %0;" : : "r"(1) : "memory");
}
