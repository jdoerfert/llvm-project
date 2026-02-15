// RUN: %libomptarget-compile-run-and-check-generic
//
#include <assert.h>

int g = 0;
#pragma omp declare target to(g)

void foo() {
#pragma omp parallel num_threads(1)
  __atomic_fetch_add(&g, 1, __ATOMIC_RELAXED);
}

void bar() {
#pragma omp parallel num_threads(2)
  __atomic_fetch_add(&g, 1, __ATOMIC_RELAXED);
}

void baz() {
#pragma omp parallel
  __atomic_fetch_add(&g, 1, __ATOMIC_RELAXED);
}

int main() {
#pragma omp target teams num_teams(1) thread_limit(1)
  foo();
#pragma omp target update from(g)
  assert(g == 1 && "Expected 1");

#pragma omp target teams num_teams(1) thread_limit(1)
  bar();
#pragma omp target update from(g)
  assert(g == 2 && "Expected 2");

#pragma omp target teams num_teams(1) thread_limit(1)
  baz();
#pragma omp target update from(g)
  assert(g == 3 && "Expected 3");
}
