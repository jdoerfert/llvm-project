#include <cmath>

void math(short s, int i, float f, double d) {
  sin(s);
  sin(i);
  sin(f);
  sin(d);
}

void foo(short s, int i, float f, double d, long double ld) {
  //sin(ld);
  math(s, i, f, d);
#pragma omp target
  { math(s, i, f, d); }
}
