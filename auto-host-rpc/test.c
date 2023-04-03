#include <stdio.h>

#pragma omp begin declare target device_type(nohost)

struct ddd {
  int num;
  int a;
  float b;
};

void foo() {
  FILE *fp = fopen("main.cpp", "r");
  struct ddd d;
  fprintf(fp, "%d", 6);
  fprintf(fp, "%f%d%s", 6.0f, 1, "hello");
  fscanf(fp, "%d", &d.a);
}

#pragma omp end declare target
