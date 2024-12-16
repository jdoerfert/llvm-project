#include <stdio.h>

struct Args {
  int n;
  double *a;
  double *b;
  double *m;
};

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((inputgen_entry))
void entry(struct Args *AP) {

  struct Args args = *AP;
  for (int i = 0; i < args.n; ++i) {
    for (int j = 0; j < args.n; ++j) {
      double t = 0.0;
      for (int k = 0; k < args.n; ++k)
        t += args.a[i * args.n + k] * args.b[k * args.n + j];
      args.m[i * args.n + j] = t;
    }
  }
  printf("args.m[args.n/2] = args.m[%i] : %lf\n", args.n / 2, args.m[args.n / 2]);
}
#ifdef __cplusplus
}
#endif
