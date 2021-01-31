// RUN: %clang_cc1 -triple=x86_64-pc-win32 -verify -fopenmp -x c -std=c99 -fms-extensions -Wno-pragma-pack %s
// expected-no-diagnostics

#pragma omp begin declare variant match(implementation={extension(match_any)})
#pragma omp begin declare variant match(device = {vendor(cray, ibm)})
 this is never reached, we cannot have a cray ibm compiler hybrid, I hope.
#pragma omp end declare variant
#pragma omp end declare variant
