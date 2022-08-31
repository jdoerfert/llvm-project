#include <omp.h>
#include <stdio.h>
#include <stdint.h>

#define NNN 512

struct IdentTy {
  int32_t reserved_1;  /**<  might be used in Fortran; see above  */
  int32_t flags;       /**<  also f.flags; KMP_IDENT_xxx flags; KMP_IDENT_KMPC
                            identifies this union member  */
  int32_t reserved_2;  /**<  not really used in Fortran any more; see above */
  int32_t reserved_3;  /**<  source[4] in Fortran, do not use for C++  */
  char const *psource; /**<  String describing the source location.
                       The string is composed of semi-colon separated fields
                       which describe the source file, the function and a pair
                       of line numbers that delimit the construct. */
};


enum class RedOp : int8_t {
  ADD,
  MUL,
  // ...
};

enum class RedDataType : int8_t {
  INT8,
  INT16,
  INT32,
  INT64,
  FLOAT,
  DOUBLE,
  CUSTOM
};

enum class RedWidth : int8_t {
  WARP,
  TEAM,
  LEAGUE,
};
struct ReductionInfo {
  RedOp Op;
  RedDataType DT;
  RedWidth Width;
  int16_t NumThreads;
  int16_t ElementSize;
  int32_t NumElements;
  void *CopyConstWrapper = nullptr;
};

void __llvm_omp_tgt_reduce(IdentTy *Loc, ReductionInfo *RI, char * Location);

#pragma omp begin declare target device_type(nohost)
  static ReductionInfo RI {
RedOp::ADD,
 RedDataType::INT32,
RedWidth::TEAM,
0,
 4,
 1,
 nullptr};
#pragma omp end declare target

int main() {

  int A[NNN], r = 0;
  for (int i =0 ; i < NNN; ++i) {
    A[i] = 1;
  }

  #pragma omp target data map(to:A[:NNN])
  {
    #pragma omp target teams num_teams(1)
    {
      #pragma omp parallel
      {
      }
    }

    #pragma omp target teams num_teams(1) thread_limit(NNN) map(tofrom:A[:NNN], r)
    {

      #pragma omp parallel
      {
        int lcl_r = 0;
        #pragma omp for
        for (int i = 0; i < NNN; ++i) {
          lcl_r += A[i];
        }
        __llvm_omp_tgt_reduce(nullptr, &RI, (char*)&lcl_r);
        if (!omp_get_thread_num())
          r += lcl_r;
       }
      #pragma omp parallel
      {
        int lcl_r = 0;
        #pragma omp for
        for (int i = 0; i < NNN; ++i) {
          lcl_r += A[i];
        }
        __llvm_omp_tgt_reduce(nullptr, &RI, (char*)&lcl_r);
        if (!omp_get_thread_num())
          r += lcl_r;
       }
      /*asm volatile("exit;");*/
    }

    printf("R: %i\n", r);

    r = 0;
    #pragma omp target teams num_teams(1) thread_limit(NNN) map(tofrom:A[:NNN], r)
    {
    #pragma omp parallel for reduction(+:r)
    for (int i = 0; i < NNN; ++i)
    {
      r += A[i];
    }
    #pragma omp parallel for reduction(+:r)
    for (int i = 0; i < NNN; ++i)
    {
      r += A[i];
    }
    }
  }

  printf("R: %i\n", r);
  return 0;
}
