#include <omp.h>
#include <stdint.h>
#include <stdio.h>

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

enum RedChoice : int8_t {
  RED_ITEMS_FULLY = 1,
  RED_ITEMS_PARTIALLY = 2,
  RED_ATOMIC_WITH_OFFSET = 4,
  RED_ATOMIC_AFTER_TEAM = 8,
  RED_ATOMIC_AFTER_WARP = 16,
  RED_LEAGUE_BUFFERED_DIRECT = 32,
  RED_LEAGUE_BUFFERED_ATOMIC = 64,
  RED_LEAGUE_BUFFERED_SYNCHRONIZED = 124,
};

struct ReductionInfo {
  RedOp Op;
  RedDataType DT;
  RedWidth Width;
  RedChoice RC;
  int8_t BatchSize;
  int32_t NumParticipants;
  int32_t NumElements;
  uint32_t* LeagueCounterPtr;
  char *LeagueBuffer = nullptr;
  void *CopyConstWrapper = nullptr;
};

struct Timer {
  double start;
  const char *Name;
  Timer(const char *Name) : start(omp_get_wtime()), Name(Name) {}
  ~Timer() {
    double end = omp_get_wtime();
    printf("Time: %70s : %lfs\n", Name, end - start);
  }
};

void __llvm_omp_tgt_reduce(IdentTy *Loc, ReductionInfo *RI, char *Location,
                           char *Output);

void reduce_host(int *A, int *r, int *lr, int NumThreads, int NE) {
  {
    Timer T(__PRETTY_FUNCTION__);
    {
#pragma omp parallel for reduction(+ : r[:NE])
      for (int t = 0; t < NumThreads; ++t) {
        for (int i = 0; i < NE; ++i) {
          r[i] += A[i];
        }
      }
    }
  }
}
template <int NE>
void reduce_old(int *A, int *r, int *lr, int NumThreads, int Teams) {
  {
    Timer T(__PRETTY_FUNCTION__);
#pragma omp target teams distribute parallel for reduction(+ : r[:NE]) num_teams(Teams) thread_limit(NumThreads)
    for (int t = 0; t < NumThreads * Teams; ++t) {
      for (int i = 0; i < NE; ++i) {
        r[i] += A[i];
      }
    }
  }
}
void reduce_old(int *A, int *r, int *lr, int NumThreads, int NE, int Teams) {
  switch (NE) {
  case 1:
    return reduce_old<1>(A, r, lr, NumThreads, Teams);
  //case 2:
    //return reduce_old<2>(A, r, lr, NumThreads, Teams);
  //case 4:
    //return reduce_old<4>(A, r, lr, NumThreads, Teams);
  //case 8:
    //return reduce_old<8>(A, r, lr, NumThreads, Teams);
  //case 16:
    //return reduce_old<16>(A, r, lr, NumThreads, Teams);
  //case 32:
    //return reduce_old<32>(A, r, lr, NumThreads, Teams);
  //case 64:
    //return reduce_old<64>(A, r, lr, NumThreads, Teams);
  //case 128:
    //return reduce_old<128>(A, r, lr, NumThreads, Teams);
  //case 256:
    //return reduce_old<256>(A, r, lr, NumThreads, Teams);
  //case 512:
    //return reduce_old<512>(A, r, lr, NumThreads, Teams);
  case 1024:
    return reduce_old<1024>(A, r, lr, NumThreads, Teams);
  //case 2048:
    //return reduce_old<2048>(A, r, lr, NumThreads, Teams);
  //case 4096:
    //return reduce_old<4096>(A, r, lr, NumThreads, Teams);
  //case 8192:
    //return reduce_old<8192>(A, r, lr, NumThreads, Teams);
  //case 16384:
    //return reduce_old<16384>(A, r, lr, NumThreads, Teams);
  //case 32768:
    //return reduce_old<32768>(A, r, lr, NumThreads, Teams);
  default:
    printf("Size %i not specialized\n", NE);
    exit(1);
  }
}

#pragma omp begin declare target device_type(nohost)
static uint32_t LeagueCounter = 0;
static char LeagueBuffer[1024 * 1024 * 4];
#pragma omp end declare target

#if 0

#define REDUCTION_TEAM_ADD_I32(RC, BS, NE)                                     \
  _Pragma("omp begin declare target device_type(nohost)")                      \
                                                                               \
  static ReductionInfo RITeamAddI32_##RC##_##BS##_##NE{                        \
      RedOp::ADD,                                                              \
      RedDataType::INT32,                                                      \
      RedWidth::LEAGUE,                                                        \
      (RedChoice)(RC | RedChoice::RED_LEAGUE_BUFFERED_DIRECT),                 \
      BS,                                                                      \
      0,                                                                       \
      NE,                                                                      \
      &LeagueCounter,                                                                       \
      &LeagueBuffer[0],                                                           \
      nullptr};                                                                \
  _Pragma("omp end declare target")                                            \
                                                                               \
      void reduce_new_##RC##_##BS##_##NE(int *A, int *r, int *lr2, int NT,      \
                                         int Teams) {                          \
    {                                                                          \
      Timer T(__PRETTY_FUNCTION__);                                            \
      _Pragma("omp target teams num_teams(Teams) thread_limit(NT)") {          \
        _Pragma("omp parallel") {                                              \
          int tid = omp_get_thread_num();                                      \
          int bid = omp_get_team_num();                                        \
          int *lr = (int*) malloc(NE * 4); \
          for (int i = 0; i < NE; ++i)                                         \
            lr[bid * NT * NE + tid * NE + i] = 0;                              \
          _Pragma("omp for") for (int t = 0; t < NT; ++t) {                    \
            for (int i = 0; i < NE; ++i) {                                     \
              lr[bid * NT * NE + tid * NE + i] += A[i];                        \
            }                                                                  \
          }                                                                    \
          __llvm_omp_tgt_reduce(nullptr, &RITeamAddI32_##RC##_##BS##_##NE,     \
                                (char *)&lr[tid * NE + bid * NT * NE],         \
                                (char *)&r[0]);                                \
        }                                                                      \
        /*asm volatile("exit;");*/                                             \
      }                                                                        \
    }                                                                          \
  }
#endif

#define REDUCTION_TEAM_ADD_I32(RC, BS, NE)                                     \
  _Pragma("omp begin declare target device_type(nohost)")                      \
                                                                               \
  static ReductionInfo RITeamAddI32_##RC##_##BS##_##NE{                        \
      RedOp::ADD,                                                              \
      RedDataType::INT32,                                                      \
      RedWidth::LEAGUE,                                                        \
      (RedChoice)(RC | RedChoice::RED_LEAGUE_BUFFERED_DIRECT),                 \
      BS,                                                                      \
      0,                                                                       \
      NE,                                                                      \
      &LeagueCounter,                                                                       \
      &LeagueBuffer[0],                                                           \
      nullptr};                                                                \
  _Pragma("omp end declare target")                                            \
                                                                               \
      void reduce_new_##RC##_##BS##_##NE(int *A, int *r, int *lr2, int NT,      \
                                         int Teams) {                          \
    {                                                                          \
      Timer T(__PRETTY_FUNCTION__);                                            \
      _Pragma("omp target teams num_teams(Teams) thread_limit(NT)") {          \
        _Pragma("omp parallel") {                                              \
          int tid = omp_get_thread_num();                                      \
          int bid = omp_get_team_num();                                        \
          int lr[NE]; \
          for (int i = 0; i < NE; ++i)                                         \
            lr[i] = 0;                              \
          _Pragma("omp for") for (int t = 0; t < NT; ++t) {                    \
            for (int i = 0; i < NE; ++i) {                                     \
              lr[i] += A[i];                        \
            }                                                                  \
          }                                                                    \
          __llvm_omp_tgt_reduce(nullptr, &RITeamAddI32_##RC##_##BS##_##NE,     \
                                (char *)&lr[0],         \
                                (char *)&r[0]);                                \
        }                                                                      \
        /*asm volatile("exit;");*/                                             \
      }                                                                        \
    }                                                                          \
  }

#if 1
REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 1, 1)
REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 1, 2)
 //REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 1, 4)
 //REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 1, 8)
 //REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 1, 16)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 1, 32)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 1, 64)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 1, 128)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 1, 256)
//REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 1, 512)
REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 1, 1024)
//REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 1, 2048)
//REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 1, 4096)
REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 1, 1)
REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 1, 2)
 //REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 1, 4)
 //REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 1, 8)
 //REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 1, 16)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 1, 32)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 1, 64)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 1, 128)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 1, 256)
//REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 1, 512)
REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 1, 1024)
//REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 1, 2048)
//REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 1, 4096)

 REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 2, 2)
 //REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 2, 4)
 //REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 2, 8)
 //REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 2, 16)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 2, 32)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 2, 64)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 2, 128)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 2, 256)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 2, 512)
 REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 2, 1024)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 2, 2048)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 2, 4096)
 REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 2, 2)
 //REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 2, 4)
 //REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 2, 8)
 //REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 2, 16)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 2, 32)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 2, 64)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 2, 128)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 2, 256)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 2, 512)
 REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 2, 1024)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 2, 2048)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 2, 4096)

 //REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 4, 4)
 //REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 4, 8)
 //REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 4, 16)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 4, 32)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 4, 64)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 4, 128)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 4, 256)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 4, 512)
 REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 4, 1024)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 4, 2048)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 4, 4096)
 //REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 4, 4)
 //REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 4, 8)
 //REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 4, 16)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 4, 32)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 4, 64)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 4, 128)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 4, 256)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 4, 512)
 REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 4, 1024)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 4, 2048)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 4, 4096)

 //REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 8, 8)
 //REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 8, 16)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 8, 32)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 8, 64)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 8, 128)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 8, 256)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 8, 512)
 REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 8, 1024)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 8, 2048)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 8, 4096)
 //REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 8, 8)
 //REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 8, 16)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 8, 32)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 8, 64)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 8, 128)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 8, 256)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 8, 512)
 REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 8, 1024)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 8, 2048)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 8, 4096)

//REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 16, 16)
//REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 16, 32)
//REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 16, 64)
//REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 16, 128)
//REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 16, 256)
//REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 16, 512)
REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 16, 1024)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 16, 2048)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 16, 4096)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 16, 8192)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 16, 16384)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_FULLY, 16, 32768)
//REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 16, 16)
//REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 16, 32)
//REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 16, 64)
//REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 16, 128)
//REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 16, 256)
//REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 16, 512)
REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 16, 1024)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 16, 2048)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 16, 4096)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 16, 8192)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 16, 16384)
// REDUCTION_TEAM_ADD_I32(RED_ITEMS_PARTIALLY, 16, 32768)
#endif

void init(unsigned *A, unsigned *rold, unsigned *rnew, unsigned *lr,
          unsigned NE, unsigned MAXNE, unsigned Teams) {
  for (unsigned i = 0; i < NE; ++i) {
    A[i] = i + 1;
    rold[i] = rnew[i] = 0 * i;
  }
  // for (unsigned i = 0; i < NE * 512*Teams; ++i) {
  // lr[i] = 0;
  //}

  for (unsigned i = NE; i < MAXNE; ++i) {
    A[i] = rold[i] = rnew[i] = -1;
  }
  // for (unsigned i = NE*512*Teams; i < MAXNE*512*Teams; ++i) {
  // lr[i] = -1;
  //}
}
void compare(int *A, int *rold, int *rnew, int NE, int MAXNE) {
  for (int i = 0; i < NE; ++i) {
    if (rold[i] != rnew[i])
      printf("Unexpected difference rold[%i] = %i vs. rnew[%i] = %i\n", i,
             rold[i], i, rnew[i]);
  }

  for (int i = NE; i < MAXNE; ++i) {
    if (A[i] != -1)
      printf("Unexpected value in suffix of A[%i] = %i\n", i, A[i]);
    if (rold[i] != -1)
      printf("Unexpected value in suffix of rold[%i] = %i\n", i, rold[i]);
    if (rnew[i] != -1)
      printf("Unexpected value in suffix of rnew[%i] = %i\n", i, rnew[i]);
  }
}

void test() {
  int NT = 512;
  int Teams = 1024;

  int MAXNE = 4096;
  int *A = (int *)malloc(sizeof(int) * MAXNE);
  int *rold = (int *)malloc(sizeof(int) * MAXNE);
  int *rnew = (int *)malloc(sizeof(int) * MAXNE);
  int *lr = (int *)malloc(sizeof(int) * MAXNE * 512 * Teams);

#define REDUCEOLD(BS, NE)                                                      \
  if (BS == 1) {                                                               \
    int N = NE;                                                                \
    init((unsigned *)A, (unsigned *)rold, (unsigned *)rnew, (unsigned *)lr,    \
         NE, MAXNE, Teams);                                                    \
    _Pragma("omp target enter data map(to : A[:N], rold[:N], rnew[:N], lr[:N*512*Teams])");                                              \
    reduce_old<NE>(A, rold, lr, NT, Teams);                                    \
    _Pragma("omp target exit data map(from : A[:N], rold[:N], rnew[:N], lr[:N*512*Teams])");                                              \
    /*reduce_host(A, rold, lr, NT, NE);*/                                      \
    /*compare(A, rold, rnew, NE, MAXNE);*/                                     \
  }

#define REDUCENEW(RC, BS, NE)                                                  \
  {                                                                            \
    int N = NE;                                                                \
    init((unsigned *)A, (unsigned *)rold, (unsigned *)rnew, (unsigned *)lr,    \
         NE, MAXNE, Teams);                                                    \
    _Pragma("omp target enter data map(to : A[:N], rold[:N], rnew[:N],lr[:N*512*Teams])");                                              \
    reduce_new_##RC##_##BS##_##NE(A, rnew, lr, NT, Teams);                     \
    _Pragma("omp target exit data map(from : A[:N], rold[:N], rnew[:N], lr[:N*512*Teams])");                                              \
    /*reduce_host(A, rold, lr, NT, NE);*/                                      \
    /*compare(A, rold, rnew, NE, MAXNE);*/                                     \
  }

#define REDUCEVERIFY(RC, BS, NE)                                               \
  {                                                                            \
    int N = NE;                                                                \
    init((unsigned *)A, (unsigned *)rold, (unsigned *)rnew, (unsigned *)lr,    \
         NE, MAXNE, Teams);                                                    \
    _Pragma("omp target enter data map(to : A[:N], rold[:N], rnew[:N],lr[:N*512*Teams])");                                              \
    reduce_old<NE>(A, rold, lr, NT, Teams);                                    \
    reduce_new_##RC##_##BS##_##NE(A, rnew, lr, NT, Teams);                     \
    _Pragma("omp target exit data map(from : A[:N], rold[:N], rnew[:N],lr[:N*512*Teams])");                                              \
    /*reduce_host(A, rold, lr, NT, NE);*/                                      \
    compare(A, rold, rnew, NE, MAXNE);                                         \
  }

#define REDUCE4(BS, NE)                                                        \
  REDUCEOLD(BS, NE);                                                           \
  REDUCENEW(RED_ITEMS_FULLY, BS, NE);                                          \
  REDUCENEW(RED_ITEMS_PARTIALLY, BS, NE);                                      \
  REDUCEOLD(BS, NE);                                                           \
  REDUCENEW(RED_ITEMS_FULLY, BS, NE);                                          \
  REDUCENEW(RED_ITEMS_PARTIALLY, BS, NE);                                      \
  REDUCENEW(RED_ITEMS_PARTIALLY, BS, NE);                                      \
  REDUCENEW(RED_ITEMS_FULLY, BS, NE);                                          \
  REDUCEOLD(BS, NE);                                                           \
  REDUCENEW(RED_ITEMS_PARTIALLY, BS, NE);                                      \
  REDUCENEW(RED_ITEMS_FULLY, BS, NE);                                          \
  REDUCEOLD(BS, NE);

#define REDUCE16(BS, NE)                                                       \
  REDUCE4(BS, NE)                                                              \
  REDUCE4(BS, NE)                                                              \
  REDUCE4(BS, NE)                                                              \
  REDUCE4(BS, NE)

#define REDUCE(BS, NE)                                                         \
  REDUCEVERIFY(RED_ITEMS_PARTIALLY, BS, NE)                                    \
  REDUCEVERIFY(RED_ITEMS_FULLY, BS, NE)                                        \
  //REDUCE16(BS, NE) \
  //REDUCE16(BS, NE) \

#if 1
  //REDUCE(1, 1)
  //REDUCE(1, 2)
  //REDUCE(1, 4)
  //REDUCE(1, 8)
  //REDUCE(1, 16)
  // REDUCE(1, 32)
  // REDUCE(1, 64)
  // REDUCE(1, 128)
  // REDUCE(1, 256)
  //REDUCE(1, 512)
  //REDUCE(1, 1024)
  // REDUCE(1, 2048)
  // REDUCE(1, 4096)
#endif
  //REDUCE(2, 2)
  //REDUCE(2, 4)
  //REDUCE(2, 8)
  //REDUCE(2, 16)
#if 0
  REDUCE(2, 32)
  REDUCE(2, 64)
  REDUCE(2, 128)
  REDUCE(2, 256)
  REDUCE(2, 512)
#endif
  //REDUCE(2, 1024)
#if 0
  REDUCE(2, 2048)
  REDUCE(2, 4096)
#endif
  //REDUCE(4, 4)
  //REDUCE(4, 8)
  //REDUCE(4, 16)
#if 0
  REDUCE(4, 32)
  REDUCE(4, 64)
  REDUCE(4, 128)
  REDUCE(4, 256)
  REDUCE(4, 512)
#endif
  REDUCE(4, 1024)
#if 0
  REDUCE(4, 2048)
  REDUCE(4, 4096)
#endif
  //REDUCE(8, 8)
  //REDUCE(8, 16)
#if 0
  REDUCE(8, 32)
  REDUCE(8, 64)
  REDUCE(8, 128)
  REDUCE(8, 256)
  REDUCE(8, 512)
#endif
  REDUCE(8, 1024)
#if 0
  REDUCE(8, 2048)
  REDUCE(8, 4096)
#endif
#if 1
  //REDUCE(16, 16)
#endif
#if 1
  // REDUCE(16, 32)
  // REDUCE(16, 64)
  // REDUCE(16, 128)
  //REDUCE(16, 512)
  REDUCE(16, 1024)

// REDUCE(16, 2048)
// REDUCE(16, 4096)
// REDUCE(16, 8192)
// REDUCE(16, 16384)
// REDUCE(16, 32768)
#endif
}

int main() {

#pragma omp target teams num_teams(1)
  {
#pragma omp parallel
    {}
  }
  test();
  return 0;
}
