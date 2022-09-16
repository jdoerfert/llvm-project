#include <omp.h>
#include <stdint.h>
#include <stdio.h>


/// TODO
enum __llvm_omp_reduction_level : unsigned char {
  _WARP = 1 << 0,
  _TEAM = 1 << 1,
  _LEAGUE = 1 << 2,
};

enum __llvm_omp_reduction_element_type : int8_t {
  _INT8,
  _INT16,
  _INT32,
  _INT64,
  _FLOAT,
  _DOUBLE,
  _CUSTOM_TYPE,
};

enum __llvm_omp_reduction_initial_value_kind : unsigned char {
  _VALUE_ONE,
  _VALUE_MIN,
  _VALUE_MAX,
};

enum __llvm_omp_reduction_operation : unsigned char {
  /// Uses 0 initializer
  _ADD,
  _SUB,
  _BIT_OR,
  _BIT_XOR,
  _LOGIC_OR,

  /// Uses ~0 initializer
  _BIT_AND,

  /// Uses 1 initializer
  _MUL,
  _LOGIC_AND,

  /// Usesmin/max value initializer
  _MAX,
  _MIN,

  /// Uses custom initializer function.
  _CUSTOM_OP,
};

enum __llvm_omp_default_reduction_choices : int32_t {
  /// By default we will reduce a batch of elements completely before we move on
  /// to the next batch. If the _REDUCE_WARP_FIRST bit is set we will instead
  /// first reduce all warps and then move on to reduce warp results further.
  _REDUCE_WARP_FIRST = 1 << 0,

  _REDUCE_ATOMICALLY_AFTER_WARP = 1 << 1,
  _REDUCE_ATOMICALLY_AFTER_TEAM = 1 << 2,

  _REDUCE_LEAGUE_VIA_ATOMICS_WITH_OFFSET = 1 << 3,
  _REDUCE_LEAGUE_VIA_LARGE_BUFFER = 1 << 4,
  _REDUCE_LEAGUE_VIA_SYNCHRONIZED_SMALL_BUFFER = 1 << 5,

  _PRIVATE_BUFFER_IS_SHARED = 1 << 6,
};

/// TODO
enum __llvm_omp_reduction_allocation_configuration : unsigned char {
  _PREALLOCATED_IN_PLACE = 1 << 0,
  _PRE_INITIALIZED = 1 << 1,
};

/// TODO
typedef __attribute__((alloc_size(1))) void *(
    __llvm_omp_reduction_allocator_fn_ty)(size_t);

/// TODO
typedef void(__llvm_omp_reduction_initializer_fn_ty)(void *);

struct __llvm_omp_default_reduction_configuration_ty {

  __llvm_omp_reduction_level __level;

  __llvm_omp_reduction_allocation_configuration __alloc_config;

  __llvm_omp_reduction_operation __op;

  __llvm_omp_reduction_element_type __element_type;

  __llvm_omp_default_reduction_choices __choices;

  int32_t __item_size;
  int32_t __num_items;

  int32_t __batch_size;

  int32_t __num_participants;

  void *__buffer;
  uint32_t *__counter_ptr;

  __llvm_omp_reduction_allocator_fn_ty *__allocator_fn;
  __llvm_omp_reduction_initializer_fn_ty *__initializer_fn;
};
struct __llvm_omp_default_reduction {
  __llvm_omp_default_reduction_configuration_ty *__config;
  void *__private_default_data;
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

void reduce_host(int *A, int *r, int NumThreads, int NE, int Teams) {
  {
    Timer T(__PRETTY_FUNCTION__);
    {
#pragma omp parallel for reduction(+ : r[:NE])
      for (int t = 0; t < NumThreads * Teams; ++t) {
        for (int i = 0; i < NE; ++i) {
          r[i] += A[i];
        }
      }
    }
  }
}
template <int NE>
void reduce_old(int *A, int *r, int NumThreads, int Teams) {
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
void reduce_old(int *A, int *r, int NumThreads, int NE, int Teams) {
  switch (NE) {
  case 1:
    return reduce_old<1>(A, r, NumThreads, Teams);
  case 2:
    return reduce_old<2>(A, r, NumThreads, Teams);
  //case 4:
    //return reduce_old<4>(A, r, NumThreads, Teams);
  //case 8:
    //return reduce_old<8>(A, r, NumThreads, Teams);
  //case 16:
    //return reduce_old<16>(A, r, NumThreads, Teams);
  //case 32:
    //return reduce_old<32>(A, r, NumThreads, Teams);
  //case 64:
    //return reduce_old<64>(A, r, NumThreads, Teams);
  //case 128:
    //return reduce_old<128>(A, r, NumThreads, Teams);
  //case 256:
    //return reduce_old<256>(A, r, NumThreads, Teams);
  //case 512:
    //return reduce_old<512>(A, r, NumThreads, Teams);
  case 1024:
    return reduce_old<1024>(A, r, NumThreads, Teams);
  //case 2048:
    //return reduce_old<2048>(A, r, NumThreads, Teams);
  case 4096:
    return reduce_old<4096>(A, r, NumThreads, Teams);
  //case 8192:
    //return reduce_old<8192>(A, r, NumThreads, Teams);
  //case 16384:
    //return reduce_old<16384>(A, r, NumThreads, Teams);
  //case 32768:
    //return reduce_old<32768>(A, r, NumThreads, Teams);
  default:
    printf("Size %i not specialized\n", NE);
    exit(1);
  }
}

#pragma omp begin declare target device_type(nohost)
static uint32_t LeagueCounter = 0;
static char LeagueBuffer[1024 * 4 * 1024 * 16 * 8];

extern "C" void __llvm_omp_default_reduction_init(
    __llvm_omp_default_reduction *__restrict__ __private_copy,
    __llvm_omp_default_reduction *const __restrict__ __original_copy);
extern "C" void __llvm_omp_default_reduction_combine(
    __llvm_omp_default_reduction *__restrict__ __shared_out_copy,
    __llvm_omp_default_reduction *__restrict__ __private_copy);

#pragma omp end declare target


#define REDUCTION_TEAM_ADD_I32(RC, BS, NE)                                     \
  _Pragma("omp begin declare target device_type(nohost)")                      \
                                                                               \
  static __llvm_omp_default_reduction_configuration_ty RITeamAddI32_##RC##_##BS##_##NE{                        \
_LEAGUE, _PREALLOCATED_IN_PLACE, _ADD, _INT32,  __llvm_omp_default_reduction_choices(RC), 4, NE, BS, 0, &LeagueBuffer, &LeagueCounter, nullptr, nullptr};\
\
  _Pragma("omp end declare target")                                            \
                                                                               \
      void reduce_new_##RC##_##BS##_##NE(int *A, int *r, int NT,      \
                                         int Teams) {                          \
    {                                                                          \
      Timer T(__PRETTY_FUNCTION__);                                            \
      _Pragma("omp target teams num_teams(Teams) thread_limit(NT)") {          \
        _Pragma("omp parallel") {                                              \
          int lr[NE]; \
          __llvm_omp_default_reduction sout { nullptr, r }; \
          __llvm_omp_default_reduction lodr { &RITeamAddI32_##RC##_##BS##_##NE, (void*)&lr[0] }; \
          __llvm_omp_default_reduction_init(&lodr, nullptr); \
          _Pragma("omp for") for (int t = 0; t < NT; ++t) {                    \
            for (int i = 0; i < NE; ++i) {                                     \
              lr[i] += A[i];                        \
            }                                                                  \
          }                                                                    \
          __llvm_omp_default_reduction_combine(&sout, &lodr) ; \
        }                                                                      \
        /*asm volatile("exit;");*/                                             \
      }                                                                        \
    }                                                                          \
  }

#if 1
REDUCTION_TEAM_ADD_I32(0, 1, 1)
REDUCTION_TEAM_ADD_I32(0, 1, 2)
 //REDUCTION_TEAM_ADD_I32(0, 1, 4)
 //REDUCTION_TEAM_ADD_I32(0, 1, 8)
 //REDUCTION_TEAM_ADD_I32(0, 1, 16)
// REDUCTION_TEAM_ADD_I32(0, 1, 32)
// REDUCTION_TEAM_ADD_I32(0, 1, 64)
// REDUCTION_TEAM_ADD_I32(0, 1, 128)
// REDUCTION_TEAM_ADD_I32(0, 1, 256)
//REDUCTION_TEAM_ADD_I32(0, 1, 512)
REDUCTION_TEAM_ADD_I32(0, 1, 1024)
//REDUCTION_TEAM_ADD_I32(0, 1, 2048)
REDUCTION_TEAM_ADD_I32(0, 1, 4096)
//REDUCTION_TEAM_ADD_I32(0, 1, 16384)
//REDUCTION_TEAM_ADD_I32(0, 1, 32768)
REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 1, 1)
REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 1, 2)
 //REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 1, 4)
 //REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 1, 8)
 //REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 1, 16)
// REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 1, 32)
// REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 1, 64)
// REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 1, 128)
// REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 1, 256)
//REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 1, 512)
REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 1, 1024)
//REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 1, 2048)
REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 1, 4096)
//REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 1, 16384)
//REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 1, 32768)

 REDUCTION_TEAM_ADD_I32(0, 2, 2)
 REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 2, 2)
#if 0
 //REDUCTION_TEAM_ADD_I32(0, 2, 4)
 //REDUCTION_TEAM_ADD_I32(0, 2, 8)
 //REDUCTION_TEAM_ADD_I32(0, 2, 16)
// REDUCTION_TEAM_ADD_I32(0, 2, 32)
// REDUCTION_TEAM_ADD_I32(0, 2, 64)
// REDUCTION_TEAM_ADD_I32(0, 2, 128)
// REDUCTION_TEAM_ADD_I32(0, 2, 256)
// REDUCTION_TEAM_ADD_I32(0, 2, 512)
 REDUCTION_TEAM_ADD_I32(0, 2, 1024)
// REDUCTION_TEAM_ADD_I32(0, 2, 2048)
// REDUCTION_TEAM_ADD_I32(0, 2, 4096)
 //REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 2, 4)
 //REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 2, 8)
 //REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 2, 16)
// REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 2, 32)
// REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 2, 64)
// REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 2, 128)
// REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 2, 256)
// REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 2, 512)
 REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 2, 1024)
// REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 2, 2048)
// REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 2, 4096)

 //REDUCTION_TEAM_ADD_I32(0, 4, 4)
 //REDUCTION_TEAM_ADD_I32(0, 4, 8)
 //REDUCTION_TEAM_ADD_I32(0, 4, 16)
// REDUCTION_TEAM_ADD_I32(0, 4, 32)
// REDUCTION_TEAM_ADD_I32(0, 4, 64)
// REDUCTION_TEAM_ADD_I32(0, 4, 128)
// REDUCTION_TEAM_ADD_I32(0, 4, 256)
// REDUCTION_TEAM_ADD_I32(0, 4, 512)
 REDUCTION_TEAM_ADD_I32(0, 4, 1024)
// REDUCTION_TEAM_ADD_I32(0, 4, 2048)
// REDUCTION_TEAM_ADD_I32(0, 4, 4096)
 //REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 4, 4)
 //REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 4, 8)
 //REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 4, 16)
// REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 4, 32)
// REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 4, 64)
// REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 4, 128)
// REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 4, 256)
// REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 4, 512)
 REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 4, 1024)
// REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 4, 2048)
// REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 4, 4096)

 //REDUCTION_TEAM_ADD_I32(0, 8, 8)
 //REDUCTION_TEAM_ADD_I32(0, 8, 16)
// REDUCTION_TEAM_ADD_I32(0, 8, 32)
// REDUCTION_TEAM_ADD_I32(0, 8, 64)
// REDUCTION_TEAM_ADD_I32(0, 8, 128)
// REDUCTION_TEAM_ADD_I32(0, 8, 256)
// REDUCTION_TEAM_ADD_I32(0, 8, 512)
 REDUCTION_TEAM_ADD_I32(0, 8, 1024)
// REDUCTION_TEAM_ADD_I32(0, 8, 2048)
// REDUCTION_TEAM_ADD_I32(0, 8, 4096)
 //REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 8, 8)
 //REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 8, 16)
// REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 8, 32)
// REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 8, 64)
// REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 8, 128)
// REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 8, 256)
// REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 8, 512)
 REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 8, 1024)
// REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 8, 2048)
// REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 8, 4096)

//REDUCTION_TEAM_ADD_I32(0, 16, 16)
//REDUCTION_TEAM_ADD_I32(0, 16, 32)
//REDUCTION_TEAM_ADD_I32(0, 16, 64)
//REDUCTION_TEAM_ADD_I32(0, 16, 128)
//REDUCTION_TEAM_ADD_I32(0, 16, 256)
//REDUCTION_TEAM_ADD_I32(0, 16, 512)
#endif
  #if 1
REDUCTION_TEAM_ADD_I32(0, 16, 1024)
// REDUCTION_TEAM_ADD_I32(0, 16, 2048)
 REDUCTION_TEAM_ADD_I32(0, 16, 4096)
// REDUCTION_TEAM_ADD_I32(0, 16, 8192)
 //REDUCTION_TEAM_ADD_I32(0, 16, 16384)
 //REDUCTION_TEAM_ADD_I32(0, 16, 32768)
//REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 16, 16)
//REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 16, 32)
//REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 16, 64)
//REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 16, 128)
//REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 16, 256)
//REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 16, 512)
REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 16, 1024)
// REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 16, 2048)
 REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 16, 4096)
// REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 16, 8192)
 //REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 16, 16384)
 //REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 16, 32768)

 //REDUCTION_TEAM_ADD_I32(0, 32, 4096)
 //REDUCTION_TEAM_ADD_I32(_REDUCE_WARP_FIRST, 32, 4096)
#endif
#endif

void init(unsigned *A, unsigned *rold, unsigned *rnew,
          unsigned NE, unsigned MAXNE, unsigned Teams) {
  for (unsigned i = 0; i < NE; ++i) {
    A[i] = i + 1;
    rold[i] = rnew[i] = 0 * i;
  }

  for (unsigned i = NE; i < MAXNE; ++i) {
    A[i] = rold[i] = rnew[i] = -1;
  }
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

void test(int argc, char **argv) {
  int NT = 512;
  int Teams = argc > 1 ? atoi(argv[1]) : 1024;

  int MAXNE = 32768;
  int *A = (int *)malloc(sizeof(int) * MAXNE);
  int *rold = (int *)malloc(sizeof(int) * MAXNE);
  int *rnew = (int *)malloc(sizeof(int) * MAXNE);

#define REDUCEOLD(BS, NE)                                                      \
  if (BS == 1) {                                                               \
    int N = NE;                                                                \
    init((unsigned *)A, (unsigned *)rold, (unsigned *)rnew,    \
         NE, MAXNE, Teams);                                                    \
    _Pragma("omp target enter data map(to : A[:N], rold[:N])");                                              \
    reduce_old<NE>(A, rold, NT, Teams);                                    \
    _Pragma("omp target exit data map(from : A[:N], rold[:N])");                                              \
    /*reduce_host(A, rold, NT, NE);*/                                      \
    /*compare(A, rold, rnew, NE, MAXNE);*/                                     \
  }

#define REDUCENEW(RC, BS, NE)                                                  \
  {                                                                            \
    int N = NE;                                                                \
    init((unsigned *)A, (unsigned *)rold, (unsigned *)rnew,    \
         NE, MAXNE, Teams);                                                    \
    _Pragma("omp target enter data map(to : A[:N],  rnew[:N])");                                              \
    reduce_new_##RC##_##BS##_##NE(A, rnew, NT, Teams);                     \
    _Pragma("omp target exit data map(from : A[:N],rnew[:N])");                                              \
    /*reduce_host(A, rold, NT, NE);*/                                      \
    /*compare(A, rold, rnew, NE, MAXNE);*/                                     \
  }

#define REDUCEVERIFY(RC, BS, NE)                                               \
  {                                                                            \
    int N = NE;                                                                \
    init((unsigned *)A, (unsigned *)rold, (unsigned *)rnew,    \
         NE, MAXNE, Teams);                                                    \
    _Pragma("omp target enter data map(to : A[:N], rold[:N], rnew[:N])");                                              \
    reduce_new_##RC##_##BS##_##NE(A, rnew, NT, Teams);                     \
    reduce_old<NE>(A, rold, NT, Teams);                                    \
    _Pragma("omp target exit data map(from : A[:N], rold[:N], rnew[:N])");                                              \
    /*reduce_host(A, rold, NT, NE);*/                                      \
    compare(A, rold, rnew, NE, MAXNE);                                         \
  }

#define REDUCE4(BS, NE)                                                        \
  REDUCEOLD(BS, NE);                                                           \
  REDUCENEW(0, BS, NE);                                          \
  REDUCENEW(_REDUCE_WARP_FIRST, BS, NE);                                      \
  REDUCEOLD(BS, NE);                                                           \
  REDUCENEW(0, BS, NE);                                          \
  REDUCENEW(_REDUCE_WARP_FIRST, BS, NE);                                      \
  REDUCENEW(_REDUCE_WARP_FIRST, BS, NE);                                      \
  REDUCENEW(0, BS, NE);                                          \
  REDUCEOLD(BS, NE);                                                           \
  REDUCENEW(_REDUCE_WARP_FIRST, BS, NE);                                      \
  REDUCENEW(0, BS, NE);                                          \
  REDUCEOLD(BS, NE);

#define REDUCE16(BS, NE)                                                       \
  REDUCE4(BS, NE)                                                              \
  REDUCE4(BS, NE)                                                              \
  REDUCE4(BS, NE)                                                              \
  REDUCE4(BS, NE)

#define REDUCE(BS, NE)                                                         \
  REDUCE16(BS, NE) \
  REDUCE16(BS, NE) \
  //REDUCEVERIFY(_REDUCE_WARP_FIRST, BS, NE)                                    \
  //REDUCEVERIFY(0, BS, NE)                                        \

#if 1
  REDUCE(1, 1)
  REDUCE(1, 2)
  //REDUCE(1, 4)
  //REDUCE(1, 8)
  //REDUCE(1, 16)
  // REDUCE(1, 32)
  // REDUCE(1, 64)
  // REDUCE(1, 128)
  // REDUCE(1, 256)
  //REDUCE(1, 512)
  REDUCE(1, 1024)
  // REDUCE(1, 2048)
   REDUCE(1, 4096)
   //REDUCE(1, 16384)
   //REDUCE(1, 32768)
#endif
  REDUCE(2, 2)
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
  //REDUCE(4, 1024)
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
  //REDUCE(8, 1024)
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
 REDUCE(16, 4096)
// REDUCE(16, 8192)
 //REDUCE(16, 16384)
 //REDUCE(16, 32768)
#endif
 //REDUCE(32, 4096)
}

int main(int argc, char **argv) {

#pragma omp target teams num_teams(1)
  {
#pragma omp parallel
    {}
  }
  test(argc, argv);
  return 0;
}
