
/// Always included.
/// clang/lib/Header/.../__omp_tgt_reduction.h

#ifdef __cplusplus
extern "C" {
#endif

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

#define _INITIALIZERS(_TYPE, _TYPE_NAME, _ONE, _MIN, _MAX)                     \
  static void __llvm_omp_reduction_initialize_value_##_TYPE_NAME##_one(        \
      void *__ptr) {                                                           \
    *reinterpret_cast<_TYPE *>(__ptr) = _ONE;                                  \
  };                                                                           \
  static void __llvm_omp_reduction_initialize_value_##_TYPE_NAME##_min(        \
      void *__ptr) {                                                           \
    *reinterpret_cast<_TYPE *>(__ptr) = _MIN;                                  \
  };                                                                           \
  static void __llvm_omp_reduction_initialize_value_##_TYPE_NAME##_max(        \
      void *__ptr) {                                                           \
    *reinterpret_cast<_TYPE *>(__ptr) = _MAX;                                  \
  };

_INITIALIZERS(char, int8, 1, CHAR_MIN, CHAR_MAX)
_INITIALIZERS(short, int16, 1, SHRT_MIN, SHRT_MAX)
_INITIALIZERS(int, int32, 1, INT_MIN, INT_MAX)
_INITIALIZERS(long, int64, 1, LONG_MIN, LONG_MAX)
_INITIALIZERS(float, float, 1.f, FLT_MIN, FLT_MAX)
_INITIALIZERS(double, double, 1., DBL_MIN, DBL_MAX)

#undef _INITIALIZERS

static __llvm_omp_reduction_initializer_fn_ty *
__llvm_omp_reduction_get_initializer_fn(
    __llvm_omp_reduction_initial_value_kind _VK,
    __llvm_omp_reduction_element_type _ET) {
#define _DISPATCH(_TYPE_NAME)                                                  \
  switch (_VK) {                                                               \
  case _VALUE_ONE:                                                             \
    return __llvm_omp_reduction_initialize_value_##_TYPE_NAME##_one;           \
  case _VALUE_MIN:                                                             \
    return __llvm_omp_reduction_initialize_value_##_TYPE_NAME##_min;           \
  case _VALUE_MAX:                                                             \
    return __llvm_omp_reduction_initialize_value_##_TYPE_NAME##_max;           \
  };

  switch (_ET) {
  case _INT8:
    _DISPATCH(int8)
  case _INT16:
    _DISPATCH(int16)
  case _INT32:
    _DISPATCH(int32)
  case _INT64:
    _DISPATCH(int64)
  case _FLOAT:
    _DISPATCH(float)
  case _DOUBLE:
    _DISPATCH(double)
  case _CUSTOM_TYPE:
    __builtin_unreachable();
  }

#undef _DISPATCH
}

/// TODO
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

/// TODO
struct __llvm_omp_default_reduction {
  __llvm_omp_default_reduction_configuration_ty *__config;
  void *__private_default_data;
};

void __llvm_omp_default_reduction_init(
    __llvm_omp_default_reduction *__restrict__ __private_copy,
    __llvm_omp_default_reduction *const __restrict__ __original_copy);

void __llvm_omp_default_reduction_combine_league(
    __llvm_omp_default_reduction *__restrict__ __shared_out_copy,
    __llvm_omp_default_reduction *__restrict__ __private_copy);

void __llvm_omp_default_reduction_combine_team(
    __llvm_omp_default_reduction *__restrict__ __shared_out_copy,
    __llvm_omp_default_reduction *__restrict__ __private_copy);

void __llvm_omp_default_reduction_combine_warp(
    __llvm_omp_default_reduction *__restrict__ __shared_out_copy,
    __llvm_omp_default_reduction *__restrict__ __private_copy);

#ifdef __cplusplus
}
#endif

/// DeviceRTL/src/Reduction.cpp


/// Code generated by Clang

#if 0
// Team reduction
void kernel_team_reduction() {
  // TODO
}

// League reduction
// Type1 value1;
// Type2 value2
// #pragma omp teams ... reduction(<op1>: value1, <op2>: value2)
// { /* see below */ }

void kernel_league_reduction(Type1 *value1Ptr, Type2 *value2Ptr) {
  __llvm_omp_default_reduction shared_out_1 = {..., value1Ptr, ...};
  __llvm_omp_default_reduction shared_out_2 = {..., value2Ptr, ...};

  struct __llvm_omp_specific_reduction_1 {
    __llvm_omp_default_reduction __default_reduction_info = {...};
    Type1 privateCopy1;
  } red1;
  struct __llvm_omp_specific_reduction_2 {
    __llvm_omp_default_reduction __default_reduction_info = {...};
    Type2 privateCopy2;
  } red2;

  // Now we call init for all of them.
  __llvm_omp_default_reduction_init(red1.__default_reduction_info,
                                    /* unsued */ &shared_out_1);
  __llvm_omp_default_reduction_init(red2.__default_reduction_info,
                                    /* unsued */ &shared_out_2);

  // User code, state machine, all the stuff. Use red1.privateCopy1, and
  // red2.privateCopy2 instead of the original variables.

  // Ensure all threads execute, so we put a call before the final return.
  __llvm_omp_default_reduction_combine(&shared_out_1,
                                       red1.__default_reduction_info);
  __llvm_omp_default_reduction_combine(&shared_out_2,
                                       red2.__default_reduction_info);
}
#endif
