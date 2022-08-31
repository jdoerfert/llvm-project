//===--- Reduction.h - OpenMP device reduction API and types ------ C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_REDUCTION_H
#define OMPTARGET_REDUCTION_H

#include "Types.h"

#ifdef __cplusplus
namespace ompx {

extern "C" {
#endif

/// TODO
enum __llvm_omp_reduction_level : uint8_t {
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

enum __llvm_omp_reduction_initial_value_kind : uint8_t {
  _VALUE_ZERO,
  _VALUE_MONE,
  _VALUE_ONE,
  _VALUE_MIN,
  _VALUE_MAX,
};

enum __llvm_omp_reduction_operation : uint8_t {
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

/// TODO
enum __llvm_omp_reduction_allocation_configuration : uint8_t {
  _PREALLOCATED = 1 << 0,
  _PRE_INITIALIZED = 1 << 1,
};

enum __llvm_omp_default_reduction_choices : uint64_t {
  /// By default we will reduce a batch of elements completely before we move on
  /// to the next batch. If the _REDUCE_WARP_FIRST bit is set we will instead
  /// first reduce all warps and then move on to reduce warp results further.
  _REDUCE_WARP_FIRST = 1 << 0,

  _REDUCE_ATOMICALLY_AFTER_WARP = 1 << 1,

  _REDUCE_TEAM_AS_PART_OF_LEAGUE = 1 << 3,

  _REDUCE_LEAGUE_VIA_ATOMICS_WITH_OFFSET = 1 << 5,

  _REDUCE_LEAGUE_VIA_SECOND_KERNEL = 1 << 15,

  _PRIVATE_BUFFER_IS_SHARED = 1 << 25,
};

/// TODO
typedef __attribute__((alloc_size(1))) void *(
    __llvm_omp_reduction_allocator_fn_ty)(size_t);

/// TODO
typedef void(__llvm_omp_reduction_reducer_fn_ty)(void *DstPtr, void *SrcPtr);

/// TODO
typedef void(__llvm_omp_reduction_initializer_fn_ty)(void *);

#define _INITIALIZERS(_TYPE, _TYPE_NAME, _ONE, _MIN, _MAX)                     \
  static void __llvm_omp_reduction_initialize_value_##_TYPE_NAME##_zero(       \
      void *__ptr) {                                                           \
    *reinterpret_cast<_TYPE *>(__ptr) = (_TYPE)(0);                            \
  };                                                                           \
  static void __llvm_omp_reduction_initialize_value_##_TYPE_NAME##_mone(       \
      void *__ptr) {                                                           \
    *reinterpret_cast<_TYPE *>(__ptr) = (_TYPE)(~0);                           \
  };                                                                           \
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

// TODO: We tried to avoid including system headers in the device runtime.
//       Rethink if we want to do that now.

_INITIALIZERS(char, int8, 1, SCHAR_MIN, SCHAR_MAX)
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
  case _VALUE_ZERO:                                                            \
    return __llvm_omp_reduction_initialize_value_##_TYPE_NAME##_zero;          \
  case _VALUE_MONE:                                                            \
    return __llvm_omp_reduction_initialize_value_##_TYPE_NAME##_mone;          \
  case _VALUE_ONE:                                                             \
    return __llvm_omp_reduction_initialize_value_##_TYPE_NAME##_one;           \
  case _VALUE_MIN:                                                             \
    return __llvm_omp_reduction_initialize_value_##_TYPE_NAME##_min;           \
  case _VALUE_MAX:                                                             \
    return __llvm_omp_reduction_initialize_value_##_TYPE_NAME##_max;           \
  default:                                                                     \
    __builtin_unreachable();                                                   \
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
  default:
    __builtin_unreachable();
  }

#undef _DISPATCH
}

struct __llvm_omp_default_reduction_league_configuration_ty {
  void *__buffer;
  int32_t __num_items;
};

/// TODO
struct __llvm_omp_default_reduction_var_configuration_ty {

  void *__restrict__ __dst_ptr;

  int32_t __item_size;
  int32_t __num_items;

  int32_t __batch_size;

  __llvm_omp_reduction_operation __op;

  __llvm_omp_reduction_element_type __element_type;

  __llvm_omp_reduction_reducer_fn_ty *__reducer_fn;
  __llvm_omp_reduction_initializer_fn_ty *__initializer_fn;
};

/// TODO
struct __llvm_omp_default_reduction_configuration_ty {

  __llvm_omp_reduction_level __level;

  __llvm_omp_reduction_allocation_configuration __alloc_config;

  __llvm_omp_default_reduction_choices __policy;

  __llvm_omp_reduction_allocator_fn_ty *__allocator_fn;

  int32_t __num_participants;

  int32_t __num_reduction_vars;

  __llvm_omp_default_reduction_var_configuration_ty *__reduction_vars[0];
};

/// TODO
struct __llvm_omp_default_reduction_private_info_ty {
  __llvm_omp_default_reduction_configuration_ty *__config;
  void *__private_default_data;
};

void __llvm_omp_default_reduction_init(
    __llvm_omp_default_reduction_private_info_ty
        *const __restrict__ __private_copy,
    __llvm_omp_default_reduction_league_configuration_ty
        *const __restrict__ __league_config);

void __llvm_omp_default_reduction_warp(
    __llvm_omp_default_reduction_private_info_ty *__restrict__ __private_info);

void __llvm_omp_default_reduction_team(
    __llvm_omp_default_reduction_private_info_ty *__restrict__ __private_info);

void __llvm_omp_default_reduction_league(
    __llvm_omp_default_reduction_private_info_ty *__restrict__ __private_info,
    __llvm_omp_default_reduction_league_configuration_ty
        *const __restrict__ __league_config);

#if 0
void __llvm_omp_default_reduction_combine(
    __llvm_omp_default_reduction_private_info_ty *__restrict__ __private_info,
    __llvm_omp_default_reduction_league_configuration_ty
        *const __restrict__ __league_config);
#endif

void __llvm_omp_default_reduction_combine_league_lvl2(
    __llvm_omp_default_reduction_private_info_ty *__restrict__ __private_info,
    __llvm_omp_default_reduction_league_configuration_ty
        *const __restrict__ __league_config);

#ifdef __cplusplus
}

} // namespace ompx
#endif

#endif
