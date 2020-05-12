//===---------- private.h - Target independent OpenMP target RTL ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Private function declarations and helper macros for debugging output.
//
//===----------------------------------------------------------------------===//

#ifndef _OMPTARGET_PRIVATE_H
#define _OMPTARGET_PRIVATE_H

#include <omptarget.h>

#include <cstdint>

extern int target_data_begin(DeviceTy &Device, int32_t arg_num,
                             void **args_base, void **args, int64_t *arg_sizes,
                             int64_t *arg_types,
                             __tgt_async_info *async_info_ptr);

extern int target_data_end(DeviceTy &Device, int32_t arg_num, void **args_base,
                           void **args, int64_t *arg_sizes, int64_t *arg_types,
                           __tgt_async_info *async_info_ptr);

extern int target_data_update(DeviceTy &Device, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types);

extern int target(int64_t device_id, void *host_ptr, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types,
    int32_t team_num, int32_t thread_limit, int IsTeamConstruct);

extern int CheckDeviceAndCtors(int64_t device_id);

// enum for OMP_TARGET_OFFLOAD; keep in sync with kmp.h definition
enum kmp_target_offload_kind {
  tgt_disabled = 0,
  tgt_default = 1,
  tgt_mandatory = 2
};
typedef enum kmp_target_offload_kind kmp_target_offload_kind_t;
extern kmp_target_offload_kind_t TargetOffloadPolicy;

// This structure stores information of a mapped memory region.
struct MapComponentInfoTy {
  void *Base;
  void *Begin;
  int64_t Size;
  int64_t Type;
  MapComponentInfoTy() = default;
  MapComponentInfoTy(void *Base, void *Begin, int64_t Size, int64_t Type)
      : Base(Base), Begin(Begin), Size(Size), Type(Type) {}
};

// This structure stores all components of a user-defined mapper. The number of
// components are dynamically decided, so we utilize C++ STL vector
// implementation here.
struct MapperComponentsTy {
  std::vector<MapComponentInfoTy> Components;
};

////////////////////////////////////////////////////////////////////////////////
// implementation for fatal messages
////////////////////////////////////////////////////////////////////////////////

#define FATAL_MESSAGE0(_num, _str)                                    \
  do {                                                                \
    fprintf(stderr, "Libomptarget fatal error %d: %s\n", _num, _str); \
    exit(1);                                                          \
  } while (0)

#define FATAL_MESSAGE(_num, _str, ...)                              \
  do {                                                              \
    fprintf(stderr, "Libomptarget fatal error %d:" _str "\n", _num, \
            __VA_ARGS__);                                           \
    exit(1);                                                        \
  } while (0)

// Implemented in libomp, they are called from within __tgt_* functions.
#ifdef __cplusplus
extern "C" {
#endif

/// OMG THE DUPLICATION IN THIS WEB OF LIBRARIES IS RIDICULES AND THE
/// DECLARATIONS ARE INCOMPATIBLE SO YOU CANNOT REUSE ANYTHING EVEN IF YOU WANTED
/// TO ...
///
///{
/*!
 * The ident structure that describes a source location.
 * The struct is identical to the one in the kmp.h file.
 * We maintain the same data structure for compatibility.
 */
typedef int kmp_int32;
typedef intptr_t kmp_intptr_t;

typedef struct ident {
  kmp_int32 reserved_1; /**<  might be used in Fortran; see above  */
  kmp_int32 flags; /**<  also f.flags; KMP_IDENT_xxx flags; KMP_IDENT_KMPC
                      identifies this union member  */
  kmp_int32 reserved_2; /**<  not really used in Fortran any more; see above */
  kmp_int32 reserved_3; /**<  source[4] in Fortran, do not use for C++  */
  char const *psource; /**<  String describing the source location.
                       The string is composed of semi-colon separated fields
                       which describe the source file, the function and a pair
                       of line numbers that delimit the construct. */
} ident_t;
// Compiler sends us this info:
typedef struct kmp_depend_info {
  kmp_intptr_t base_addr;
  size_t len;
  struct {
    bool in : 1;
    bool out : 1;
    bool mtx : 1;
  } flags;
} kmp_depend_info_t;
///}

// functions that extract info from libomp; keep in sync
int omp_get_default_device(void) __attribute__((weak));
int32_t __kmpc_omp_taskwait(void *loc_ref, int32_t gtid) __attribute__((weak));
int32_t __kmpc_global_thread_num(void *) __attribute__((weak));
int __kmpc_get_target_offload(void) __attribute__((weak));
void __kmpc_omp_wait_deps(ident_t *loc_ref, kmp_int32 gtid, kmp_int32 ndeps,
                          kmp_depend_info_t *dep_list, kmp_int32 ndeps_noalias,
                          kmp_depend_info_t *noalias_dep_list)
    __attribute__((weak));
#ifdef __cplusplus
}
#endif

#ifdef OMPTARGET_DEBUG
extern int DebugLevel;

#define DP(...) \
  do { \
    if (DebugLevel > 0) { \
      DEBUGP("Libomptarget", __VA_ARGS__); \
    } \
  } while (false)
#else // OMPTARGET_DEBUG
#define DP(...) {}
#endif // OMPTARGET_DEBUG

#endif
