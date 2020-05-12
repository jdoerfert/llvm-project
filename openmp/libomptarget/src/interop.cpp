//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// TODO: The buildsystem seems to not include the `omp.h` of the current build
/// but the system `omp.h` :( Also, omp.h has obviously conflicting declarations
/// with other headers, because, why not.
//#include "omp.h"

#include "device.h"
#include "omptargetplugin.h"
#include "private.h"
#include "rtl.h"
//#include "../deviceRTLs/interface.h"

#include <cassert>

/// Content that belongs in omp.h
///{
/* OpenMP 5.1 Interop API */
typedef enum omp_interop_property_t {
  omp_interop_type = -1,
  omp_interop_type_name = -2,
  omp_interop_vendor = -3,
  omp_interop_vendor_name = -4,
  omp_interop_device_id = -5,
  omp_interop_device = -6,
  omp_interop_device_context = -7,
  omp_interop_tasksync = -8,
  omp_interop_first = -8
} omp_interop_property_t;

typedef enum omp_interop_err_t {
  omp_interop_err_noval = 1,
  omp_interop_err_success = 0,
  omp_interop_err_none = -1,
  omp_interop_err_range = -2,
  omp_interop_err_result_is_int = -3,
  omp_interop_err_result_is_str = -4,
  omp_interop_err_result_is_ptr = -5,
  omp_interop_err_no_interop = -6,
  omp_interop_err_failure_str = -7,
  omp_interop_err_wrong_interop_type = -8,
  omp_interop_err_unknown = -9,
} omp_interop_err_t;

typedef struct kmp_tasking_flags { /* Total struct must be exactly 32 bits */
  /* Compiler flags */ /* Total compiler flags must be 16 bits */
  unsigned tiedness : 1; /* task is either tied (1) or untied (0) */
  unsigned final : 1; /* task is final(1) so execute immediately */
  unsigned merged_if0 : 1; // no __kmpc_task_{begin/complete}_if0 calls in if0
  unsigned destructors_thunk : 1; // set if the compiler creates a thunk to
  unsigned proxy : 1; // task is a proxy task (it will be executed outside the
  unsigned priority_specified : 1; // set if the compiler provides priority
  unsigned detachable : 1; // 1 == can detach */
  unsigned unshackled : 1; /* 1 == unshackled task */
  unsigned target : 1; /* 1 == target task */
  unsigned reserved : 7; /* reserved for compiler use */
  unsigned tasktype : 1; /* task is either explicit(1) or implicit (0) */
  unsigned task_serial : 1; // task is executed immediately (1) or deferred (0)
  unsigned tasking_ser : 1; // all tasks in team are either executed immediately
  unsigned team_serial : 1; // entire team is serial (1) [1 thread] or parallel
  unsigned started : 1; /* 1==started, 0==not started     */
  unsigned executing : 1; /* 1==executing, 0==not executing */
  unsigned complete : 1; /* 1==complete, 0==not complete   */
  unsigned freed : 1; /* 1==freed, 0==allocated        */
  unsigned native : 1; /* 1==gcc-compiled task, 0==intel */
  unsigned reserved31 : 7; /* reserved for library use */

} kmp_tasking_flags_t;


typedef enum omp_interop_backend_type_t {
  // reserve 0
  omp_interop_backend_type_cuda_1 = 1,
} omp_interop_backend_type_t;

typedef struct omp_interop_val_t *omp_interop_t;
constexpr omp_interop_t omp_interop_none = NULL;

///}

// OMP Interop API
typedef enum kmp_interop_type_t {
  kmp_interop_type_unknown = -1,
  kmp_interop_type_platform,
  kmp_interop_type_device,
  kmp_interop_type_tasksync,
} kmp_interop_type_t;

extern "C" {

/// The interop value type, aka. the interop object.
typedef struct omp_interop_val_t {
  /// Device and interop-type are determined at construction time and fix.
  omp_interop_val_t(intptr_t device_id, kmp_interop_type_t interop_type)
      : interop_type(interop_type), device_id(device_id) {
    assert(interop_type != kmp_interop_type_unknown);
    assert(!async_info);
    assert(!device_info.Device);
    assert(!device_info.Context);
  }
  const char *err_str = nullptr;
  __tgt_async_info *async_info = nullptr;
  __tgt_device_info device_info;
  const kmp_interop_type_t interop_type;
  const intptr_t device_id;
  const intptr_t vendor_id = /* LLVM? */ 1;
  const intptr_t backend_type_id = omp_interop_backend_type_cuda_1;
} omp_interop_val_t;
}

static omp_interop_err_t
__kmpc_interop_get_property_err_type(omp_interop_property_t property) {
  switch (property) {
  case omp_interop_type:
    return omp_interop_err_result_is_int;
  case omp_interop_type_name:
    return omp_interop_err_result_is_str;
  case omp_interop_vendor:
    return omp_interop_err_result_is_int;
  case omp_interop_vendor_name:
    return omp_interop_err_result_is_str;
  case omp_interop_device_id:
    return omp_interop_err_result_is_int;
  case omp_interop_device:
    return omp_interop_err_result_is_ptr;
  case omp_interop_device_context:
    return omp_interop_err_result_is_ptr;
  case omp_interop_tasksync:
    return omp_interop_err_result_is_ptr;
  };
  return omp_interop_err_unknown;
}

static void __kmpc_interop_type_mismatch(omp_interop_property_t property,
                                         int *err) {
  if (err)
    *err = __kmpc_interop_get_property_err_type(property);
}

static const char *__kmpc_interop_vendor_id_to_str(intptr_t vendor_id) {
  return "CUDA_driver";
}

template <typename PropertyTy>
static PropertyTy __kmpc_interop_get_property(omp_interop_val_t &interop_val,
                                              omp_interop_property_t property,
                                              int *err);

template <>
intptr_t __kmpc_interop_get_property<intptr_t>(omp_interop_val_t &interop_val,
                                               omp_interop_property_t property,
                                               int *err) {
  switch (property) {
  case omp_interop_type:
    return interop_val.backend_type_id;
  case omp_interop_vendor:
    return interop_val.vendor_id;
  case omp_interop_device_id:
    return interop_val.device_id;
  default:
    ;
    //assert(__kmpc_interop_get_property_err_type(property) !=
               //omp_interop_err_result_is_int &&
           //"Integer property not handled in switch");
  }
  __kmpc_interop_type_mismatch(property, err);
  return 0;
}

template <>
const char *__kmpc_interop_get_property<const char *>(
    omp_interop_val_t &interop_val, omp_interop_property_t property, int *err) {
  switch (property) {
  case omp_interop_type_name:
    return interop_val.interop_type == kmp_interop_type_tasksync ? "tasksync"
                                                                 : "device+context";
  case omp_interop_vendor_name:
    return __kmpc_interop_vendor_id_to_str(interop_val.vendor_id);
  default:
    ;
    //assert(__kmpc_interop_get_property_err_type(property) !=
               //omp_interop_err_result_is_str &&
           //"String property not handled in switch");
  }
  __kmpc_interop_type_mismatch(property, err);
  return nullptr;
}

template <>
void *__kmpc_interop_get_property<void *>(omp_interop_val_t &interop_val,
                                          omp_interop_property_t property,
                                          int *err) {
  switch (property) {
  case omp_interop_device:
    if (interop_val.device_info.Device)
      return interop_val.device_info.Device;
    *err = omp_interop_err_failure_str;
    return const_cast<char *>(interop_val.err_str);
  case omp_interop_device_context:
    return interop_val.device_info.Context;
  case omp_interop_tasksync:
    return interop_val.async_info->Queue;
  default:
    ;
    //assert(__kmpc_interop_get_property_err_type(property) !=
               //omp_interop_err_result_is_str &&
           //"Pointer property not handled in switch");
  }
  __kmpc_interop_type_mismatch(property, err);
  return nullptr;
}

static bool __kmpc_interop_get_property_check(omp_interop_val_t **interop_ptr,
                                              omp_interop_property_t property,
                                              int *err) {
  if (err)
    *err = omp_interop_err_success;
  if (!interop_ptr) {
    if (err)
      *err = omp_interop_err_no_interop;
    return false;
  }
  if (property >= 0 || property < omp_interop_first) {
    if (err)
      *err = omp_interop_err_range;
    return false;
  }
  if (property == omp_interop_tasksync &&
      (*interop_ptr)->interop_type != kmp_interop_type_tasksync) {
    if (err)
      *err = omp_interop_err_wrong_interop_type;
    return false;
  }
  if ((property == omp_interop_device || property == omp_interop_device_context) &&
      (*interop_ptr)->interop_type == kmp_interop_type_tasksync) {
    if (err)
      *err = omp_interop_err_wrong_interop_type;
    return false;
  }
  return true;
}

#define __OMP_GET_INTEROP_TY(RETURN_TYPE, SUFFIX)                              \
  EXTERN RETURN_TYPE omp_get_interop_##SUFFIX(omp_interop_val_t **interop_ptr, \
                                              omp_interop_property_t property, \
                                              int *err) {                      \
    if (!__kmpc_interop_get_property_check(interop_ptr, property, err))        \
      return (RETURN_TYPE)(0);                                                 \
    return __kmpc_interop_get_property<RETURN_TYPE>(**interop_ptr, property,   \
                                                    err);                      \
  }
__OMP_GET_INTEROP_TY(intptr_t, int)
__OMP_GET_INTEROP_TY(void *, ptr)
__OMP_GET_INTEROP_TY(const char *, str)
#undef __OMP_GET_INTEROP_TY


typedef int64_t kmp_int64;

typedef kmp_int32 (*kmp_routine_entry_t)(kmp_int32, void *);

int waitForDeps(DeviceTy &Device, __tgt_async_info *AsyncInfo);
int recordEvent(DeviceTy &Device, __tgt_async_info *AsyncInfo);
int queryAndWait(DeviceTy &Device, __tgt_async_info *AsyncInfo);

extern "C" {
int __kmpc_set_async_info(void *async_info);
}

typedef union kmp_cmplrdata {
  kmp_int32 priority; /**< priority specified by user for the task */
  kmp_routine_entry_t
      destructors; /* pointer to function to invoke deconstructors of
                      firstprivate C++ objects */
  /* future data */
} kmp_cmplrdata_t;
typedef struct kmp_task { /* GEH: Shouldn't this be aligned somehow? */
  void *shareds; /**< pointer to block of pointers to shared vars   */
  kmp_routine_entry_t
      routine; /**< pointer to routine to call for executing task */
  kmp_int32 part_id; /**< part id for the task                          */
  kmp_cmplrdata_t
      data1; /* Two known optional additions: destructors and priority */
  kmp_cmplrdata_t data2; /* Process destructors first, priority second */
  /* future data */
  /*  private vars  */
} kmp_task_t;

kmp_int32 __fake_task_destroy(kmp_int32 i, void *t) {
    //printf("Fake DESTORY %i %p\n", i, t);
    kmp_task_t *task = (kmp_task_t*)t;
    DeviceTy &Device = *(((DeviceTy**)task->shareds)[0]);
    __tgt_async_info *AsyncInfo = ((__tgt_async_info**)task->shareds)[1];


    // Attach the async info to current task such that all dependent tasks can
    // start wait for the event if there is any dependency
    bool HasDependency = __kmpc_set_async_info(AsyncInfo);
    (void)HasDependency;
    //printf("HasDeps: %i\n", HasDependency);

    int Ret = 0;
    Ret = recordEvent(Device, AsyncInfo);
    //printf("Ret: %i\n", Ret);
    assert(Ret == OFFLOAD_SUCCESS);

    Ret = queryAndWait(Device, AsyncInfo);
    //printf("Ret: %i\n", Ret);
    assert(Ret == OFFLOAD_SUCCESS);
    return Ret;
}

kmp_int32 __fake_task_use(kmp_int32 i, void *t) {
    //printf("Fake USE %i %p\n", i, t);
    kmp_task_t *task = (kmp_task_t*)t;
    DeviceTy &Device = *(((DeviceTy**)task->shareds)[0]);
    __tgt_async_info *AsyncInfo = ((__tgt_async_info**)task->shareds)[1];

    int Ret = 0;
    Ret = waitForDeps(Device, AsyncInfo);
    //printf("Ret: %i %p\n", Ret, &Device);
    assert(Ret == OFFLOAD_SUCCESS);

    __fake_task_destroy(i, t);
    return Ret;
}


extern "C" {
kmp_task_t *__kmpc_omp_task_alloc(ident_t *loc_ref, kmp_int32 gtid,
                                         kmp_int32 flags,
                                         size_t sizeof_kmp_task_t,
                                         size_t sizeof_shareds,
                                         kmp_routine_entry_t task_entry,
                                         kmp_int64 device_id);
kmp_int32 __kmpc_omp_task_with_deps(ident_t *loc_ref, kmp_int32 gtid,
                                    kmp_task_t *new_task, kmp_int32 ndeps,
                                    kmp_depend_info_t *dep_list,
                                    kmp_int32 ndeps_noalias,
                                    kmp_depend_info_t *noalias_dep_list);
}

EXTERN void __kmpc_interop_init(ident_t *loc_ref, kmp_int32 gtid,
                         omp_interop_val_t **interop_ptr,
                         kmp_interop_type_t interop_type, kmp_int32 device_id,
                         kmp_int32 ndeps, kmp_depend_info_t *dep_list,
                         kmp_int32 ndeps_noalias,
                         kmp_depend_info_t *noalias_dep_list) {
  assert(interop_ptr && "Cannot initialize nullptr!");
  assert(interop_type != kmp_interop_type_unknown &&
         "Cannot initialize with unknown interop_type!");
  if (device_id == -1)
    device_id = omp_get_default_device();

  *interop_ptr = new omp_interop_val_t(device_id, interop_type);

  if (device_id == omp_get_initial_device()) {
    // TODO?
    return;
  }

  if (!device_is_ready(device_id)) {
    (*interop_ptr)->err_str = "Device not ready!";
    return;
  }

  DeviceTy &Device = Devices[device_id];
  //printf("%i\n", (!Device.RTL || !Device.RTL->init_async_info));

  if (interop_type == kmp_interop_type_tasksync) {
#ifdef SYNC_INTEROP
    __kmpc_omp_wait_deps(loc_ref, gtid, ndeps, dep_list, ndeps_noalias,
                         noalias_dep_list);
#else
#endif
  }

  if (interop_type == kmp_interop_type_tasksync) {
    if (!Device.RTL || !Device.RTL->init_async_info ||
        Device.RTL->init_async_info(device_id, &(*interop_ptr)->async_info)) {
      // error
      delete *interop_ptr;
      *interop_ptr = omp_interop_none;
    }
  } else {
    if (!Device.RTL || !Device.RTL->init_device_info ||
        Device.RTL->init_device_info(device_id, &(*interop_ptr)->device_info,
                                     &(*interop_ptr)->err_str)) {
      // error
      delete *interop_ptr;
      *interop_ptr = omp_interop_none;
    }
  }

  (*interop_ptr)->async_info->DeviceID = device_id;
}

EXTERN void __kmpc_interop_use(ident_t *loc_ref, kmp_int32 gtid,
                               omp_interop_val_t **interop_ptr,
                               kmp_interop_type_t interop_type,
                               kmp_int32 device_id, kmp_int32 ndeps,
                               kmp_depend_info_t *dep_list,
                               kmp_int32 ndeps_noalias,
                               kmp_depend_info_t *noalias_dep_list) {
  assert(interop_ptr && "Cannot use nullptr!");
  omp_interop_val_t *interop_val = *interop_ptr;
  assert(interop_val != omp_interop_none &&
         "Cannot use uninitialized interop_ptr!");
  assert((interop_type == kmp_interop_type_unknown ||
          interop_val->interop_type == interop_type) &&
         "Inconsistent interop_ptr-type usage!");
  assert((device_id == -1 || interop_val->device_id == device_id) &&
         "Inconsistent device-id usage!");

  if (interop_val->interop_type == kmp_interop_type_tasksync) {
#ifdef SYNC_INTEROP
    __kmpc_omp_wait_deps(loc_ref, gtid, ndeps, dep_list, ndeps_noalias,
                         noalias_dep_list);
#else
    kmp_tasking_flags flags;
    //flags.tiedness = 1;
    //flags.proxy = 1;
    //flags.task_serial = 1;
    //flags.reserved = 42;
    flags.target = 1;
    flags.unshackled = 1;
    kmp_int32 flags32 = *((kmp_int32*)&flags);

    DeviceTy &Device = Devices[device_id];
    kmp_task_t *task = __kmpc_omp_task_alloc(loc_ref, gtid, flags32,
                                            40,
                                            16,
                                            &__fake_task_use,
                                            device_id);
    //printf("Dev %p: T: %p\n", &Device, task);
    ((DeviceTy**)task->shareds)[0] = &Device;
    ((__tgt_async_info**)task->shareds)[1] = interop_val->async_info;
    assert(((DeviceTy**)task->shareds)[0] == &Device);

    int r = __kmpc_omp_task_with_deps(loc_ref, gtid,
                                    task, ndeps,
                                    dep_list,
                                    ndeps_noalias,
                                    noalias_dep_list);
    //printf("Done with fake task: %i \n", r);
#endif
  }
}

EXTERN void __kmpc_interop_destroy(ident_t *loc_ref, kmp_int32 gtid,
                            omp_interop_val_t **interop_ptr,
                            kmp_interop_type_t interop_type,
                            kmp_int32 device_id, kmp_int32 ndeps,
                            kmp_depend_info_t *dep_list,
                            kmp_int32 ndeps_noalias,
                            kmp_depend_info_t *noalias_dep_list) {
  assert(interop_ptr && "Cannot use nullptr!");
  omp_interop_val_t *interop_val = *interop_ptr;
  // Gracefully handle the destruction of none objects, I guess.
  if (interop_val == omp_interop_none)
    return;

  assert((interop_type == kmp_interop_type_unknown ||
          interop_val->interop_type == interop_type) &&
         "Inconsistent interop_ptr-type usage!");
  assert((device_id == -1 || interop_val->device_id == device_id) &&
         "Inconsistent device-id usage!");

  if (interop_val->interop_type == kmp_interop_type_tasksync) {
#ifdef SYNC_INTEROP
    __kmpc_omp_wait_deps(loc_ref, gtid, ndeps, dep_list, ndeps_noalias,
                         noalias_dep_list);
#else
    kmp_tasking_flags flags;
    //flags.tiedness = 1;
    //flags.proxy = 1;
    //flags.task_serial = 1;
    //flags.reserved = 42;
    flags.target = 1;
    flags.unshackled = 1;
    kmp_int32 flags32 = *((kmp_int32*)&flags);

    DeviceTy &Device = Devices[device_id];
    kmp_task_t *task  =__kmpc_omp_task_alloc(loc_ref, gtid, flags32,
                                            40,
                                            16,
                                            &__fake_task_use,
                                            device_id);
    //printf("Dev %p: T: %p\n", &Device, task);
    ((DeviceTy**)task->shareds)[0] = &Device;
    ((__tgt_async_info**)task->shareds)[1] = interop_val->async_info;
    assert(((DeviceTy**)task->shareds)[0] == &Device);

    int r = __kmpc_omp_task_with_deps(loc_ref, gtid,
                                    task, ndeps,
                                    dep_list,
                                    ndeps_noalias,
                                    noalias_dep_list);
    //printf("Done with fake task: %i \n", r);

#endif
  }

  delete *interop_ptr;
  *interop_ptr = omp_interop_none;
}
