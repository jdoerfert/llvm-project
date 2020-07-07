//===-------- interface.cpp - Target independent OpenMP target RTL --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of the interface to be used by Clang during the codegen of a
// target region.
//
//===----------------------------------------------------------------------===//

#include <omptarget.h>

#include "device.h"
#include "private.h"
#include "rtl.h"

#include <cassert>
#include <cstdlib>
#include <mutex>

// Store target policy (disabled, mandatory, default)
kmp_target_offload_kind_t TargetOffloadPolicy = tgt_default;
std::mutex TargetOffloadMtx;

////////////////////////////////////////////////////////////////////////////////
/// manage the success or failure of a target construct

static void HandleDefaultTargetOffload() {
  TargetOffloadMtx.lock();
  if (TargetOffloadPolicy == tgt_default) {
    if (omp_get_num_devices() > 0) {
      DP("Default TARGET OFFLOAD policy is now mandatory "
         "(devices were found)\n");
      TargetOffloadPolicy = tgt_mandatory;
    } else {
      DP("Default TARGET OFFLOAD policy is now disabled "
         "(no devices were found)\n");
      TargetOffloadPolicy = tgt_disabled;
    }
  }
  TargetOffloadMtx.unlock();
}

static int IsOffloadDisabled() {
  if (TargetOffloadPolicy == tgt_default) HandleDefaultTargetOffload();
  return TargetOffloadPolicy == tgt_disabled;
}

static void HandleTargetOutcome(bool success) {
  switch (TargetOffloadPolicy) {
    case tgt_disabled:
      if (success) {
        FATAL_MESSAGE0(1, "expected no offloading while offloading is disabled");
      }
      break;
    case tgt_default:
      FATAL_MESSAGE0(1, "default offloading policy must be switched to "
                        "mandatory or disabled");
      break;
    case tgt_mandatory:
      if (!success) {
        FATAL_MESSAGE0(
            1, "failure of target construct while offloading is mandatory");
      }
      break;
    }
}

template <bool Begin> static bool checkAndInitDevice(int64_t &DeviceId) {
  if (IsOffloadDisabled())
    return false;

  // No devices available?
  if (DeviceId == OFFLOAD_DEVICE_DEFAULT) {
    DeviceId = omp_get_default_device();
    DP("Use default device id %" PRId64 "\n", DeviceId);
  }

  // Invalid device id as we always expect a non-negative device id and it must
  // be less than the size of all device RTLs
  if (DeviceId < 0 || static_cast<uint64_t>(DeviceId) >= Devices.size()) {
    DP("Invalid device %" PRId64 "\n", DeviceId);
    return false;
  }

  if (!Begin)
    return true;

  if (CheckDeviceAndCtors(DeviceId) != OFFLOAD_SUCCESS) {
    DP("Failed to get device %" PRId64 " ready\n", DeviceId);
    HandleTargetOutcome(false);
    return false;
  } else {
    return true;
  }
}

////////////////////////////////////////////////////////////////////////////////
/// adds requires flags
EXTERN void __tgt_register_requires(int64_t flags) {
  RTLs->RegisterRequires(flags);
}

////////////////////////////////////////////////////////////////////////////////
/// adds a target shared library to the target execution image
EXTERN void __tgt_register_lib(__tgt_bin_desc *desc) {
  RTLs->RegisterLib(desc);
}

////////////////////////////////////////////////////////////////////////////////
/// unloads a target shared library
EXTERN void __tgt_unregister_lib(__tgt_bin_desc *desc) {
  RTLs->UnregisterLib(desc);
}

/// creates host-to-target data mapping, stores it in the
/// libomptarget.so internal structure (an entry in a stack of data maps)
/// and passes the data to the device.
EXTERN void __tgt_target_data_begin(int64_t device_id, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types) {
  // device_id will be corrected if it is default value
  if (!checkAndInitDevice<true>(device_id))
    return;

  DP("Entering data begin region for device %" PRId64 " with %d mappings\n",
     device_id, arg_num);

  DeviceTy &Device = Devices[device_id];

#ifdef OMPTARGET_DEBUG
  for (int i = 0; i < arg_num; ++i) {
    DP("Entry %2d: Base=" DPxMOD ", Begin=" DPxMOD ", Size=%" PRId64
       ", Type=0x%" PRIx64 "\n",
       i, DPxPTR(args_base[i]), DPxPTR(args[i]), arg_sizes[i], arg_types[i]);
  }
#endif

  const int rc = targetData<TargetDataFuncTy::Begin>(
      Device, arg_num, args_base, args, arg_sizes, arg_types);
  HandleTargetOutcome(rc == OFFLOAD_SUCCESS);
}

EXTERN void __tgt_target_data_begin_nowait(int64_t device_id, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types,
    int32_t depNum, void *depList, int32_t noAliasDepNum,
    void *noAliasDepList) {
  // device_id will be corrected if it is default value
  if (!checkAndInitDevice<true>(device_id))
    return;

  DP("Entering data begin region for device %" PRId64 " with %d mappings\n",
     device_id, arg_num);

  DeviceTy &Device = Devices[device_id];

  const int rc = targetDataNowait<TargetDataFuncTy::Begin>(
      Device, arg_num, args_base, args, arg_sizes, arg_types, depNum, depList,
      noAliasDepNum, noAliasDepList);
  HandleTargetOutcome(rc == OFFLOAD_SUCCESS);
}

/// passes data from the target, releases target memory and destroys
/// the host-target mapping (top entry from the stack of data maps)
/// created by the last __tgt_target_data_begin.
EXTERN void __tgt_target_data_end(int64_t device_id, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types) {
  // device_id will be corrected if it is default value
  if (!checkAndInitDevice<false>(device_id))
    return;

  DP("Entering data end region for device %" PRId64 " with %d mappings\n",
     device_id, arg_num);

  RTLsMtx->lock();
  size_t Devices_size = Devices.size();
  RTLsMtx->unlock();
  if (Devices_size <= (size_t)device_id) {
    DP("Device ID  %" PRId64 " does not have a matching RTL.\n", device_id);
    HandleTargetOutcome(false);
    return;
  }

  DeviceTy &Device = Devices[device_id];
  if (!Device.IsInit) {
    DP("Uninit device: ignore");
    HandleTargetOutcome(false);
    return;
  }

#ifdef OMPTARGET_DEBUG
  for (int i = 0; i < arg_num; ++i) {
    DP("Entry %2d: Base=" DPxMOD ", Begin=" DPxMOD ", Size=%" PRId64
       ", Type=0x%" PRIx64 "\n",
       i, DPxPTR(args_base[i]), DPxPTR(args[i]), arg_sizes[i], arg_types[i]);
  }
#endif

  const int rc = targetData<TargetDataFuncTy::End>(Device, arg_num, args_base,
                                                   args, arg_sizes, arg_types);
  HandleTargetOutcome(rc == OFFLOAD_SUCCESS);
}

EXTERN void __tgt_target_data_end_nowait(int64_t device_id, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types,
    int32_t depNum, void *depList, int32_t noAliasDepNum,
    void *noAliasDepList) {
  // device_id will be corrected if it is default value
  if (!checkAndInitDevice<true>(device_id))
    return;

  DP("Entering data end region for device %" PRId64 " with %d mappings\n",
     device_id, arg_num);

  DeviceTy &Device = Devices[device_id];

  const int rc = targetDataNowait<TargetDataFuncTy::End>(
      Device, arg_num, args_base, args, arg_sizes, arg_types, depNum, depList,
      noAliasDepNum, noAliasDepList);
  HandleTargetOutcome(rc == OFFLOAD_SUCCESS);
}

EXTERN void __tgt_target_data_update(int64_t device_id, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types) {
  // device_id will be corrected if it is default value
  if (!checkAndInitDevice<true>(device_id))
    return;

  DP("Entering data update region for device %" PRId64 " with %d mappings\n",
     device_id, arg_num);

  DeviceTy &Device = Devices[device_id];
  const int rc = targetData<TargetDataFuncTy::Update>(
      Device, arg_num, args_base, args, arg_sizes, arg_types);
  HandleTargetOutcome(rc == OFFLOAD_SUCCESS);
}

EXTERN void __tgt_target_data_update_nowait(
    int64_t device_id, int32_t arg_num, void **args_base, void **args,
    int64_t *arg_sizes, int64_t *arg_types, int32_t depNum, void *depList,
    int32_t noAliasDepNum, void *noAliasDepList) {
  // device_id will be corrected if it is default value
  if (!checkAndInitDevice<true>(device_id))
    return;

  DP("Entering data update region for device %" PRId64 " with %d mappings\n",
     device_id, arg_num);

  DeviceTy &Device = Devices[device_id];

  // TODO: this part should be refined maybe in case of memory error
  __tgt_async_info *async_info = new __tgt_async_info;
  async_info->DeviceID = device_id;

  const int rc = targetDataNowait<TargetDataFuncTy::Update>(
      Device, arg_num, args_base, args, arg_sizes, arg_types, depNum, depList,
      noAliasDepNum, noAliasDepList);
  HandleTargetOutcome(rc == OFFLOAD_SUCCESS);
}

EXTERN int __tgt_target(int64_t device_id, void *host_ptr, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types) {
  // device_id will be corrected if it is default value
  if (!checkAndInitDevice<true>(device_id))
    return OFFLOAD_FAIL;

  DP("Entering target region with entry point " DPxMOD " and device Id %" PRId64
     "\n",
     DPxPTR(host_ptr), device_id);

#ifdef OMPTARGET_DEBUG
  for (int i = 0; i < arg_num; ++i) {
    DP("Entry %2d: Base=" DPxMOD ", Begin=" DPxMOD ", Size=%" PRId64
       ", Type=0x%" PRIx64 "\n",
       i, DPxPTR(args_base[i]), DPxPTR(args[i]), arg_sizes[i], arg_types[i]);
  }
#endif

  const int rc =
      target(device_id, host_ptr, arg_num, args_base, args, arg_sizes,
             arg_types, /* TeamNum */ 0, /* ThreadLimit */ 0,
             /* IsTeamConstruct*/ false);
  HandleTargetOutcome(rc == OFFLOAD_SUCCESS);
  return rc;
}

EXTERN int __tgt_target_nowait(int64_t device_id, void *host_ptr,
    int32_t arg_num, void **args_base, void **args, int64_t *arg_sizes,
    int64_t *arg_types, int32_t depNum, void *depList, int32_t noAliasDepNum,
    void *noAliasDepList) {
  // device_id will be corrected if it is default value
  if (!checkAndInitDevice<true>(device_id))
    return OFFLOAD_FAIL;

  DP("Entering target region with entry point " DPxMOD " and device Id %" PRId64
     "\n",
     DPxPTR(host_ptr), device_id);

#ifdef OMPTARGET_DEBUG
  for (int i = 0; i < arg_num; ++i) {
    DP("Entry %2d: Base=" DPxMOD ", Begin=" DPxMOD ", Size=%" PRId64
       ", Type=0x%" PRIx64 "\n",
       i, DPxPTR(args_base[i]), DPxPTR(args[i]), arg_sizes[i], arg_types[i]);
  }
#endif

  const int rc = targetNowait(
      device_id, host_ptr, arg_num, args_base, args, arg_sizes, arg_types,
      /* TeamNum */ 0, /* ThreadLimit */ 0, /* IsTeamConstruct */ false, depNum,
      depList, noAliasDepNum, noAliasDepList);
  HandleTargetOutcome(rc == OFFLOAD_SUCCESS);
  return rc;
}

EXTERN int __tgt_target_teams(int64_t device_id, void *host_ptr,
    int32_t arg_num, void **args_base, void **args, int64_t *arg_sizes,
    int64_t *arg_types, int32_t team_num, int32_t thread_limit) {
  // device_id will be corrected if it is default value
  if (!checkAndInitDevice<true>(device_id))
    return OFFLOAD_FAIL;

  DP("Entering target region with entry point " DPxMOD " and device Id %" PRId64
     "\n",
     DPxPTR(host_ptr), device_id);

#ifdef OMPTARGET_DEBUG
  for (int i = 0; i < arg_num; ++i) {
    DP("Entry %2d: Base=" DPxMOD ", Begin=" DPxMOD ", Size=%" PRId64
       ", Type=0x%" PRIx64 "\n",
       i, DPxPTR(args_base[i]), DPxPTR(args[i]), arg_sizes[i], arg_types[i]);
  }
#endif

  const int rc = target(device_id, host_ptr, arg_num, args_base, args,
                        arg_sizes, arg_types, team_num, thread_limit,
                        /* IsTeamConstruct */ true);
  HandleTargetOutcome(rc == OFFLOAD_SUCCESS);

  return rc;
}

EXTERN int __tgt_target_teams_nowait(int64_t device_id, void *host_ptr,
    int32_t arg_num, void **args_base, void **args, int64_t *arg_sizes,
    int64_t *arg_types, int32_t team_num, int32_t thread_limit, int32_t depNum,
    void *depList, int32_t noAliasDepNum, void *noAliasDepList) {
  // device_id will be corrected if it is default value
  if (!checkAndInitDevice<true>(device_id))
    return OFFLOAD_FAIL;

  DP("Entering target region with entry point " DPxMOD " and device Id %" PRId64
     "\n",
     DPxPTR(host_ptr), device_id);

#ifdef OMPTARGET_DEBUG
  for (int i = 0; i < arg_num; ++i) {
    DP("Entry %2d: Base=" DPxMOD ", Begin=" DPxMOD ", Size=%" PRId64
       ", Type=0x%" PRIx64 "\n",
       i, DPxPTR(args_base[i]), DPxPTR(args[i]), arg_sizes[i], arg_types[i]);
  }
#endif

  const int rc = targetNowait(device_id, host_ptr, arg_num, args_base, args,
                              arg_sizes, arg_types, team_num, thread_limit,
                              /* IsTeamConstruct */ true, depNum, depList,
                              noAliasDepNum, noAliasDepList);
  HandleTargetOutcome(rc == OFFLOAD_SUCCESS);
  return rc;
}

// Get the current number of components for a user-defined mapper.
EXTERN int64_t __tgt_mapper_num_components(void *rt_mapper_handle) {
  auto *MapperComponentsPtr = (struct MapperComponentsTy *)rt_mapper_handle;
  int64_t size = MapperComponentsPtr->Components.size();
  DP("__tgt_mapper_num_components(Handle=" DPxMOD ") returns %" PRId64 "\n",
     DPxPTR(rt_mapper_handle), size);
  return size;
}

// Push back one component for a user-defined mapper.
EXTERN void __tgt_push_mapper_component(void *rt_mapper_handle, void *base,
                                        void *begin, int64_t size,
                                        int64_t type) {
  DP("__tgt_push_mapper_component(Handle=" DPxMOD
     ") adds an entry (Base=" DPxMOD ", Begin=" DPxMOD ", Size=%" PRId64
     ", Type=0x%" PRIx64 ").\n",
     DPxPTR(rt_mapper_handle), DPxPTR(base), DPxPTR(begin), size, type);
  auto *MapperComponentsPtr = (struct MapperComponentsTy *)rt_mapper_handle;
  MapperComponentsPtr->Components.push_back(
      MapComponentInfoTy(base, begin, size, type));
}

EXTERN void __kmpc_push_target_tripcount(int64_t device_id,
    uint64_t loop_tripcount) {
  if (IsOffloadDisabled())
    return;

  if (device_id == OFFLOAD_DEVICE_DEFAULT) {
    device_id = omp_get_default_device();
  }

  if (CheckDeviceAndCtors(device_id) != OFFLOAD_SUCCESS) {
    DP("Failed to get device %" PRId64 " ready\n", device_id);
    HandleTargetOutcome(false);
    return;
  }

  DP("__kmpc_push_target_tripcount(%" PRId64 ", %" PRIu64 ")\n", device_id,
     loop_tripcount);
  TblMapMtx->lock();
  Devices[device_id].LoopTripCnt.emplace(__kmpc_global_thread_num(NULL),
                                         loop_tripcount);
  TblMapMtx->unlock();
}

EXTERN void __kmpc_free_async_info(void *Ptr) {
  if (!Ptr)
    return;
  __tgt_async_info *AsyncInfo = reinterpret_cast<__tgt_async_info *>(Ptr);
  int DeviceId = AsyncInfo->DeviceID;

  assert(DeviceId >= 0 && "Invalid DeviceId");

  Devices[DeviceId].releaseAsyncInfo(AsyncInfo);
}
