//===- DeviceInterface.cpp - Target independent plugin device interface ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "DeviceInterface.h"
#include "Debug.h"
#include "GlobalHandler.h"
#include "omptarget.h"
#include "omptargetplugin.h"
#include <cstdint>
#include <limits>

using namespace llvm;
using namespace omp;
using namespace plugin;

AsyncInfoWrapperTy::~AsyncInfoWrapperTy() {
  // If we used a local async info object we want synchronous behavior.
  // In that case, and assuming the current status code is OK, we will
  // synchronize explicitly when the object is deleted.
  if (AsyncInfoPtr == &LocalAsyncInfo && SC.isOK())
    SC = Device.synchronize(&LocalAsyncInfo);
}

void GenericKernelTy::init(GenericDeviceTy &GenericDevice) {
  if (Initialized)
    return;

  PreferredThreadCount = getDefaultThreadCount(GenericDevice);
  if (isGenericMode())
    PreferredThreadCount += GenericDevice.getWarpSize();

  MaxThreadCount = GenericDevice.getThreadLimit();

  DynamicMemorySize = GenericDevice.getDynamicMemorySize();

  initImpl(GenericDevice);

  Initialized = true;
}

StatusCode GenericKernelTy::launch(GenericDeviceTy &GenericDevice,
                                   void **ArgPtrs, ptrdiff_t *ArgOffsets,
                                   int32_t NumArgs, int32_t NumTeamsClause,
                                   int32_t ThreadLimitClause,
                                   int32_t LoopTripCount,
                                   AsyncInfoWrapperTy &AsyncInfoWrapper) {
  init(GenericDevice);

  void *KernelArgsPtr =
      argumentPrepareImpl(GenericDevice, ArgPtrs, ArgOffsets, NumArgs, AsyncInfoWrapper);

  int32_t NumThreads = getNumThreads(GenericDevice, ThreadLimitClause);
  int32_t NumBlocks =
      getNumBlocks(GenericDevice, NumTeamsClause, LoopTripCount, NumThreads);

  INFO(OMP_INFOTYPE_PLUGIN_KERNEL, GenericDevice.DeviceId,
       "Launching kernel %s with %d blocks and %d threads in %s mode\n",
       getName(), NumBlocks, NumThreads, getExecutionModeName());
  StatusCode SC =
      launchImpl(GenericDevice, NumThreads, NumBlocks, DynamicMemorySize,
                 KernelArgsPtr, AsyncInfoWrapper);
  if (SC) {
    REPORT("Failure to launch kernel %s\n", getName());
  }
  return SC;
}

int32_t GenericKernelTy::getNumThreads(GenericDeviceTy &GenericDevice,
                                       int32_t ThreadLimitClause) const {
  return std::min(MaxThreadCount, ThreadLimitClause > 0 ? ThreadLimitClause
                                                        : PreferredThreadCount);
}

int32_t GenericKernelTy::getNumBlocks(GenericDeviceTy &GenericDevice,
                                      int32_t NumTeamsClause,
                                      int32_t LoopTripCount,
                                      int32_t NumThreads) const {
  int32_t PreferredNumBlocks = getDefaultBlockCount(GenericDevice);
  if (NumTeamsClause > 0) {
    PreferredNumBlocks = NumTeamsClause;
  } else if (LoopTripCount > 0) {
    if (isSPMDMode()) {
      // We have a combined construct, i.e. `target teams distribute
      // parallel for [simd]`. We launch so many teams so that each thread
      // will execute one iteration of the loop. round up to the nearest
      // integer
      PreferredNumBlocks = ((LoopTripCount - 1) / NumThreads) + 1;
    } else {
      assert((isGenericMode() || isGenericSPMDMode()) &&
             "Unexpected execution mode!");
      // If we reach this point, then we have a non-combined construct, i.e.
      // `teams distribute` with a nested `parallel for` and each team is
      // assigned one iteration of the `distribute` loop. E.g.:
      //
      // #pragma omp target teams distribute
      // for(...loop_tripcount...) {
      //   #pragma omp parallel for
      //   for(...) {}
      // }
      //
      // Threads within a team will execute the iterations of the `parallel`
      // loop.
      PreferredNumBlocks = LoopTripCount;
    }
  }
  return std::min(PreferredNumBlocks, GenericDevice.getBlockLimit());
}

GenericDeviceTy::GenericDeviceTy(int32_t DeviceId,
                                 const llvm::omp::GV &GridValues)
    : DeviceId(DeviceId), OMP_TeamLimit("OMP_TEAM_LIMIT", DeviceId),
      OMP_NumTeams("OMP_NUM_TEAMS", DeviceId),
      OMPX_DebugKind("LIBOMPTARGET_DEVICE_RTL_DEBUG", DeviceId, 0),
      OMPX_SharedMemorySize("LIBOMPTARGET_SHARED_MEMORY_SIZE", DeviceId, 0),
      OMP_TeamsThreadLimit("OMP_TEAMS_THREAD_LIMIT", DeviceId),
      OMPX_TargetStackSize(
          "LIBOMPTARGET_STACK_SIZE", DeviceId,
          [this](uint64_t &V) { return getDeviceStackSize(V); },
          [this](uint64_t V) { return setDeviceStackSize(V); }),
      OMPX_TargetHeapSize(
          "LIBOMPTARGET_HEAP_SIZE", DeviceId,
          [this](uint64_t &V) { return getDeviceHeapSize(V); },
          [this](uint64_t V) { return setDeviceHeapSize(V); }),
      GridValues(GridValues) {
  if (OMP_NumTeams > 0)
    this->GridValues.GV_Max_Teams =
        std::min(this->GridValues.GV_Max_Teams, uint32_t(OMP_NumTeams));
  if (OMP_TeamsThreadLimit > 0)
    this->GridValues.GV_Max_WG_Size = std::min(this->GridValues.GV_Max_WG_Size,
                                               uint32_t(OMP_TeamsThreadLimit));

#if 0
    INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
         "Device supports up to %d CUDA blocks and %d threads with a "
         "warp size of %d\n",
         Devices[DeviceId].BlocksPerGrid, Devices[DeviceId].ThreadsPerBlock,
         Devices[DeviceId].WarpSize);
    INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
         "Device heap size is %d Bytes, device stack size is %d Bytes per "
         "thread\n",
         (int)HeapLimit, (int)StackLimit);
#endif
};

StatusCode GenericDeviceTy::init(GenericPluginTy &Plugin) {
  StatusCode SC = setupDeviceEnvironment(Plugin);
  if (SC) {
    INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
         "Failed setting up the device environment.");
    return SC;
  }

  return initImpl(Plugin);
}

__tgt_target_table *
GenericDeviceTy::loadBinary(const __tgt_device_image *Image) {
  DP("Load data from image " DPxMOD "\n", DPxPTR(Image->ImageStart));
  this->Image = Image;

  StatusCode SC = loadBinaryImpl();
  if (SC) {
    REPORT("Failure to load binary (%p) in vendor specific part.", Image);
    return nullptr;
  }

  // Clear the offload table as we are going to create a new one.
  OffloadEntryTable.clear();

  if (!registerOffloadEntries()) {
    INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
         "Failed to register offload entries from device image.");
  }

  return OffloadEntryTable;
}

StatusCode GenericDeviceTy::setupDeviceEnvironment(GenericPluginTy &Plugin) {
  DeviceEnvironment.DebugKind = OMPX_DebugKind;
  DeviceEnvironment.NumDevices = Plugin.getNumDevices();
  // TODO: The device ID used here is not the real device ID used by OpenMP.
  DeviceEnvironment.DeviceNum = DeviceId;
  DeviceEnvironment.DynamicMemSize = OMPX_SharedMemorySize;

  GlobalHandlerTy &GlobalHandler = Plugin.getGlobalHandler();
  GlobalTy DeviceEnvGlobal("omptarget_device_environment",
                           sizeof(DeviceEnvironmentTy), &DeviceEnvironment);
  if (!GlobalHandler.writeGlobalToImage(*this, DeviceEnvGlobal)) {
    INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
         "Failed to write device environment into image.");
    // TODO: Check the device gfx name against the image gfx name.
    return StatusCode::FAIL;
  }
  return StatusCode::OK;
}

StatusCode GenericDeviceTy::registerOffloadEntries() {
  const __tgt_offload_entry *Begin = Image->EntriesBegin;
  const __tgt_offload_entry *End = Image->EntriesEnd;
  for (const __tgt_offload_entry *Entry = Begin; Entry != End; ++Entry) {
    if (!Entry->addr) {
      // The host should have always something in the address to
      // uniquely identify the entry.
      INFO(OMP_INFOTYPE_ALL, DeviceId,
           "Unexpected host entry without address (size: %ld), abort!\n",
           Entry->size);
      return StatusCode::FAIL;
    }

    if (Entry->size) {
      if (!registerGlobalOffloadEntry(Entry))
        return StatusCode::FAIL;
    } else {
      if (!registerKernelOffloadEntry(Entry))
        return StatusCode::FAIL;
    }
  }
  return StatusCode::OK;
}

StatusCode GenericDeviceTy::registerGlobalOffloadEntry(
    const __tgt_offload_entry *GlobalEntry) {

  GenericPluginTy &Plugin = getPlugin();
  GlobalHandlerTy &GlobalHandler = Plugin.getGlobalHandler();
  __tgt_offload_entry DeviceEntry = *GlobalEntry;
  StaticGlobalTy<void *> Global(GlobalEntry->name);
  if (!GlobalHandler.getGlobalMetadataFromDevice(*this, Global))
    return StatusCode::FAIL;
  DeviceEntry.addr = Global.getPtr();

  // Note: In the current implementation declare target variables
  // can either be link or to. This means that once unified
  // memory is activated via the requires directive, the variable
  // can be used directly from the host in both cases.
  if (Plugin.getRequiresFlags() & OMP_REQ_UNIFIED_SHARED_MEMORY &&
      (GlobalEntry->flags & OMP_DECLARE_TARGET_LINK)) {
    // If unified memory is present any target link or to variables
    // can access host addresses directly. There is no longer a
    // need for device copies.
    Global.setValue(GlobalEntry->addr);
    if (!GlobalHandler.writeGlobalToDevice(*this, Global))
      return StatusCode::FAIL;
  }

  OffloadEntryTable.addEntry(DeviceEntry);
  return StatusCode::OK;
}

StatusCode GenericDeviceTy::registerKernelOffloadEntry(
    const __tgt_offload_entry *KernelEntry) {
  __tgt_offload_entry DeviceEntry = *KernelEntry;
  void *Kernel = constructKernelEntry(KernelEntry);
  if (!Kernel)
    return StatusCode::FAIL;
  DeviceEntry.addr = (void *)Kernel;
  OffloadEntryTable.addEntry(DeviceEntry);
  return StatusCode::OK;
}

StatusCode GenericDeviceTy::synchronize(__tgt_async_info *AsyncInfo) {
  if (!AsyncInfo || !AsyncInfo->Queue)
    return StatusCode::FAIL;
  StatusCode SC = synchronizeImpl(*AsyncInfo);
  if (SC) {
    REPORT("Error when synchronizing stream (%p): %s\n", AsyncInfo->Queue,
           getErrorStr(SC));
  }
  return SC;
}

void *GenericDeviceTy::dataAlloc(int64_t Size, void *HostPtr,
                                 TargetAllocTy Kind) {
  switch (Kind) {
  case TARGET_ALLOC_DEFAULT:
  case TARGET_ALLOC_DEVICE:
    if (MemoryManager)
      return MemoryManager->allocate(Size, HostPtr);
    LLVM_FALLTHROUGH;
  case TARGET_ALLOC_HOST:
  case TARGET_ALLOC_SHARED:
    return allocate(Size, HostPtr, Kind);
  }

  REPORT("Invalid target data allocation kind or requested allocator not "
         "implemented yet\n");

  return nullptr;
}

StatusCode GenericDeviceTy::dataDelete(void *TgtPtr) {
  StatusCode SC = StatusCode::OK;

  if (MemoryManager) {
    SC = StatusCode(MemoryManager->free(TgtPtr));
  } else {
    SC = StatusCode(free(TgtPtr));
  }

  if (SC)
    REPORT("Failed to deallocate device pointer %p\n", TgtPtr);
  return SC;
}

StatusCode GenericDeviceTy::dataSubmit(void *TgtPtr, const void *HstPtr,
                                       int64_t Size,
                                       __tgt_async_info *AsyncInfo) {
  StatusCode SC = StatusCode::OK;
  AsyncInfoWrapperTy AsyncInfoWrapper(SC, *this, AsyncInfo);
  SC = dataSubmitImpl(TgtPtr, HstPtr, Size, AsyncInfoWrapper);
  if (SC) {
    REPORT("Error when copying data from host to device. Pointers: host "
           "= " DPxMOD ", device = " DPxMOD ", size = %" PRId64 "\n",
           DPxPTR(HstPtr), DPxPTR(TgtPtr), Size);
  }
  return SC;
}

StatusCode GenericDeviceTy::dataRetrieve(void *HstPtr, const void *TgtPtr,
                                         int64_t Size,
                                         __tgt_async_info *AsyncInfo) {
  StatusCode SC = StatusCode::OK;
  AsyncInfoWrapperTy AsyncInfoWrapper(SC, *this, AsyncInfo);
  SC = dataRetrieveImpl(HstPtr, TgtPtr, Size, AsyncInfoWrapper);
  if (SC) {
    REPORT("Error when copying data from device to host. Pointers: host "
           "= " DPxMOD ", device = " DPxMOD ", size = %" PRId64 "\n",
           DPxPTR(HstPtr), DPxPTR(TgtPtr), Size);
  }
  return SC;
}

StatusCode GenericDeviceTy::dataExchange(const void *SrcPtr, int32_t DstDevId,
                                         void *DstPtr, int64_t Size,
                                         __tgt_async_info *AsyncInfo) {
  StatusCode SC = StatusCode::OK;
  AsyncInfoWrapperTy AsyncInfoWrapper(SC, *this, AsyncInfo);
  SC = dataExchangeImpl(SrcPtr, DstDevId, DstPtr, Size, AsyncInfoWrapper);
  if (SC) {
    REPORT("Error when copying data from device (%d) to device (%d). Pointers: "
           "host "
           "= " DPxMOD ", device = " DPxMOD ", size = %" PRId64 "\n",
           DeviceId, DstDevId, DPxPTR(SrcPtr), DPxPTR(DstPtr), Size);
  }
  return SC;
}

StatusCode GenericDeviceTy::runTargetTeamRegion(
    void *EntryPtr, void **ArgPtrs, ptrdiff_t *ArgOffsets, int32_t NumArgs,
    int32_t NumTeamsClause, int32_t ThreadLimitClause, uint64_t LoopTripCount,
    __tgt_async_info *AsyncInfo) {
  StatusCode SC = StatusCode::OK;
  AsyncInfoWrapperTy AsyncInfoWrapper(SC, *this, AsyncInfo);

  GenericKernelTy &GenericKernel =
      *reinterpret_cast<GenericKernelTy *>(EntryPtr);

  int32_t LoopTripCount32 =
      LoopTripCount > uint64_t(std::numeric_limits<int32_t>::max())
          ? std::numeric_limits<int32_t>::max()
          : LoopTripCount;

  SC = GenericKernel.launch(*this, ArgPtrs, ArgOffsets, NumArgs, NumTeamsClause,
                            ThreadLimitClause, LoopTripCount32,
                            AsyncInfoWrapper);
  return SC;
}

void GenericDeviceTy::printInfo() {
  // TODO: Print generic information here
  printInfoImpl();
}

StatusCode GenericDeviceTy::createEvent(void **EventPtrStorage) {
  StatusCode SC = createEventImpl(EventPtrStorage);
  if (SC) {
    REPORT("Failure to create event: %s\n", getErrorStr(SC));
  }
  return SC;
}
StatusCode GenericDeviceTy::destroyEvent(void *EventPtr) {
  StatusCode SC = destroyEventImpl(EventPtr);
  if (SC) {
    REPORT("Failure to destroy event (%p): %s\n", EventPtr, getErrorStr(SC));
  }
  return SC;
}
StatusCode GenericDeviceTy::recordEvent(void *EventPtr,
                                        __tgt_async_info *AsyncInfo) {
  StatusCode SC = StatusCode::OK;
  AsyncInfoWrapperTy AsyncInfoWrapper(SC, *this, AsyncInfo);
  SC = recordEventImpl(EventPtr, AsyncInfoWrapper);
  if (SC) {
    REPORT("Failure to record event (%p): %s\n", EventPtr, getErrorStr(SC));
  }
  return SC;
}
StatusCode GenericDeviceTy::waitEvent(void *EventPtr,
                                      __tgt_async_info *AsyncInfo) {
  StatusCode SC = StatusCode::OK;
  AsyncInfoWrapperTy AsyncInfoWrapper(SC, *this, AsyncInfo);
  SC = waitEventImpl(EventPtr, AsyncInfoWrapper);
  if (SC) {
    REPORT("Failure to wait for event (%p): %s\n", EventPtr, getErrorStr(SC));
  }
  return SC;
}
StatusCode GenericDeviceTy::syncEvent(void *EventPtr) {
  StatusCode SC = syncEventImpl(EventPtr);
  if (SC) {
    REPORT("Failure to sync event (%p): %s\n", EventPtr, getErrorStr(SC));
  }
  return SC;
}

/// Exposed library API function, basically wrappers around the GenericDeviceTy
/// functionality with the same name. All "non-async" functions are redirected
/// to the "async" versions right away with a NULL async_info_ptr.
#ifdef __cplusplus
extern "C" {
#endif

int32_t __tgt_rtl_is_valid_binary(__tgt_device_image *image) {
  return elf_check_machine(image, getPlugin().getMagicElfBits());
}

int32_t __tgt_rtl_number_of_devices() { return getPlugin().getNumDevices(); }

int64_t __tgt_rtl_init_requires(int64_t RequiresFlags) {
  getPlugin().setRequiresFlag(RequiresFlags);
  return RequiresFlags;
}

int32_t __tgt_rtl_is_data_exchangable(int32_t src_dev_id, int32_t dst_dev_id) {
  return getPlugin().isDataExchangable(src_dev_id, dst_dev_id);
}

int32_t __tgt_rtl_init_device(int32_t device_id) {
  return getPlugin().getDevice(device_id).init(getPlugin());
}

__tgt_target_table *__tgt_rtl_load_binary(int32_t device_id,
                                          __tgt_device_image *image) {
  return getPlugin().getDevice(device_id).loadBinary(image);
}

void *__tgt_rtl_data_alloc(int32_t device_id, int64_t size, void *HostPtr,
                           int32_t kind) {
  return getPlugin().getDevice(device_id).dataAlloc(size, HostPtr,
                                                    (TargetAllocTy)kind);
}

int32_t __tgt_rtl_data_delete(int32_t device_id, void *tgt_ptr) {
  return getPlugin().getDevice(device_id).dataDelete(tgt_ptr);
}

int32_t __tgt_rtl_data_submit(int32_t device_id, void *tgt_ptr, void *hst_ptr,
                              int64_t size) {
  return __tgt_rtl_data_submit_async(device_id, tgt_ptr, hst_ptr, size,
                                     /* async_info_ptr */ nullptr);
}

int32_t __tgt_rtl_data_submit_async(int32_t device_id, void *tgt_ptr,
                                    void *hst_ptr, int64_t size,
                                    __tgt_async_info *async_info_ptr) {
  return getPlugin().getDevice(device_id).dataSubmit(tgt_ptr, hst_ptr, size,
                                                     async_info_ptr);
}

int32_t __tgt_rtl_data_retrieve(int32_t device_id, void *hst_ptr, void *tgt_ptr,
                                int64_t size) {
  return __tgt_rtl_data_retrieve_async(device_id, hst_ptr, tgt_ptr, size,
                                       /* async_info_ptr */ nullptr);
}

int32_t __tgt_rtl_data_retrieve_async(int32_t device_id, void *hst_ptr,
                                      void *tgt_ptr, int64_t size,
                                      __tgt_async_info *async_info_ptr) {
  return getPlugin().getDevice(device_id).dataRetrieve(hst_ptr, tgt_ptr, size,
                                                       async_info_ptr);
}

int32_t __tgt_rtl_data_exchange(int32_t src_dev_id, void *src_ptr,
                                int32_t dst_dev_id, void *dst_ptr,
                                int64_t size) {
  return __tgt_rtl_data_exchange_async(src_dev_id, src_ptr, dst_dev_id, dst_ptr,
                                       size, /* async_info_ptr */ nullptr);
}

int32_t __tgt_rtl_data_exchange_async(int32_t src_dev_id, void *src_ptr,
                                      int dst_dev_id, void *dst_ptr,
                                      int64_t size,
                                      __tgt_async_info *AsyncInfo) {
  return getPlugin()
      .getDevice(src_dev_id)
      .dataExchange(src_ptr, dst_dev_id, dst_ptr, size, AsyncInfo);
}

int32_t __tgt_rtl_run_target_team_region(int32_t device_id, void *tgt_entry_ptr,
                                         void **tgt_args,
                                         ptrdiff_t *tgt_offsets,
                                         int32_t arg_num, int32_t team_num,
                                         int32_t thread_limit,
                                         uint64_t loop_tripcount) {
  return __tgt_rtl_run_target_team_region_async(
      device_id, tgt_entry_ptr, tgt_args, tgt_offsets, arg_num, team_num,
      thread_limit, loop_tripcount,
      /* async_info_ptr */ nullptr);
}

int32_t __tgt_rtl_run_target_team_region_async(
    int32_t device_id, void *tgt_entry_ptr, void **tgt_args,
    ptrdiff_t *tgt_offsets, int32_t arg_num, int32_t team_num,
    int32_t thread_limit, uint64_t loop_tripcount,
    __tgt_async_info *async_info_ptr) {
  return getPlugin().getDevice(device_id).runTargetTeamRegion(
      tgt_entry_ptr, tgt_args, tgt_offsets, arg_num, team_num, thread_limit,
      loop_tripcount, async_info_ptr);
}

int32_t __tgt_rtl_synchronize(int32_t device_id,
                              __tgt_async_info *async_info_ptr) {
  return getPlugin().getDevice(device_id).synchronize(async_info_ptr);
}

int32_t __tgt_rtl_run_target_region(int32_t device_id, void *tgt_entry_ptr,
                                    void **tgt_args, ptrdiff_t *tgt_offsets,
                                    int32_t arg_num) {
  return __tgt_rtl_run_target_region_async(device_id, tgt_entry_ptr, tgt_args,
                                           tgt_offsets, arg_num,
                                           /* async_info_ptr */ nullptr);
}

int32_t __tgt_rtl_run_target_region_async(int32_t device_id,
                                          void *tgt_entry_ptr, void **tgt_args,
                                          ptrdiff_t *tgt_offsets,
                                          int32_t arg_num,
                                          __tgt_async_info *async_info_ptr) {
  return __tgt_rtl_run_target_team_region_async(
      device_id, tgt_entry_ptr, tgt_args, tgt_offsets, arg_num,
      /* team num*/ 1, /* thread_limit */ 1, /* loop_tripcount */ 0,
      async_info_ptr);
}

void __tgt_rtl_print_device_info(int32_t device_id) {
  getPlugin().getDevice(device_id).printInfo();
}

int32_t __tgt_rtl_create_event(int32_t device_id, void **event) {
  return getPlugin().getDevice(device_id).createEvent(event);
}

int32_t __tgt_rtl_record_event(int32_t device_id, void *event_ptr,
                               __tgt_async_info *async_info_ptr) {
  return getPlugin().getDevice(device_id).recordEvent(event_ptr,
                                                      async_info_ptr);
}

int32_t __tgt_rtl_wait_event(int32_t device_id, void *event_ptr,
                             __tgt_async_info *async_info_ptr) {
  return getPlugin().getDevice(device_id).waitEvent(event_ptr, async_info_ptr);
}

int32_t __tgt_rtl_sync_event(int32_t device_id, void *event_ptr) {
  return getPlugin().getDevice(device_id).syncEvent(event_ptr);
}

int32_t __tgt_rtl_destroy_event(int32_t device_id, void *event_ptr) {
  return getPlugin().getDevice(device_id).destroyEvent(event_ptr);
}

void __tgt_rtl_set_info_flag(uint32_t NewInfoLevel) {
  std::atomic<uint32_t> &InfoLevel = getInfoLevelInternal();
  InfoLevel.store(NewInfoLevel);
}

#ifdef __cplusplus
}
#endif
