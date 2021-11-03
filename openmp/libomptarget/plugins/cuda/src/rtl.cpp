//===----RTLs/cuda/src/rtl.cpp - Target RTLs Implementation ------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RTL for CUDA machine
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cstddef>
#include <cuda.h>
#include <string>
#include <unordered_map>

#include "Debug.h"

#define TARGET_NAME CUDA
#define DEBUG_PREFIX "Target " GETNAME(TARGET_NAME) " RTL"

#include "DeviceInterface.h"
#include "GlobalHandler.h"

#include "llvm/Frontend/OpenMP/DeviceEnvironment.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/Frontend/OpenMP/OMPGridValues.h"

using namespace llvm;
using namespace omp;
using namespace plugin;

/// Forward declarations for all specialized data structures.
struct CUDAKernelTy;
struct CUDADeviceTy;
struct CUDAPluginTy;
struct CUDAStreamManagerTy;

template <typename... ArgsTy>
static bool checkResult(CUresult Err, const char *ErrMsg, ArgsTy... Args) {
  if (Err == CUDA_SUCCESS)
    return false;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wformat-security"
  REPORT(ErrMsg, Args...);
#pragma clang diagnostic pop

  const char *ErrStr = nullptr;
  CUresult ErrStrStatus = cuGetErrorString(Err, &ErrStr);
  if (ErrStrStatus == CUDA_ERROR_INVALID_VALUE) {
    REPORT("Unrecognized " GETNAME(TARGET_NAME) " error code: %d\n", Err);
  } else if (ErrStrStatus == CUDA_SUCCESS) {
    REPORT(GETNAME(TARGET_NAME) " error is: %s\n", ErrStr);
  } else {
    REPORT("Unresolved " GETNAME(
               TARGET_NAME) " error code: %d\n"
                            "Unsuccessful cuGetErrorString return status: %d\n",
           Err, ErrStrStatus);
  }
  return true;
}

/// Forward declaration to access the singleton plugin in lieu of a header.
static CUDAPluginTy &getCUDAPlugin();

struct CUDAKernelTy : public GenericKernelTy {
  CUDAKernelTy(const char *Name, OMPTgtExecModeFlags ExecutionMode,
               CUfunction Func)
      : GenericKernelTy(Name, ExecutionMode), Func(Func) {}

  void initImpl(GenericDeviceTy &GenericDevice) override {
    int MaxThreads;
    CUresult Err = cuFuncGetAttribute(
        &MaxThreads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, Func);
    if (!checkResult(Err, "Error returned from cuFuncGetAttribute\n"))
      MaxThreadCount = std::min(MaxThreadCount, MaxThreads);
  }

  void *argumentPrepareImpl(GenericDeviceTy &GenericDevice,void **ArgPtrs, ptrdiff_t *ArgOffsets,
                            int32_t NumArgs, AsyncInfoWrapperTy &AsyncInfoWrapper) override;
  StatusCode launchImpl(GenericDeviceTy &GenericDevice, int32_t NumThreads,
                        int32_t NumBlocks, int32_t DynamicMemorySize,
                        void *KernelArgs,
                        AsyncInfoWrapperTy &AsyncInfoWrapper) override;

  int32_t getDefaultBlockCount(GenericDeviceTy &GenericDevice) const override {
    return GenericDevice.getDefaultBlockCount();
  }
  int32_t getDefaultThreadCount(GenericDeviceTy &GenericDevice) const override {
    return GenericDevice.getDefaultThreadCount();
  }

private:
  SmallVector<void *, 16> Args;
  SmallVector<void *, 16> Ptrs;
  CUfunction Func;
};

struct CUDAStreamManagerTy : public StreamManagerTy<CUstream> {
  CUDAStreamManagerTy(CUDADeviceTy &CUDADevice);

  ~CUDAStreamManagerTy();

  StatusCode resizeStreamPoolImpl(int32_t OldSize, int32_t NewSize) override;

private:
  CUDADeviceTy &CUDADevice;
};

/// Structure contains per-device data.
struct CUDADeviceTy : public GenericDeviceTy {

  CUDADeviceTy(int32_t DeviceId)
      : GenericDeviceTy(DeviceId, NVPTXGridValues), CUDAStreamManager(*this) {}
  ~CUDADeviceTy() {
    // Close module
    if (Module)
      checkResult(cuModuleUnload(Module),
                  "Error returned from cuModuleUnload\n");

    if (Context) {
      if (setContext())
        return;
      checkResult(cuDevicePrimaryCtxRelease(Device),
                  "Error returned from cuDevicePrimaryCtxRelease\n");
    }
  }

  StatusCode initImpl(GenericPluginTy &Plugin) override {

    CUresult Err = cuDeviceGet(&Device, DeviceId);
    if (checkResult(Err, "Error returned from cuDeviceGet\n"))
      return StatusCode::FAIL;

    // Query the current flags of the primary context and set its flags if
    // it is inactive
    unsigned int FormerPrimaryCtxFlags = 0;
    int FormerPrimaryCtxIsActive = 0;
    Err = cuDevicePrimaryCtxGetState(Device, &FormerPrimaryCtxFlags,
                                     &FormerPrimaryCtxIsActive);
    if (checkResult(Err, "Error returned from cuDevicePrimaryCtxGetState\n"))
      return StatusCode::FAIL;

    if (FormerPrimaryCtxIsActive) {
      INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
           "The primary context is active, no change to its flags\n");
      if ((FormerPrimaryCtxFlags & CU_CTX_SCHED_MASK) !=
          CU_CTX_SCHED_BLOCKING_SYNC)
        INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
             "Warning: The current flags are not CU_CTX_SCHED_BLOCKING_SYNC\n");
    } else {
      INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
           "The primary context is inactive, set its flags to "
           "CU_CTX_SCHED_BLOCKING_SYNC\n");
      Err = cuDevicePrimaryCtxSetFlags(Device, CU_CTX_SCHED_BLOCKING_SYNC);
      if (checkResult(Err, "Error returned from cuDevicePrimaryCtxSetFlags\n"))
        return StatusCode::FAIL;
    }

    // Retain the per device primary context and save it to use whenever this
    // device is selected.
    Err = cuDevicePrimaryCtxRetain(&Context, Device);
    if (checkResult(Err, "Error returned from cuDevicePrimaryCtxRetain\n"))
      return StatusCode::FAIL;

    if (setContext())
      return StatusCode::FAIL;

    // Initialize stream pool
    if (CUDAStreamManager.init())
      return StatusCode::FAIL;

    // Query attributes to determine number of threads/block and blocks/grid.
    if (getDeviceAttr(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
                      GridValues.GV_Max_Teams))
      return StatusCode::FAIL;
    if (getDeviceAttr(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
                      GridValues.GV_Max_WG_Size))
      return StatusCode::FAIL;
    if (getDeviceAttr(CU_DEVICE_ATTRIBUTE_WARP_SIZE, GridValues.GV_Warp_Size))
      return StatusCode::FAIL;

    return StatusCode::OK;
  }

  GenericKernelTy *
  constructKernelEntry(const __tgt_offload_entry *KernelEntry) override {
    CUfunction Func;
    CUresult Err = cuModuleGetFunction(&Func, Module, KernelEntry->name);
    if (checkResult(Err, "Loading '%s' Failed\n", KernelEntry->name))
      return nullptr;

    DP("Entry point " DPxMOD " maps to %s (" DPxMOD ")\n", DPxPTR(KernelEntry),
       KernelEntry->name, DPxPTR(Func));

    StaticGlobalTy<llvm::omp::OMPTgtExecModeFlags> ExecModeGlobal(
        KernelEntry->name, "_exec_mode");
    if (!getPlugin().getGlobalHandler().readGlobalFromImage(*this,
                                                            ExecModeGlobal)) {
      INFO(OMP_INFOTYPE_DATA_TRANSFER, DeviceId,
           "Failed to read execution mode for %s, defaulting to SPMD.",
           KernelEntry->name);
      ExecModeGlobal.setValue(llvm::omp::OMP_TGT_EXEC_MODE_SPMD);
    }

    CUDAKernelTy *CUDAKernel = getPlugin().allocate<CUDAKernelTy>();
    new (CUDAKernel)
        CUDAKernelTy(KernelEntry->name, ExecModeGlobal.getValue(), Func);
    return CUDAKernel;
  }

  StatusCode setContext() {
    CUresult Err = cuCtxSetCurrent(Context);
    if (checkResult(Err, "Error returned from cuCtxSetCurrent\n"))
      return StatusCode::FAIL;
    return StatusCode::OK;
  }

  CUstream getStream(AsyncInfoWrapperTy &AsyncInfoWrapper) {
    CUstream &Stream = AsyncInfoWrapper.getQueueAs<CUstream>();
    if (!Stream)
      Stream = CUDAStreamManager.getStream();
    return Stream;
  }

  CUcontext getCUDAContext() const { return Context; }
  CUmodule getCUDAModule() const { return Module; }
  CUdevice getCUDADevice() const { return Device; }

  StatusCode loadBinaryImpl() override {
    if (setContext())
      return StatusCode::FAIL;
    return StatusCode(cuModuleLoadDataEx(&Module, getImage()->ImageStart, 0,
                                         nullptr, nullptr));
  }

  void *allocate(size_t Size, void *, TargetAllocTy Kind) override {
    if (Size == 0)
      return nullptr;

    if (setContext())
      return nullptr;

    CUresult Err;
    void *MemAlloc = nullptr;
    switch (Kind) {
    case TARGET_ALLOC_DEFAULT:
    case TARGET_ALLOC_DEVICE:
      CUdeviceptr DevicePtr;
      Err = cuMemAlloc(&DevicePtr, Size);
      MemAlloc = (void *)DevicePtr;
      if (checkResult(Err, "Error returned from cuMemAlloc\n"))
        return nullptr;
      break;
    case TARGET_ALLOC_HOST:
      void *HostPtr;
      Err = cuMemAllocHost(&HostPtr, Size);
      MemAlloc = HostPtr;
      if (checkResult(Err, "Error returned from cuMemAllocHost\n"))
        return nullptr;
      HostPinnedAllocs[MemAlloc] = Kind;
      break;
    case TARGET_ALLOC_SHARED:
      CUdeviceptr SharedPtr;
      Err = cuMemAllocManaged(&SharedPtr, Size, CU_MEM_ATTACH_GLOBAL);
      MemAlloc = (void *)SharedPtr;
      if (checkResult(Err, "Error returned from cuMemAllocManaged\n"))
        return nullptr;
      break;
    }

    return MemAlloc;
  }

  int free(void *TgtPtr) override {
    if (setContext())
      return OFFLOAD_FAIL;

    // Host pinned memory must be freed differently.
    TargetAllocTy Kind =
        (HostPinnedAllocs.find(TgtPtr) == HostPinnedAllocs.end())
            ? TARGET_ALLOC_DEFAULT
            : TARGET_ALLOC_HOST;
    CUresult Err;
    switch (Kind) {
    case TARGET_ALLOC_DEFAULT:
    case TARGET_ALLOC_DEVICE:
    case TARGET_ALLOC_SHARED:
      Err = cuMemFree((CUdeviceptr)TgtPtr);
      if (checkResult(Err, "Error returned from cuMemFree\n"))
        return OFFLOAD_FAIL;
      break;
    case TARGET_ALLOC_HOST:
      Err = cuMemFreeHost(TgtPtr);
      if (checkResult(Err, "Error returned from cuMemFreeHost\n"))
        return OFFLOAD_FAIL;
      break;
    }

    return OFFLOAD_SUCCESS;
  }

  StatusCode synchronizeImpl(__tgt_async_info &AsyncInfo) override {
    CUstream Stream = reinterpret_cast<CUstream>(AsyncInfo.Queue);
    CUresult Err = cuStreamSynchronize(Stream);

    // Once the stream is synchronized, return it to stream pool and reset
    // AsyncInfo. This is to make sure the synchronization only works for its
    // own tasks.
    CUDAStreamManager.returnStream(Stream);
    AsyncInfo.Queue = nullptr;

    return StatusCode(Err);
  }

  StatusCode dataSubmitImpl(void *TgtPtr, const void *HstPtr, int64_t Size,
                            AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    if (setContext())
      return StatusCode::FAIL;
    CUstream Stream = getStream(AsyncInfoWrapper);
    return StatusCode(
        cuMemcpyHtoDAsync((CUdeviceptr)TgtPtr, HstPtr, Size, Stream));
  }

  StatusCode dataRetrieveImpl(void *HstPtr, const void *TgtPtr, int64_t Size,
                              AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    if (setContext())
      return StatusCode::FAIL;
    CUstream Stream = getStream(AsyncInfoWrapper);
    return StatusCode(
        cuMemcpyDtoHAsync(HstPtr, (CUdeviceptr)TgtPtr, Size, Stream));
  }

  StatusCode dataExchangeImpl(const void *SrcPtr, int32_t DstDevId,
                              void *DstPtr, int64_t Size,
                              AsyncInfoWrapperTy &AsyncInfoWrapper) override;

  /// Event API
  ///{
  StatusCode createEventImpl(void **EventPtrStorage) override {
    CUevent *Event = reinterpret_cast<CUevent *>(EventPtrStorage);
    return StatusCode(cuEventCreate(Event, CU_EVENT_DEFAULT));
  }
  StatusCode destroyEventImpl(void *EventPtr) override {
    CUevent Event = reinterpret_cast<CUevent>(EventPtr);
    return StatusCode(cuEventDestroy(Event));
  }
  StatusCode recordEventImpl(void *EventPtr,
                             AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    CUevent Event = reinterpret_cast<CUevent>(EventPtr);
    return StatusCode(cuEventRecord(Event, getStream(AsyncInfoWrapper)));
  }
  StatusCode waitEventImpl(void *EventPtr,
                           AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    // We don't use CU_EVENT_WAIT_DEFAULT here as it is only available from
    // specific CUDA version, and defined as 0x0. In previous version, per CUDA
    // API document, that argument has to be 0x0.
    CUevent Event = reinterpret_cast<CUevent>(EventPtr);
    return StatusCode(cuStreamWaitEvent(getStream(AsyncInfoWrapper), Event, 0));
  }
  StatusCode syncEventImpl(void *EventPtr) override {
    CUevent Event = reinterpret_cast<CUevent>(EventPtr);
    return StatusCode(cuEventSynchronize(Event));
  }
  ///}

  void printInfoImpl() override {
#define BOOL2TEXT(b) ((b) ? "Yes" : "No")

    char TmpChar[1000];
    std::string TmpStr;
    size_t TmpSt;
    int TmpInt, TmpInt2, TmpInt3;

    cuDriverGetVersion(&TmpInt);
    printf("    CUDA Driver Version: \t\t%d \n", TmpInt);
    printf("    CUDA Device Number: \t\t%d \n", DeviceId);
    checkResult(cuDeviceGetName(TmpChar, 1000, Device),
                "Error returned from cuDeviceGetName\n");
    printf("    Device Name: \t\t\t%s \n", TmpChar);
    checkResult(cuDeviceTotalMem(&TmpSt, Device),
                "Error returned from cuDeviceTotalMem\n");
    printf("    Global Memory Size: \t\t%zu bytes \n", TmpSt);
    checkResult(cuDeviceGetAttribute(
                    &TmpInt, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, Device),
                "Error returned from cuDeviceGetAttribute\n");
    printf("    Number of Multiprocessors: \t\t%d \n", TmpInt);
    checkResult(
        cuDeviceGetAttribute(&TmpInt, CU_DEVICE_ATTRIBUTE_GPU_OVERLAP, Device),
        "Error returned from cuDeviceGetAttribute\n");
    printf("    Concurrent Copy and Execution: \t%s \n", BOOL2TEXT(TmpInt));
    checkResult(cuDeviceGetAttribute(
                    &TmpInt, CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, Device),
                "Error returned from cuDeviceGetAttribute\n");
    printf("    Total Constant Memory: \t\t%d bytes\n", TmpInt);
    checkResult(
        cuDeviceGetAttribute(
            &TmpInt, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, Device),
        "Error returned from cuDeviceGetAttribute\n");
    printf("    Max Shared Memory per Block: \t%d bytes \n", TmpInt);
    checkResult(
        cuDeviceGetAttribute(
            &TmpInt, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, Device),
        "Error returned from cuDeviceGetAttribute\n");
    printf("    Registers per Block: \t\t%d \n", TmpInt);
    checkResult(
        cuDeviceGetAttribute(&TmpInt, CU_DEVICE_ATTRIBUTE_WARP_SIZE, Device),
        "Error returned from cuDeviceGetAttribute\n");
    printf("    Warp Size: \t\t\t\t%d Threads \n", TmpInt);
    checkResult(cuDeviceGetAttribute(
                    &TmpInt, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, Device),
                "Error returned from cuDeviceGetAttribute\n");
    printf("    Maximum Threads per Block: \t\t%d \n", TmpInt);
    checkResult(cuDeviceGetAttribute(
                    &TmpInt, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, Device),
                "Error returned from cuDeviceGetAttribute\n");
    checkResult(cuDeviceGetAttribute(
                    &TmpInt2, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, Device),
                "Error returned from cuDeviceGetAttribute\n");
    checkResult(cuDeviceGetAttribute(
                    &TmpInt3, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, Device),
                "Error returned from cuDeviceGetAttribute\n");
    printf("    Maximum Block Dimensions: \t\t%d, %d, %d \n", TmpInt, TmpInt2,
           TmpInt3);
    checkResult(cuDeviceGetAttribute(
                    &TmpInt, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, Device),
                "Error returned from cuDeviceGetAttribute\n");
    checkResult(cuDeviceGetAttribute(
                    &TmpInt2, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, Device),
                "Error returned from cuDeviceGetAttribute\n");
    checkResult(cuDeviceGetAttribute(
                    &TmpInt3, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, Device),
                "Error returned from cuDeviceGetAttribute\n");
    printf("    Maximum Grid Dimensions: \t\t%d x %d x %d \n", TmpInt, TmpInt2,
           TmpInt3);
    checkResult(
        cuDeviceGetAttribute(&TmpInt, CU_DEVICE_ATTRIBUTE_MAX_PITCH, Device),
        "Error returned from cuDeviceGetAttribute\n");
    printf("    Maximum Memory Pitch: \t\t%d bytes \n", TmpInt);
    checkResult(cuDeviceGetAttribute(
                    &TmpInt, CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT, Device),
                "Error returned from cuDeviceGetAttribute\n");
    printf("    Texture Alignment: \t\t\t%d bytes \n", TmpInt);
    checkResult(
        cuDeviceGetAttribute(&TmpInt, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, Device),
        "Error returned from cuDeviceGetAttribute\n");
    printf("    Clock Rate: \t\t\t%d kHz\n", TmpInt);
    checkResult(cuDeviceGetAttribute(
                    &TmpInt, CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT, Device),
                "Error returned from cuDeviceGetAttribute\n");
    printf("    Execution Timeout: \t\t\t%s \n", BOOL2TEXT(TmpInt));
    checkResult(
        cuDeviceGetAttribute(&TmpInt, CU_DEVICE_ATTRIBUTE_INTEGRATED, Device),
        "Error returned from cuDeviceGetAttribute\n");
    printf("    Integrated Device: \t\t\t%s \n", BOOL2TEXT(TmpInt));
    checkResult(cuDeviceGetAttribute(
                    &TmpInt, CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, Device),
                "Error returned from cuDeviceGetAttribute\n");
    printf("    Can Map Host Memory: \t\t%s \n", BOOL2TEXT(TmpInt));
    checkResult(
        cuDeviceGetAttribute(&TmpInt, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, Device),
        "Error returned from cuDeviceGetAttribute\n");
    if (TmpInt == CU_COMPUTEMODE_DEFAULT)
      TmpStr = "DEFAULT";
    else if (TmpInt == CU_COMPUTEMODE_PROHIBITED)
      TmpStr = "PROHIBITED";
    else if (TmpInt == CU_COMPUTEMODE_EXCLUSIVE_PROCESS)
      TmpStr = "EXCLUSIVE PROCESS";
    else
      TmpStr = "unknown";
    printf("    Compute Mode: \t\t\t%s \n", TmpStr.c_str());
    checkResult(cuDeviceGetAttribute(
                    &TmpInt, CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS, Device),
                "Error returned from cuDeviceGetAttribute\n");
    printf("    Concurrent Kernels: \t\t%s \n", BOOL2TEXT(TmpInt));
    checkResult(
        cuDeviceGetAttribute(&TmpInt, CU_DEVICE_ATTRIBUTE_ECC_ENABLED, Device),
        "Error returned from cuDeviceGetAttribute\n");
    printf("    ECC Enabled: \t\t\t%s \n", BOOL2TEXT(TmpInt));
    checkResult(cuDeviceGetAttribute(
                    &TmpInt, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, Device),
                "Error returned from cuDeviceGetAttribute\n");
    printf("    Memory Clock Rate: \t\t\t%d kHz\n", TmpInt);
    checkResult(
        cuDeviceGetAttribute(
            &TmpInt, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, Device),
        "Error returned from cuDeviceGetAttribute\n");
    printf("    Memory Bus Width: \t\t\t%d bits\n", TmpInt);
    checkResult(cuDeviceGetAttribute(&TmpInt, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
                                     Device),
                "Error returned from cuDeviceGetAttribute\n");
    printf("    L2 Cache Size: \t\t\t%d bytes \n", TmpInt);
    checkResult(cuDeviceGetAttribute(
                    &TmpInt, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
                    Device),
                "Error returned from cuDeviceGetAttribute\n");
    printf("    Max Threads Per SMP: \t\t%d \n", TmpInt);
    checkResult(cuDeviceGetAttribute(
                    &TmpInt, CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT, Device),
                "Error returned from cuDeviceGetAttribute\n");
    printf("    Async Engines: \t\t\t%s (%d) \n", BOOL2TEXT(TmpInt), TmpInt);
    checkResult(cuDeviceGetAttribute(
                    &TmpInt, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, Device),
                "Error returned from cuDeviceGetAttribute\n");
    printf("    Unified Addressing: \t\t%s \n", BOOL2TEXT(TmpInt));
    checkResult(cuDeviceGetAttribute(
                    &TmpInt, CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY, Device),
                "Error returned from cuDeviceGetAttribute\n");
    printf("    Managed Memory: \t\t\t%s \n", BOOL2TEXT(TmpInt));
    checkResult(
        cuDeviceGetAttribute(
            &TmpInt, CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, Device),
        "Error returned from cuDeviceGetAttribute\n");
    printf("    Concurrent Managed Memory: \t\t%s \n", BOOL2TEXT(TmpInt));
    checkResult(
        cuDeviceGetAttribute(
            &TmpInt, CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED, Device),
        "Error returned from cuDeviceGetAttribute\n");
    printf("    Preemption Supported: \t\t%s \n", BOOL2TEXT(TmpInt));
    checkResult(cuDeviceGetAttribute(
                    &TmpInt, CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH, Device),
                "Error returned from cuDeviceGetAttribute\n");
    printf("    Cooperative Launch: \t\t%s \n", BOOL2TEXT(TmpInt));
    checkResult(cuDeviceGetAttribute(
                    &TmpInt, CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD, Device),
                "Error returned from cuDeviceGetAttribute\n");
    printf("    Multi-Device Boars: \t\t%s \n", BOOL2TEXT(TmpInt));
    checkResult(
        cuDeviceGetAttribute(
            &TmpInt, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, Device),
        "Error returned from cuDeviceGetAttribute\n");
    checkResult(
        cuDeviceGetAttribute(
            &TmpInt2, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, Device),
        "Error returned from cuDeviceGetAttribute\n");
    printf("    Compute Capabilities: \t\t%d%d \n", TmpInt, TmpInt2);

#undef BOOL2TEXT
  }

  StatusCode getDeviceStackSize(uint64_t &Value) {
    return getCtxLimit(CU_LIMIT_STACK_SIZE, Value);
  }
  StatusCode setDeviceStackSize(uint64_t Value) {
    return setCtxLimit(CU_LIMIT_STACK_SIZE, Value);
  }
  StatusCode getDeviceHeapSize(int64_t &Value) {
    return setCtxLimit(CU_LIMIT_MALLOC_HEAP_SIZE, Value);
  }
  StatusCode setDeviceHeapSize(int32_t Value) {
    return setCtxLimit(CU_LIMIT_MALLOC_HEAP_SIZE, Value);
  }

  StatusCode setCtxLimit(uint32_t Kind, uint64_t Value) {
    return StatusCode(cuCtxSetLimit((CUlimit)Kind, Value));
  }
  StatusCode getCtxLimit(uint32_t Kind, uint64_t &Value) {
    return StatusCode(cuCtxGetLimit(&Value, (CUlimit)Kind));
  }

  StatusCode getDeviceAttr(uint32_t Kind, uint32_t &Value) {
    // TODO: Warn if the new value is larget than the old.
    CUresult Err =
        cuDeviceGetAttribute((int *)&Value, (CUdevice_attribute)Kind, Device);
    if (checkResult(Err, "Error returned from cuDeviceGetAttribute\n"))
      return StatusCode::FAIL;
    return StatusCode::OK;
  }

private:
  std::unordered_map<void *, TargetAllocTy> HostPinnedAllocs;

  CUDAStreamManagerTy CUDAStreamManager;
  CUcontext Context = nullptr;
  CUdevice Device = 0;
  CUmodule Module = nullptr;
};

CUDAStreamManagerTy ::CUDAStreamManagerTy(CUDADeviceTy &CUDADevice)
    : StreamManagerTy<CUstream>(CUDADevice.DeviceId), CUDADevice(CUDADevice) {}

CUDAStreamManagerTy::~CUDAStreamManagerTy() {
  if (CUDADevice.setContext())
    return;

  for (CUstream &S : StreamPool) {
    if (S)
      checkResult(cuStreamDestroy(S), "Error returned from cuStreamDestroy\n");
  }
}

StatusCode CUDAStreamManagerTy::resizeStreamPoolImpl(int32_t OldSize,
                                                     int32_t NewSize) {
  if (CUDADevice.setContext())
    return StatusCode::FAIL;

  for (int32_t I = OldSize; I < NewSize; ++I) {
    if (checkResult(cuStreamCreate(&StreamPool[I], CU_STREAM_NON_BLOCKING),
                    "Error returned from cuStreamCreate\n"))
      return StatusCode::FAIL;
  }
  return StatusCode::OK;
}

void *CUDAKernelTy::argumentPrepareImpl(GenericDeviceTy &GenericDevice,void **ArgPtrs, ptrdiff_t *ArgOffsets,
                                        int32_t NumArgs, AsyncInfoWrapperTy &AsyncInfoWrapper) {
  Args.resize(NumArgs);
  Ptrs.resize(NumArgs);

  for (int I = 0; I < NumArgs; ++I) {
    Ptrs[I] = (void *)((intptr_t)ArgPtrs[I] + ArgOffsets[I]);
    Args[I] = &Ptrs[I];
  }

  return &Args[0];
}

StatusCode CUDAKernelTy::launchImpl(GenericDeviceTy &GenericDevice,
                                    int32_t NumThreads, int32_t NumBlocks,
                                    int32_t DynamicMemorySize, void *KernelArgs,
                                    AsyncInfoWrapperTy &AsyncInfoWrapper) {
  CUDADeviceTy &CUDADevice = static_cast<CUDADeviceTy &>(GenericDevice);
  CUstream Stream = CUDADevice.getStream(AsyncInfoWrapper);
  return StatusCode(cuLaunchKernel(Func, NumBlocks, /* gridDimY */ 1,
                                   /* gridDimZ */ 1, NumThreads,
                                   /* blockDimY */ 1, /* blockDimZ */ 1,
                                   DynamicMemorySize, Stream,
                                   (void **)KernelArgs, nullptr));
}

struct CUDAPluginTy final : public GenericPluginTy {

  // This class should not be copied
  CUDAPluginTy(const CUDAPluginTy &) = delete;
  CUDAPluginTy(CUDAPluginTy &&) = delete;

  CUDAPluginTy() : GenericPluginTy() {
    CUresult Err = cuInit(0);
    if (Err == CUDA_ERROR_INVALID_HANDLE) {
      // Can't call cuGetErrorString if dlsym failed
      INFO(OMP_INFOTYPE_PLUGIN_KERNEL, -1,
           "Failed to load CUDA shared library\n");
      return;
    }
    if (checkResult(Err, "Error returned from cuInit\n"))
      return;

    int NumDevices;
    Err = cuDeviceGetCount(&NumDevices);
    if (checkResult(Err, "Error returned from cuDeviceGetCount\n"))
      return;

    if (NumDevices == 0) {
      INFO(OMP_INFOTYPE_PLUGIN_KERNEL, -1,
           "There are no devices supporting CUDA.\n");
      return;
    }
    GenericPluginTy::init(NumDevices);
  }

  ~CUDAPluginTy() {}

  uint16_t getMagicElfBits() const override { return /* EM_CUDA */ 190; }

  CUDADeviceTy &getDevice(int32_t DeviceId) override {
    assert(isValidDeviceId(DeviceId) && "Device Id is invalid");
    return Devices[DeviceId];
  }

private:
  SmallVector<CUDADeviceTy, 8> Devices;
};

StatusCode
CUDADeviceTy::dataExchangeImpl(const void *SrcPtr, int32_t DstDevId,
                               void *DstPtr, int64_t Size,
                               AsyncInfoWrapperTy &AsyncInfoWrapper) {
  if (setContext())
    return StatusCode::FAIL;

  CUstream Stream = getStream(AsyncInfoWrapper);
  // If they are two devices, we try peer to peer copy first
  if (DeviceId == DstDevId)
    return StatusCode(cuMemcpyDtoDAsync((CUdeviceptr)DstPtr,
                                        (CUdeviceptr)SrcPtr, Size, Stream));

  int CanAccessPeer = 0;
  CUresult Err;
  Err = cuDeviceCanAccessPeer(&CanAccessPeer, DeviceId, DstDevId);
  if (checkResult(Err,
                  "Error returned from cuDeviceCanAccessPeer. src = %" PRId32
                  ", dst = %" PRId32 "\n",
                  DeviceId, DstDevId))
    return StatusCode(cuMemcpyDtoDAsync((CUdeviceptr)DstPtr,
                                        (CUdeviceptr)SrcPtr, Size, Stream));

  if (!CanAccessPeer) {
    DP("P2P memcpy not supported so fall back to D2D memcpy");
    return StatusCode(cuMemcpyDtoDAsync((CUdeviceptr)DstPtr,
                                        (CUdeviceptr)SrcPtr, Size, Stream));
  }

  CUcontext DstContext = getCUDAPlugin().getDevice(DstDevId).getCUDAContext();
  Err = cuCtxEnablePeerAccess(DstContext, 0);
  if (checkResult(Err,
                  "Error returned from cuCtxEnablePeerAccess. src = %" PRId32
                  ", dst = %" PRId32 "\n",
                  DeviceId, DstDevId))
    return StatusCode(cuMemcpyDtoDAsync((CUdeviceptr)DstPtr,
                                        (CUdeviceptr)SrcPtr, Size, Stream));

  Err = cuMemcpyPeerAsync((CUdeviceptr)DstPtr, DstContext, (CUdeviceptr)SrcPtr,
                          Context, Size, Stream);
  if (Err == CUDA_SUCCESS)
    return StatusCode::OK;

  checkResult(Err,
              "Error returned from cuMemcpyPeerAsync. src_ptr = " DPxMOD
              ", src_id =%" PRId32 ", dst_ptr = " DPxMOD ", dst_id =%" PRId32
              "\n",
              DPxPTR(SrcPtr), DeviceId, DPxPTR(DstPtr), DstDevId);
  return StatusCode(cuMemcpyDtoDAsync((CUdeviceptr)DstPtr, (CUdeviceptr)SrcPtr,
                                      Size, Stream));
}

int32_t llvm::omp::plugin::GlobalHandlerTy::getGlobalMetadataFromDevice(
    GenericDeviceTy &Device, GlobalTy &DeviceGlobal) {
  CUDADeviceTy &CUDADevice = static_cast<CUDADeviceTy &>(Device);
  CUmodule Module = CUDADevice.getCUDAModule();
  CUdeviceptr CUPtr;
  size_t CUSize;
  const char *Name = DeviceGlobal.getName().c_str();
  CUresult Err = cuModuleGetGlobal(&CUPtr, &CUSize, Module, Name);
  if (checkResult(Err, "Loading global '%s' Failed\n", Name))
    return OFFLOAD_FAIL;
  if (CUSize != DeviceGlobal.getSize()) {
    DP("Loading global '%s' - size mismatch (%zd != %zd)\n", Name, CUSize,
       DeviceGlobal.getSize());
    return OFFLOAD_FAIL;
  }

  DP("Entry point " DPxMOD " maps to global %s (" DPxMOD ")\n",
     DPxPTR(E - HostBegin), Name, DPxPTR(CUPtr));
  DeviceGlobal.setPtr(reinterpret_cast<void *>(CUPtr));
  return OFFLOAD_SUCCESS;
}

/// Expose the plugin to the generic part. Not ideal but not the worst.
namespace {
CUDAPluginTy Plugin;
} // namespace

GenericPluginTy &llvm::omp::plugin::getPlugin() { return Plugin; }
CUDAPluginTy &getCUDAPlugin() {
  return static_cast<CUDAPluginTy &>(getPlugin());
}
