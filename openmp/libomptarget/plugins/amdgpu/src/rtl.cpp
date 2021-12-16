//===--- amdgpu/src/rtl.cpp --------------------------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RTL for AMD hsa machine
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cstdint>
#include <string>
#include <unordered_map>

#include "Debug.h"
#include "hsa.h"
#include "hsa_ext_amd.h"
#include "omptarget.h"

#define TARGET_NAME AMDGPU
#define DEBUG_PREFIX "Target " GETNAME(TARGET_NAME) " RTL"

#include "DeviceInterface.h"
#include "GlobalHandler.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/Frontend/OpenMP/OMPGridValues.h"

using namespace llvm;
using namespace omp;
using namespace plugin;

static_assert(int(Status::SUCCESS) == int(HSA_STATUS_SUCCESS),
              "SUCCESS status enum should match HSA SUCCESS enum.");
static bool operator==(hsa_status_t HS, Status S) {
  return HS == hsa_status_t(S);
}
static bool operator==(Status S, hsa_status_t HS) {
  return HS == hsa_status_t(S);
}

int32_t getErrorString(int32_t ErrorCode, const char **ErrorString) {
  return Status(hsa_status_string(hsa_status_t(ErrorCode), ErrorString));
}

template <typename CBArgTy, typename IterFnTy, typename IterFnArgTy,
          typename CBTy>
static hsa_status_t iter(IterFnTy IterFn, IterFnArgTy IterFnArg, CBTy CB) {
  auto Wrapper = [](CBArgTy Arg, void *data) -> hsa_status_t {
    CBTy *CB = static_cast<CBTy *>(data);
    return (*CB)(Arg);
  };
  return IterFn(IterFnArg, Wrapper, static_cast<void *>(&CB));
}

/// Forward declarations for all specialized data structures.
struct AMDGPUKernelTy;
struct AMDGPUDeviceTy;
struct AMDGPUPluginTy;
struct AMDGPUStreamManagerTy;

template <typename... ArgsTy>
static bool checkResult(hsa_status_t Err, const char *ErrMsg, ArgsTy... Args) {
  if (Err == Status::SUCCESS)
    return false;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wformat-security"
  REPORT(ErrMsg, Args...);
#pragma clang diagnostic pop

  const char *ErrStr = nullptr;
  hsa_status_t ErrStrStatus =
      int32_t(Err) == -1 ? HSA_STATUS_ERROR : getErrorString(Err, &ErrStr);
  if (int32_t(ErrStrStatus) == -1) {
    REPORT("Unrecognized " GETNAME(TARGET_NAME) " error code: %d\n", Err);
  } else if (ErrStrStatus == Status::SUCCESS) {
    REPORT(GETNAME(TARGET_NAME) " error is: %s\n", ErrStr);
  } else {
    REPORT("Unresolved " GETNAME(TARGET_NAME) " error code: %d\n"
                                              "Unsuccessful in getting error "
                                              "string, error code: %d\n",
           Err, ErrStrStatus);
  }
  return true;
}

struct AMDGPUMemoryPoolTy {

  StatusCode init() {
    bool AllocAllowed = false;
    hsa_status_t Err = hsa_amd_memory_pool_get_info(
        MemoryPool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED,
        &AllocAllowed);
    if (checkResult(
            Err,
            "Memory pool allocation policy is static or unknown, skip it.") ||
        !AllocAllowed)
      return StatusCode::FAIL;

    Err = hsa_amd_memory_pool_get_info(MemoryPool,
                                       HSA_AMD_MEMORY_POOL_INFO_SIZE, &Size);
    if (checkResult(Err, "Memory pool size is 0 or unknown, skip it.") ||
        Size == 0)
      return StatusCode::FAIL;

    Err = hsa_amd_memory_pool_get_info(
        MemoryPool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &GlobalFlags);
    if (checkResult(Err, "Memory pool characteristics unknown, skip it."))
      return StatusCode::FAIL;

    return StatusCode::OK;
  }

  bool isFineGrained() const {
    return GlobalFlags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED;
  }
  bool isCoarseGrained() const {
    return GlobalFlags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED;
  }
  bool isKernargInit() const {
    return GlobalFlags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT;
  }

private:
  size_t Size = 0;
  uint32_t GlobalFlags = 0;
  hsa_amd_memory_pool_t MemoryPool;
};

struct AMDGPUStreamTy {
  AMDGPUStreamTy() {}

  StatusCode init(AMDGPUDeviceTy &AMDGPUDevice,
                  AMDGPUMemoryPoolTy &KernArgPool);

  hsa_signal_t &peekSignal() { return Signals[SignalIndex]; }
  hsa_signal_t &pushSignal() {
    SignalIndex += 1;
    Signals.resize(SignalIndex);
    hsa_signal_store_relaxed(Signals[SignalIndex], 1);
    return Signals[SignalIndex];
  }

  void reset() { SignalIndex = -1; }

  static uint64_t acquire_available_packet_id(hsa_queue_t *queue) {
    uint64_t packet_id = hsa_queue_add_write_index_relaxed(queue, 1);
    bool full = true;
    while (full) {
      full = packet_id >=
             (queue->size + hsa_queue_load_read_index_scacquire(queue));
    }
    return packet_id;
  }

  hsa_kernel_dispatch_packet_t *getKernelPacket(uint64_t &PacketId) {
    PacketId = acquire_available_packet_id(Queue);

    const uint32_t Mask = Queue->size - 1; // size is a power of 2
    hsa_kernel_dispatch_packet_t *Packet =
        (hsa_kernel_dispatch_packet_t *)Queue->base_address + (PacketId & Mask);

    Packet->completion_signal = pushSignal();

    return Packet;
  }

private:
  int32_t SignalIndex = -1;
  std::deque<hsa_signal_t> Signals;
  hsa_queue_t *Queue;
  AMDGPUMemoryPoolTy *KernArgPoolPtr = nullptr;
};

/// Forward declaration to access the singleton plugin in lieu of a header.
static AMDGPUPluginTy &getAMDGPUPlugin();

struct AMDGPUKernelTy : public GenericKernelTy {
  AMDGPUKernelTy(const char *Name, OMPTgtExecModeFlags ExecutionMode)
      : GenericKernelTy(Name, ExecutionMode) {}

  void initImpl(GenericDeviceTy &GenericDevice) override;
  void *argumentPrepareImpl(GenericDeviceTy &GenericDevice, void **ArgPtrs,
                            ptrdiff_t *ArgOffsets, int32_t NumArgs,
                            AsyncInfoWrapperTy &AsyncInfoWrapper) override;
  StatusCode launchImpl(GenericDeviceTy &GenericDevice, int32_t NumThreads,
                        int32_t NumBlocks, int32_t DynamicMemorySize,
                        void *KernelArgs,
                        AsyncInfoWrapperTy &AsyncInfoWrapper) override;

  // Heuristic parameters used for kernel launch
  // Number of teams per CU to allow scheduling flexibility
  static const unsigned DefaultTeamsPerCU = 4;
  int32_t getDefaultBlockCount(GenericDeviceTy &GenericDevice) const override {
    return GenericDevice.getDefaultBlockCount() * DefaultTeamsPerCU;
  }
  int32_t getDefaultThreadCount(GenericDeviceTy &GenericDevice) const override {
    return GenericDevice.getDefaultThreadCount();
  }

private:
  hsa_executable_symbol_t Symbol;
  uint64_t Object;
  uint32_t KernelArgumentSegmentSize;
  uint32_t GroupSegmentSize;
  uint32_t PrivateSegmentSize;
};

struct AMDGPUStreamManagerTy : public StreamManagerTy<AMDGPUStreamTy *> {
  AMDGPUStreamManagerTy(AMDGPUDeviceTy &AMDGPUDevice);

  ~AMDGPUStreamManagerTy();

  StatusCode resizeStreamPoolImpl(int32_t OldSize, int32_t NewSize) override;

private:
  std::deque<AMDGPUStreamTy> StreamStorage;
  AMDGPUDeviceTy &AMDGPUDevice;
};

/// Structure contains per-device data.
struct AMDGPUDeviceTy : public GenericDeviceTy {

  AMDGPUDeviceTy(int32_t DeviceId)
      : GenericDeviceTy(DeviceId, NVPTXGridValues), AMDGPUStreamManager(*this) {
  }
  ~AMDGPUDeviceTy() {}

  StatusCode initImpl(GenericPluginTy &Plugin) override {

    // Initialize stream pool
    if (AMDGPUStreamManager.init())
      return StatusCode::FAIL;

    if (initMemoryPools())
      return StatusCode::FAIL;

    // Query attributes to determine number of threads/block and blocks/grid.
    if (getDeviceAttr<uint32_t>(HSA_AGENT_INFO_GRID_MAX_DIM,
                                GridValues.GV_Max_Teams))
      return StatusCode::FAIL;
    if (getDeviceAttr<uint16_t, 3>(HSA_AGENT_INFO_WORKGROUP_MAX_DIM,
                                   GridValues.GV_Max_WG_Size))
      return StatusCode::FAIL;
    if (getDeviceAttr<uint32_t>(HSA_AGENT_INFO_WAVEFRONT_SIZE,
                                GridValues.GV_Warp_Size))
      return StatusCode::FAIL;
    if (getDeviceAttr<uint32_t>(HSA_AGENT_INFO_QUEUE_MAX_SIZE, QueueSize))
      return StatusCode::FAIL;
    if (getDeviceAttr<uint32_t>(HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT,
                                ComputeUnitCount))
      return StatusCode::FAIL;
    if (getDeviceAttr<char, 64, std ::string, /* AsPtr */ true>(
            HSA_AGENT_INFO_NAME, GPUName))
      return StatusCode::FAIL;

    return StatusCode::OK;
  }

  StatusCode initMemoryPools() {
    auto CB = [&](hsa_amd_memory_pool_t MemoryPool) -> hsa_status_t {
      MemoryPools.emplace_back(MemoryPool);
      if (MemoryPools.back().init())
        MemoryPools.pop_back();
      return HSA_STATUS_SUCCESS;
    };
    hsa_status_t Err = iter<hsa_amd_memory_pool_t>(
        hsa_amd_agent_iterate_memory_pools, Agent, CB);
    if (checkResult(Err, "Failure to initialize memory pools\n"))
      return StatusCode::FAIL;

    if (MemoryPools.empty()) {
      REPORT("Failure to initialize a single device memory pool\n");
      return StatusCode::FAIL;
    }
    return StatusCode::OK;
  }

  GenericKernelTy *
  constructKernelEntry(const __tgt_offload_entry *KernelEntry) override {

    GlobalHandlerTy GlobalHandler = getPlugin().getGlobalHandler();
    // Read execution mode global from the binary
    StaticGlobalTy<llvm::omp::OMPTgtExecModeFlags> ExecModeGlobal(
        KernelEntry->name, "_exec_mode");
    if (!GlobalHandler.readGlobalFromImage(*this, ExecModeGlobal)) {
      INFO(OMP_INFOTYPE_DATA_TRANSFER, DeviceId,
           "Failed to read execution mode for %s, defaulting to SPMD.",
           KernelEntry->name);
      ExecModeGlobal.setValue(llvm::omp::OMP_TGT_EXEC_MODE_SPMD);
    }

    AMDGPUKernelTy *AMDKernel = getPlugin().allocate<AMDGPUKernelTy>();
    new (AMDKernel)
        AMDGPUKernelTy(KernelEntry->name, ExecModeGlobal.getValue(), Func);
    return AMDKernel;
  }

  AMDGPUStreamTy &getStream(const AsyncInfoWrapperTy &AsyncInfoWrapper) {
    AMDGPUStreamTy *&Stream = AsyncInfoWrapper.getQueueAs<AMDGPUStreamTy *>();
    if (!Stream)
      Stream = AMDGPUStreamManager.getStream();
    return *Stream;
  }

  StatusCode loadBinaryImpl() override {
    GlobalHandlerTy GlobalHandler = getPlugin().getGlobalHandler();

    hsa_status_t Err;
    Err = hsa_agent_get_info(Agent, HSA_AGENT_INFO_PROFILE, &Profile);
    if (checkResult(Err, "Error returned from hsa_agent_get_info\n"))
      return StatusCode(Err);

    Err = hsa_executable_create(Profile, HSA_EXECUTABLE_STATE_UNFROZEN, "",
                                &Executable);
    if (checkResult(Err, "Error returned from hsa_executable_create\n"))
      return StatusCode(Err);

    Err = hsa_code_object_deserialize(getImage()->ImageStart, getImageSize(),
                                      NULL, &CodeObject);
    if (checkResult(Err, "Error returned from hsa_code_object_deserialize\n"))
      return StatusCode(Err);

    Err = hsa_executable_load_code_object(Executable, Agent, CodeObject, NULL);
    if (checkResult(Err,
                    "Error returned from hsa_executable_load_code_object\n"))
      return StatusCode(Err);

    Err = hsa_executable_freeze(Executable, "");
    if (checkResult(Err, "Error returned from hsa_executable_freeze\n"))
      return StatusCode(Err);

    return StatusCode::OK;
  }

  void *allocate(size_t Size, void *, TargetAllocTy Kind) override {
    if (Size == 0)
      return nullptr;

    int32_t MemoryPoolIdx = 0;
    // For SHARED allocations we pick a fine grained memory pool.
    if (Kind == TARGET_ALLOC_SHARED) {
      int32_t MemoryPoolSize = MemoryPools.size();
      while (MemoryPoolIdx < MemoryPoolSize) {
        if (MemoryPools[MemoryPoolIdx].isFineGrained())
          break;
        ++MemoryPoolIdx;
      }
      if (MemoryPoolIdx == MemoryPoolSize) {
        REPORT("No fine grained memory pool found for \"managed\" allocation");
        return nullptr;
      }
    }

    hsa_status_t Err;
    void *MemAlloc = nullptr;
    switch (Kind) {
    case TARGET_ALLOC_DEFAULT:
    case TARGET_ALLOC_DEVICE:
    case TARGET_ALLOC_HOST:
    case TARGET_ALLOC_SHARED:
      Err = hsa_amd_memory_pool_allocate(MemoryPools[MemoryPoolIdx], Size, 0,
                                         &MemAlloc);
      if (checkResult(Err,
                      "Error returned from hsa_amd_memory_pool_allocate\n"))
        return nullptr;
      break;
    }

    if (Kind == TARGET_ALLOC_HOST) {
      // TODO:
      Err = allow_access_to_all_gpu_agents();
      if (checkResult(Err, "Error returned from ...\n"))
        return OFFLOAD_FAIL;
      break;
    }

    return MemAlloc;
  }

  int free(void *TgtPtr) override {
    if (setContext())
      return OFFLOAD_FAIL;

    hsa_status_t Err;
    switch (Kind) {
    case TARGET_ALLOC_DEFAULT:
    case TARGET_ALLOC_DEVICE:
    case TARGET_ALLOC_SHARED:
    case TARGET_ALLOC_HOST:
      Err = hsa_amd_memory_pool_free(TgtPtr);
      if (checkResult(Err, "Error returned from hsa_amd_memory_pool_free\n"))
        return OFFLOAD_FAIL;
      break;
    }

    return OFFLOAD_SUCCESS;
  }

  StatusCode synchronizeImpl(__tgt_async_info &AsyncInfo) override {
    StatusCode SC = StatusCode::OK;
    AMDGPUStreamTy &Stream = getStream(AsyncInfoWrapperTy(SC, *this, &AsyncInfo));
    hsa_signal_t Signal = Stream.peekSignal();
    while (hsa_signal_wait_scacquire(Signal, HSA_SIGNAL_CONDITION_EQ, 0,
                                     UINT64_MAX, HSA_WAIT_STATE_BLOCKED) != 0)
      ;

    // Once the stream is synchronized, return it to stream pool and reset
    // AsyncInfo. This is to make sure the synchronization only works for its
    // own tasks.
    Stream.reset();
    AMDGPUStreamManager.returnStream(&Stream);
    AsyncInfo.Queue = nullptr;

    return SC;
  }

  StatusCode dataCopy(void *DstPtr, hsa_agent_t DstAgent, const void *SrcPtr,
                      hsa_agent_t SrcAgent, int64_t Size,
                      AsyncInfoWrapperTy &AsyncInfoWrapper) {
    hsa_signal_t *CurrentSignal, NextSignal;
    getCurrentAndNextSignal(CurrentSignal, NextSignal);
    return StatusCode(hsa_amd_memory_async_copy(
        TgtPtr, Agent, HstPtr, Agent, Size, 1, CurrentSignal, NextSignal));
  }
  StatusCode dataSubmitImpl(void *TgtPtr, const void *HstPtr, int64_t Size,
                            AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    // TODO: Host agent?
    return dataCopy(TgtPtr, Agent, HstPtr, Agent, Size, AsyncInfoWrapper);
  }

  StatusCode dataRetrieveImpl(void *HstPtr, const void *TgtPtr, int64_t Size,
                              AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    // TODO: Host agent?
    return dataCopy(HstPtr, Agent, TgtPtr, Agent, Size, AsyncInfoWrapper);
  }

  StatusCode dataExchangeImpl(const void *SrcPtr, int32_t DstDevId,
                              void *DstPtr, int64_t Size,
                              AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    hsa_agent_t &DstAgent = getAMDGPUPlugin().getDevice(DstDevId).getAgent();
    return dataCopy(DstPtr, DstAgent, SrcPtr, Agent, Size, AsyncInfoWrapper);
  }

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
    // specific AMDGPU version, and defined as 0x0. In previous version, per
    // AMDGPU API document, that argument has to be 0x0.
    CUevent Event = reinterpret_cast<CUevent>(EventPtr);
    return StatusCode(cuStreamWaitEvent(getStream(AsyncInfoWrapper), Event, 0));
  }
  StatusCode syncEventImpl(void *EventPtr) override {
    hsa_signal_t Signal = reinterpret_cast<hsa_signal_t>(EventPtr);
    return StatusCode(hsa_signal_wait_acquire(Signal, HSA_SIGNAL_CONDITION_LT,
                                              1, uint64_t(-1),
                                              HSA_WAIT_STATE_ACTIVE));
  }
  ///}

  void printInfoImpl() override {
    // TODO
  }

  template <typename DataTy, unsigned NumElements = 1,
            typename ResultTy = uint32_t, bool AsPtr = false>
  StatusCode getDeviceAttr(uint32_t Kind, ResultTy &Value) {
    DataTy Data[NumElements];
    // TODO: Warn if the new value is larget than the old.
    hsa_status_t Err =
        hsa_agent_get_info(Agent, hsa_agent_info_t(Kind), &Data[0]);
    if (checkResult(Err, "Error returned from hsa_agent_get_info\n"))
      return StatusCode::FAIL;
    if (AsPtr)
      Value = &Data[0];
    else
      Value = Data[0];
    return StatusCode::OK;
  }

  hsa_agent_t getAgent() const { return Agent; }
  hsa_executable_t getExecutable() const { return Executable; }

  uint32_t getQueueSize() const { return QueueSize; }
  uint32_t getComputeUnitCount() const { return ComputeUnitCount; }
  const std::string getGPUName() const { return GPUName; }

  MemoryManagerTy &getKernelArgMemoryManager() {
    return *KernelArgMemoryManager;
  };

private:
  AMDGPUStreamManagerTy AMDGPUStreamManager;
  hsa_agent_t Agent;
  hsa_profile_t Profile;
  hsa_code_object_t CodeObject;
  hsa_executable_t Executable;
  hsa_executable_t HSAExecutable;

  std::string GPUName;
  uint32_t QueueSize;
  uint32_t ComputeUnitCount;

  SmallVector<AMDGPUMemoryPoolTy, 8> MemoryPools;

  std::unique_ptr<MemoryManagerTy> KernelArgMemoryManager;
};

AMDGPUStreamManagerTy ::AMDGPUStreamManagerTy(AMDGPUDeviceTy &AMDGPUDevice)
    : StreamManagerTy<AMDGPUStreamTy>(AMDGPUDevice.DeviceId),
      AMDGPUDevice(AMDGPUDevice) {}

AMDGPUStreamManagerTy::~AMDGPUStreamManagerTy() {}

StatusCode AMDGPUStreamManagerTy::resizeStreamPoolImpl(int32_t OldSize,
                                                       int32_t NewSize) {
  StreamStorage.resize(NewSize);
  for (int32_t I = OldSize; I < NewSize; ++I) {
    StreamStorage[I].init(AMDGPUDevice);
    StreamPool[I] = &StreamStorage[I];
  }
  return StatusCode::OK;
}

void AMDGPUKernelTy::initImpl(GenericDeviceTy &GenericDevice) {
  AMDGPUDeviceTy &AMDGPUDevice = static_cast<AMDGPUDeviceTy &>(GenericDevice);
  hsa_status_t Err;
  Err =
      hsa_executable_get_symbol_by_name(AMDGPUDevice.getExecutable(), getName(),
                                        &AMDGPUDevice.getAgent(), &Symbol);
  if (checkResult(Err,
                  "Error returned from hsa_executable_get_symbol_by_name\n"))
    return;

  std::pair<void*, hsa_executable_symbol_info_t> KernelInfos[] = {
      {&Object, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT},
      {&GroupSegmentSize, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE},
      {&PrivateSegmentSize, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE},
  };
  for (auto &It : KernelInfos) {
    Err = hsa_executable_symbol_get_info(Symbol, It.second, It.first);
    if (checkResult(Err,
                    "Error returned from hsa_executable_symbol_get_info\n"))
      return;
  }

  // MaxThreadCount = std::min(MaxThreadCount, MaxThreads);
}

void *AMDGPUKernelTy::argumentPrepareImpl(
    GenericDeviceTy &GenericDevice, void **ArgPtrs, ptrdiff_t *ArgOffsets,
    int32_t NumArgs, AsyncInfoWrapperTy &AsyncInfoWrapper) {
  AMDGPUDeviceTy &AMDGPUDevice = static_cast<AMDGPUDeviceTy &>(GenericDevice);
  AMDGPUStreamTy& Stream = AMDGPUDevice.getStream(AsyncInfoWrapper);

  void *&KernelArgMemory =
      KernelArgumentMemoryPerDevice[GenericDevice.DeviceId];
  if (!KernelArgMemory) {
    AMDGPUDeviceTy &AMDGPUDevice = static_cast<AMDGPUDeviceTy &>(GenericDevice);
    assert(KernelArgumentSegmentSize == NumArgs * sizeof(void *) &&
           "Inconsistency in the kernel argument size calculation");
    size_t NumBytes = NumArgs * (sizeof(void *)) + sizeof(impl_implicit_args_t);
    KernelArgMemory =
        AMDGPUDevice.getKernelArgMemoryManager().allocate(NumBytes, nullptr);
  }

  Ptrs.resize(NumArgs);
  for (int I = 0; I < NumArgs; ++I) {
    Ptrs[I] = (void *)((intptr_t)ArgPtrs[I] + ArgOffsets[I]);
    KernelArgMemory[I] = &Ptrs[I];
  }

  memset(&KernelArgMemory[NumArgs], 0, sizeof(impl_implicit_args_t));

  return KernelArgMemory;
}

StatusCode AMDGPUKernelTy::launchImpl(GenericDeviceTy &GenericDevice,
                                      int32_t NumThreads, int32_t NumBlocks,
                                      int32_t DynamicMemorySize,
                                      void *KernelArgs,
                                      AsyncInfoWrapperTy &AsyncInfoWrapper) {
  AMDGPUDeviceTy &AMDGPUDevice = static_cast<AMDGPUDeviceTy &>(GenericDevice);
  AMDGPUStreamTy& Stream = AMDGPUDevice.getStream(AsyncInfoWrapper);

  uint64_t &PacketId;
  hsa_kernel_dispatch_packet_t *Packet = Stream.getKernelPacket(PacketId);

  Packet->setup = UINT16_C(1) << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
  Packet->workgroup_size_x = NumThreads;
  Packet->workgroup_size_y = 1;
  Packet->workgroup_size_z = 1;
  Packet->reserved0 = 0;
  Packet->grid_size_x = NumBlocks;
  Packet->grid_size_y = 1;
  Packet->grid_size_z = 1;
  Packet->private_segment_size = KernelInfoEntry.private_segment_size;
  Packet->group_segment_size = KernelInfoEntry.group_segment_size;
  Packet->kernel_object = KernelInfoEntry.kernel_object;
  Packet->kernarg_address = KernelArgs;
  Packet->reserved2 = 0; // impl writes id_ here

  // Publish the Packet indicating it is ready to be processed
  uint16_t header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
  header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;
  __atomic_store_n((uint32_t *)Packet, header | (Packet->setup << 16),
                   __ATOMIC_RELEASE);

  // Since the packet is already published, its contents must not be
  // accessed any more
  hsa_signal_store_relaxed(queue->doorbell_signal, packet_id);

  DP("Kernel completed\n");
  return OFFLOAD_SUCCESS;
}

struct AMDGPUPluginTy final : public GenericPluginTy {

  // This class should not be copied
  AMDGPUPluginTy(const AMDGPUPluginTy &) = delete;
  AMDGPUPluginTy(AMDGPUPluginTy &&) = delete;

  AMDGPUPluginTy() : GenericPluginTy() {
    hsa_status_t Err = cuInit(0);
    if (Err == AMDGPU_ERROR_INVALID_HANDLE) {
      // Can't call cuGetErrorString if dlsym failed
      INFO(OMP_INFOTYPE_PLUGIN_KERNEL, -1,
           "Failed to load AMDGPU shared library\n");
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
           "There are no devices supporting AMDGPU.\n");
      return;
    }
    GenericPluginTy::init(NumDevices);
  }

  ~AMDGPUPluginTy() {}

  uint16_t getMagicElfBits() const override { return /* EM_AMDGPU */ 190; }

  AMDGPUDeviceTy &getDevice(int32_t DeviceId) override {
    assert(isValidDeviceId(DeviceId) && "Device Id is invalid");
    return Devices[DeviceId];
  }

private:
  SmallVector<AMDGPUDeviceTy, 8> Devices;
};

int32_t llvm::omp::plugin::GlobalHandlerTy::getGlobalMetadataFromDevice(
    GenericDeviceTy &Device, GlobalTy &DeviceGlobal) {
  AMDGPUDeviceTy &AMDGPUDevice = static_cast<AMDGPUDeviceTy &>(Device);
  SymbolInfoTableTy &SymbolInfoTable =
      *static_cast<GenericPluginAPIPayloadTy *>(Payload)->SymbolInfoTable;
  void *DevPtr;
  unsigned DevSize;
  const char *Name = DeviceGlobal.getName().c_str();
  hsa_status_t Err = interop_hsa_get_symbol_info(SymbolInfoTable, DeviceId,
                                                 Name, &DevPtr, &DevSize);

  if (Err != Status::SUCCESS) {
    // Inform the user what symbol prevented offloading
    DP("Loading global '%s' (Failed)\n", Name);
    return OFFLOAD_FAIL;
  }

  if (DevSize != DeviceGlobal.getSize()) {
    DP("Loading global '%s' - size mismatch (%u != %u)\n", Name, DevSize,
       DeviceGlobal.getSize());
    return OFFLOAD_FAIL;
  }

  DP("Entry point " DPxMOD " maps to global %s (" DPxMOD ")\n",
     DPxPTR(e - HostBegin), Name, DPxPTR(DevPtr));
  DeviceGlobal.setPtr(DevPtr);
  return OFFLOAD_SUCCESS;
}

StatusCode AMDGPUStreamTy::init(AMDGPUDeviceTy &AMDGPUDevice,
                                AMDGPUMemoryPoolTy &KernArgPool) {
  KernArgPoolPtr = &KernArgPool;
  return StatusCode(hsa_queue_create(
      AMDGPUDevice.getAgent(), AMDGPUDevice.getQueueSize(),
      HSA_QUEUE_TYPE_MULTI, nullptr, NULL, UINT32_MAX, UINT32_MAX, &Queue));
}

/// Expose the plugin to the generic part. Not ideal but not the worst.
namespace {
/// Wrapper around HSA used to ensure it is constructed before other types
/// and destructed after, which means said other types can use RAII for
/// cleanup without risking running outside of the lifetime of HSA
struct HSAWrapperTY {
  HSALifetime() : Err(hsa_init()) {
    if (Err != Status::SUCCESS)
      REPORT("Initializing HSA failed: %d", int32_t(Err));
  }

  ~HSALifetime() {
    if (Err != Status::SUCCESS)
      return;
    hsa_status_t Err = hsa_shut_down();
    if (Err != Status::SUCCESS)
      REPORT("Shutting down HSA failed: %d", int32_t(Err));
  }

  bool success() { return Err == Status::SUCCESS; }

  AMDGPUPluginTy Plugin;

private:
  hsa_status_t Err;
};

HSAWrapperTY HSAWrapper;

} // namespace

GenericPluginTy &llvm::omp::plugin::getPlugin() { return HSAWrapper.Plugin; }
AMDGPUPluginTy &getAMDGPUPlugin() {
  return static_cast<AMDGPUPluginTy &>(getPlugin());
}

namespace hsa {
template <typename C> hsa_status_t iterate_agents(C cb) {
  auto L = [](hsa_agent_t agent, void *data) -> hsa_status_t {
    C *unwrapped = static_cast<C *>(data);
    return (*unwrapped)(agent);
  };
  return hsa_iterate_agents(L, static_cast<void *>(&cb));
}

template <typename C>
hsa_status_t amd_agent_iterate_memory_pools(hsa_agent_t Agent, C cb) {
  auto L = [](hsa_amd_memory_pool_t MemoryPool, void *data) -> hsa_status_t {
    C *unwrapped = static_cast<C *>(data);
    return (*unwrapped)(MemoryPool);
  };

  return hsa_amd_agent_iterate_memory_pools(Agent, L, static_cast<void *>(&cb));
}

} // namespace hsa

template <typename Callback> static hsa_status_t FindAgents(Callback CB) {

  hsa_status_t Err =
      hsa::iterate_agents([&](hsa_agent_t agent) -> hsa_status_t {
        hsa_device_type_t device_type;
        // get_info fails iff HSA runtime not yet initialized
        hsa_status_t Err =
            hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);

        if (Err != Status::SUCCESS) {
          if (print_kernel_trace > 0)
            DP("rtl.cpp: Err %s\n", get_error_string(Err));

          return Err;
        }

        CB(device_type, agent);
        return Status::SUCCESS;
      });

  // iterate_agents fails iff HSA runtime not yet initialized
  if (print_kernel_trace > 0 && Err != Status::SUCCESS) {
    DP("rtl.cpp: Err %s\n", get_error_string(Err));
  }

  return Err;
}

namespace core {
namespace {
void packet_store_release(uint32_t *packet, uint16_t header, uint16_t rest) {
  __atomic_store_n(packet, header | (rest << 16), __ATOMIC_RELEASE);
}

uint16_t create_header() {
  uint16_t header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
  header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;
  return header;
}

hsa_status_t addMemoryPool(hsa_amd_memory_pool_t MemoryPool, void *Data) {
  std::vector<hsa_amd_memory_pool_t> *Result =
      static_cast<std::vector<hsa_amd_memory_pool_t> *>(Data);

  hsa_status_t Err;
  if ((Err = isValidMemoryPool(MemoryPool)) != Status::SUCCESS) {
    return Err;
  }

  Result->push_back(MemoryPool);
  return Status::SUCCESS;
}

} // namespace
} // namespace core

struct EnvironmentVariables {
  int NumTeams;
  int TeamLimit;
  int TeamThreadLimit;
  int MaxTeamsDefault;
};

template <uint32_t wavesize>
static constexpr const llvm::omp::GV &getGridValue() {
  return llvm::omp::getAMDGPUGridValues<wavesize>();
}

/// Class containing all the device information
class RTLDeviceInfoTy final : public GenericPluginTy {
  std::vector<std::list<KernelTy>> KernelLists;

  HSALifetime HSA; // First field => constructed first and destructed last

  struct QueueDeleter {
    void operator()(hsa_queue_t *Q) {
      if (Q) {
        hsa_status_t Err = hsa_queue_destroy(Q);
        if (Err != Status::SUCCESS) {
          DP("Error destroying hsa queue: %s\n", get_error_string(Err));
        }
      }
    }
  };

public:
  bool ConstructionSucceeded = false;

  // load binary populates symbol tables and mutates various global state
  // run uses those symbol tables
  std::shared_timed_mutex load_run_lock;

  int NumberOfDevices = 0;

  // GPU devices
  std::vector<hsa_agent_t> HSAAgents;
  std::vector<std::unique_ptr<hsa_queue_t, QueueDeleter>>
      HSAQueues; // one per gpu

  // CPUs
  std::vector<hsa_agent_t> CPUAgents;

  // Device properties
  std::vector<int> ComputeUnits;
  std::vector<int> GroupsPerDevice;
  std::vector<int> ThreadsPerGroup;
  std::vector<int> WarpSize;
  std::vector<std::string> GPUName;

  // OpenMP properties
  std::vector<int> NumTeams;
  std::vector<int> NumThreads;

  // OpenMP Environment properties
  EnvironmentVariables Env;

  // Resource pools
  SignalPoolT FreeSignalPool;

  bool hostcall_required = false;

  std::vector<hsa_executable_t> HSAExecutables;

  std::vector<KernelInfoTableTy> KernelInfoTables;
  std::vector<SymbolInfoTableTy> SymbolInfoTables;

  hsa_amd_memory_pool_t KernArgPool;

  // fine grained memory pool for host allocations
  hsa_amd_memory_pool_t HostFineGrainedMemoryPool;

  // fine and coarse-grained memory pools per offloading device
  std::vector<hsa_amd_memory_pool_t> DeviceFineGrainedMemoryPools;
  std::vector<hsa_amd_memory_pool_t> DeviceCoarseGrainedMemoryPools;

  struct implFreePtrDeletor {
    void operator()(void *p) {
      core::Runtime::Memfree(p); // ignore failure to free
    }
  };

  // device_State shared across loaded binaries, error if inconsistent size
  std::vector<std::pair<std::unique_ptr<void, implFreePtrDeletor>, uint64_t>>
      deviceStateStore;

  static const unsigned HardTeamLimit = 0;

  // These need to be per-device since different devices can have different
  // wave sizes, but are currently the same number for each so that refactor
  // can be postponed.
  static_assert(getGridValue<32>().GV_Max_Teams ==
                    getGridValue<64>().GV_Max_Teams,
                "");
  static const int Max_Teams = getGridValue<64>().GV_Max_Teams;

  static_assert(getGridValue<32>().GV_Max_WG_Size ==
                    getGridValue<64>().GV_Max_WG_Size,
                "");
  static const int Max_WG_Size = getGridValue<64>().GV_Max_WG_Size;

  static_assert(getGridValue<32>().GV_Default_WG_Size ==
                    getGridValue<64>().GV_Default_WG_Size,
                "");
  static const int Default_WG_Size = getGridValue<64>().GV_Default_WG_Size;

  using MemcpyFunc = hsa_status_t (*)(hsa_signal_t, void *, const void *,
                                      size_t size, hsa_agent_t,
                                      hsa_amd_memory_pool_t);
  hsa_status_t freesignalpool_memcpy(void *dest, const void *src, size_t size,
                                     MemcpyFunc Func, int32_t deviceId) {
    hsa_agent_t agent = HSAAgents[deviceId];
    hsa_signal_t s = FreeSignalPool.pop();
    if (s.handle == 0) {
      return HSA_STATUS_ERROR;
    }
    hsa_status_t r = Func(s, dest, src, size, agent, HostFineGrainedMemoryPool);
    FreeSignalPool.push(s);
    return r;
  }

  hsa_status_t freesignalpool_memcpy_d2h(void *dest, const void *src,
                                         size_t size, int32_t deviceId) {
    return freesignalpool_memcpy(dest, src, size, impl_memcpy_d2h, deviceId);
  }

  hsa_status_t freesignalpool_memcpy_h2d(void *dest, const void *src,
                                         size_t size, int32_t deviceId) {
    return freesignalpool_memcpy(dest, src, size, impl_memcpy_h2d, deviceId);
  }

  uint16_t getMagicElfBits() const override { return /* EM_AMDGPU */ 224; }

  bool isDataExchangable(int32_t SrcDeviceId,
                         int32_t DstDeviceId) const override {
    // TODO: Implement support!
    return false;
  }

  StatusCode queryNumDevices(int32_t &NumDevices) const override{
      // TODO;
  };

  hsa_status_t setupHostMemoryPools(std::vector<hsa_agent_t> &Agents) {
    std::vector<hsa_amd_memory_pool_t> HostPools;

    // collect all the "valid" pools for all the given agents.
    for (const auto &Agent : Agents) {
      hsa_status_t Err = hsa_amd_agent_iterate_memory_pools(
          Agent, core::addMemoryPool, static_cast<void *>(&HostPools));
      if (Err != Status::SUCCESS) {
        DP("addMemoryPool returned %s, continuing\n", get_error_string(Err));
      }
    }

    // We need two fine-grained pools.
    //  1. One with kernarg flag set for storing kernel arguments
    //  2. Second for host allocations
    bool FineGrainedMemoryPoolSet = false;
    bool KernArgPoolSet = false;
    for (const auto &MemoryPool : HostPools) {
      hsa_status_t Err = Status::SUCCESS;
      uint32_t GlobalFlags = 0;
      Err = hsa_amd_memory_pool_get_info(
          MemoryPool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &GlobalFlags);
      if (Err != Status::SUCCESS) {
        DP("Get memory pool info failed: %s\n", get_error_string(Err));
        return Err;
      }

      if (GlobalFlags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) {
        if (GlobalFlags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT) {
          KernArgPool = MemoryPool;
          KernArgPoolSet = true;
        }
        HostFineGrainedMemoryPool = MemoryPool;
        FineGrainedMemoryPoolSet = true;
      }
    }

    if (FineGrainedMemoryPoolSet && KernArgPoolSet)
      return Status::SUCCESS;

    return HSA_STATUS_ERROR;
  }

  hsa_amd_memory_pool_t getDeviceMemoryPool(int DeviceId) {
    assert(DeviceId >= 0 && DeviceId < DeviceCoarseGrainedMemoryPools.size() &&
           "Invalid device Id");
    return DeviceCoarseGrainedMemoryPools[DeviceId];
  }

  hsa_amd_memory_pool_t getHostMemoryPool() {
    return HostFineGrainedMemoryPool;
  }

  static int readEnvElseMinusOne(const char *Env) {
    const char *envStr = getenv(Env);
    int res = -1;
    if (envStr) {
      res = std::stoi(envStr);
      DP("Parsed %s=%d\n", Env, res);
    }
    return res;
  }

  RTLDeviceInfoTy() {
    DP("Start initializing " GETNAME(TARGET_NAME) "\n");

    // LIBOMPTARGET_KERNEL_TRACE provides a kernel launch trace to stderr
    // anytime. You do not need a debug library build.
    //  0 => no tracing
    //  1 => tracing dispatch only
    // >1 => verbosity increase

    if (!HSA.success()) {
      DP("Error when initializing HSA in " GETNAME(TARGET_NAME) "\n");
      return;
    }

    if (char *envStr = getenv("LIBOMPTARGET_KERNEL_TRACE"))
      print_kernel_trace = atoi(envStr);
    else
      print_kernel_trace = 0;

    hsa_status_t Err = core::atl_init_gpu_context();
    if (Err != Status::SUCCESS) {
      DP("Error when initializing " GETNAME(TARGET_NAME) "\n");
      return;
    }

    // Init hostcall soon after initializing hsa
    hostrpc_init();

    Err = FindAgents([&](hsa_device_type_t DeviceType, hsa_agent_t Agent) {
      if (DeviceType == HSA_DEVICE_TYPE_CPU) {
        CPUAgents.push_back(Agent);
      } else {
        HSAAgents.push_back(Agent);
      }
    });
    if (Err != Status::SUCCESS)
      return;

    NumberOfDevices = (int)HSAAgents.size();

    if (NumberOfDevices == 0) {
      DP("There are no devices supporting HSA.\n");
      return;
    } else {
      DP("There are %d devices supporting HSA.\n", NumberOfDevices);
    }

    initDeviceInterface(NumberOfDevices);

    // Init the device info
    HSAQueues.resize(NumberOfDevices);
    FuncGblEntries.resize(NumberOfDevices);
    ThreadsPerGroup.resize(NumberOfDevices);
    ComputeUnits.resize(NumberOfDevices);
    GPUName.resize(NumberOfDevices);
    GroupsPerDevice.resize(NumberOfDevices);
    WarpSize.resize(NumberOfDevices);
    NumTeams.resize(NumberOfDevices);
    NumThreads.resize(NumberOfDevices);
    deviceStateStore.resize(NumberOfDevices);
    KernelInfoTables.resize(NumberOfDevices);
    SymbolInfoTables.resize(NumberOfDevices);
    DeviceCoarseGrainedMemoryPools.resize(NumberOfDevices);
    DeviceFineGrainedMemoryPools.resize(NumberOfDevices);

    Err = setupDevicePools(HSAAgents);
    if (Err != Status::SUCCESS) {
      DP("Setup for Device Memory Pools failed\n");
      return;
    }

    Err = setupHostMemoryPools(CPUAgents);
    if (Err != Status::SUCCESS) {
      DP("Setup for Host Memory Pools failed\n");
      return;
    }

    for (int i = 0; i < NumberOfDevices; i++) {
      uint32_t queue_size = 0;
      {
        hsa_status_t Err = hsa_agent_get_info(
            HSAAgents[i], HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size);
        if (Err != Status::SUCCESS) {
          DP("HSA query QUEUE_MAX_SIZE failed for agent %d\n", i);
          return;
        }
        enum { MaxQueueSize = 4096 };
        if (queue_size > MaxQueueSize) {
          queue_size = MaxQueueSize;
        }
      }

      {
        hsa_queue_t *Q = nullptr;
        hsa_status_t rc =
            hsa_queue_create(HSAAgents[i], queue_size, HSA_QUEUE_TYPE_MULTI,
                             nullptr, NULL, UINT32_MAX, UINT32_MAX, &Q);
        if (rc != Status::SUCCESS) {
          DP("Failed to create HSA queue %d\n", i);
          return;
        }
        HSAQueues[i].reset(Q);
      }

      deviceStateStore[i] = {nullptr, 0};
    }

    for (int i = 0; i < NumberOfDevices; i++) {
      ThreadsPerGroup[i] = RTLDeviceInfoTy::Default_WG_Size;
      GroupsPerDevice[i] = RTLDeviceInfoTy::DefaultNumTeams;
      ComputeUnits[i] = 1;
      DP("Device %d: Initial groupsPerDevice %d & threadsPerGroup %d\n", i,
         GroupsPerDevice[i], ThreadsPerGroup[i]);
    }

    // Get environment variables regarding teams
    Env.TeamLimit = readEnvElseMinusOne("OMP_TEAM_LIMIT");
    Env.NumTeams = readEnvElseMinusOne("OMP_NUM_TEAMS");
    Env.MaxTeamsDefault = readEnvElseMinusOne("OMP_MAX_TEAMS_DEFAULT");
    Env.TeamThreadLimit = readEnvElseMinusOne("OMP_TEAMS_THREAD_LIMIT");

    ConstructionSucceeded = true;
  }

  ~RTLDeviceInfoTy() {
    DP("Finalizing the " GETNAME(TARGET_NAME) " Plugin.\n");
    if (!HSA.success()) {
      // Then none of these can have been set up and they can't be torn down
      return;
    }
    // Run destructors on types that use HSA before
    // impl_finalize removes access to it
    deviceStateStore.clear();
    KernelArgPoolMap.clear();
    // Terminate hostrpc before finalizing hsa
    hostrpc_terminate();

    hsa_status_t Err;
    for (uint32_t I = 0; I < HSAExecutables.size(); I++) {
      Err = hsa_executable_destroy(HSAExecutables[I]);
      if (Err != Status::SUCCESS) {
        DP("[%s:%d] %s failed: %s\n", __FILE__, __LINE__,
           "Destroying executable", get_error_string(Err));
      }
    }
  }
};

pthread_mutex_t SignalPoolT::mutex = PTHREAD_MUTEX_INITIALIZER;

static RTLDeviceInfoTy Plugin;

namespace {

int32_t dataRetrieve(int32_t DeviceId, void *HstPtr, void *TgtPtr, int64_t Size,
                     __tgt_async_info *AsyncInfo) {
  assert(AsyncInfo && "AsyncInfo is nullptr");
  assert(DeviceId < Plugin.NumberOfDevices && "Device ID too large");
  // Return success if we are not copying back to host from target.
  if (!HstPtr)
    return OFFLOAD_SUCCESS;
  hsa_status_t Err;
  DP("Retrieve data %ld bytes, (tgt:%016llx) -> (hst:%016llx).\n", Size,
     (long long unsigned)(Elf64_Addr)TgtPtr,
     (long long unsigned)(Elf64_Addr)HstPtr);

  Err =
      Plugin.freesignalpool_memcpy_d2h(HstPtr, TgtPtr, (size_t)Size, DeviceId);

  if (Err != Status::SUCCESS) {
    DP("Error when copying data from device to host. Pointers: "
       "host = 0x%016lx, device = 0x%016lx, size = %lld\n",
       (Elf64_Addr)HstPtr, (Elf64_Addr)TgtPtr, (unsigned long long)Size);
    return OFFLOAD_FAIL;
  }
  DP("DONE Retrieve data %ld bytes, (tgt:%016llx) -> (hst:%016llx).\n", Size,
     (long long unsigned)(Elf64_Addr)TgtPtr,
     (long long unsigned)(Elf64_Addr)HstPtr);
  return OFFLOAD_SUCCESS;
}

int32_t dataSubmit(int32_t DeviceId, void *TgtPtr, void *HstPtr, int64_t Size,
                   __tgt_async_info *AsyncInfo) {
  assert(AsyncInfo && "AsyncInfo is nullptr");
  hsa_status_t Err;
  assert(DeviceId < Plugin.NumberOfDevices && "Device ID too large");
  // Return success if we are not doing host to target.
  if (!HstPtr)
    return OFFLOAD_SUCCESS;

  DP("Submit data %ld bytes, (hst:%016llx) -> (tgt:%016llx).\n", Size,
     (long long unsigned)(Elf64_Addr)HstPtr,
     (long long unsigned)(Elf64_Addr)TgtPtr);
  Err =
      Plugin.freesignalpool_memcpy_h2d(TgtPtr, HstPtr, (size_t)Size, DeviceId);
  if (Err != Status::SUCCESS) {
    DP("Error when copying data from host to device. Pointers: "
       "host = 0x%016lx, device = 0x%016lx, size = %lld\n",
       (Elf64_Addr)HstPtr, (Elf64_Addr)TgtPtr, (unsigned long long)Size);
    return OFFLOAD_FAIL;
  }
  return OFFLOAD_SUCCESS;
}

// Async.
// The implementation was written with cuda streams in mind. The semantics of
// that are to execute kernels on a queue in order of insertion. A synchronise
// call then makes writes visible between host and device. This means a series
// of N data_submit_async calls are expected to execute serially. HSA offers
// various options to run the data copies concurrently. This may require changes
// to libomptarget.

// __tgt_async_info* contains a void * Queue. Queue = 0 is used to indicate that
// there are no outstanding kernels that need to be synchronized. Any async call
// may be passed a Queue==0, at which point the cuda implementation will set it
// to non-null (see getStream). The cuda streams are per-device. Upstream may
// change this interface to explicitly initialize the AsyncInfo_pointer, but
// until then hsa lazily initializes it as well.

void initAsyncInfo(__tgt_async_info *AsyncInfo) {
  // set non-null while using async calls, return to null to indicate completion
  assert(AsyncInfo);
  if (!AsyncInfo->Queue) {
    AsyncInfo->Queue = reinterpret_cast<void *>(UINT64_MAX);
  }
}
void finiAsyncInfo(__tgt_async_info *AsyncInfo) {
  assert(AsyncInfo);
  assert(AsyncInfo->Queue);
  AsyncInfo->Queue = 0;
}
} // namespace

namespace {
template <typename T> bool enforce_upper_bound(T *value, T upper) {
  bool changed = *value > upper;
  if (changed) {
    *value = upper;
  }
  return changed;
}
} // namespace

int32_t __tgt_rtl_init_device(int device_id) {
  hsa_status_t Err;

  // this is per device id init
  DP("Initialize the device id: %d\n", device_id);

  hsa_agent_t agent = Plugin.HSAAgents[device_id];

  // Get number of Compute Unit
  uint32_t compute_units = 0;
  Err = hsa_agent_get_info(
      agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT,
      &compute_units);
  if (Err != Status::SUCCESS) {
    Plugin.ComputeUnits[device_id] = 1;
    DP("Error getting compute units : settiing to 1\n");
  } else {
    Plugin.ComputeUnits[device_id] = compute_units;
    DP("Using %d compute unis per grid\n", Plugin.ComputeUnits[device_id]);
  }

  char GetInfoName[64]; // 64 max size returned by get info
  Err = hsa_agent_get_info(agent, (hsa_agent_info_t)HSA_AGENT_INFO_NAME,
                           (void *)GetInfoName);
  if (Err)
    Plugin.GPUName[device_id] = "--unknown gpu--";
  else {
    Plugin.GPUName[device_id] = GetInfoName;
  }

  if (print_kernel_trace & STARTUP_DETAILS)
    DP("Device#%-2d CU's: %2d %s\n", device_id, Plugin.ComputeUnits[device_id],
       Plugin.GPUName[device_id].c_str());

  // Query attributes to determine number of threads/block and blocks/grid.
  uint16_t workgroup_max_dim[3];
  Err = hsa_agent_get_info(agent, HSA_AGENT_INFO_WORKGROUP_MAX_DIM,
                           &workgroup_max_dim);
  if (Err != Status::SUCCESS) {
    Plugin.GroupsPerDevice[device_id] = RTLDeviceInfoTy::DefaultNumTeams;
    DP("Error getting grid dims: num groups : %d\n",
       RTLDeviceInfoTy::DefaultNumTeams);
  } else if (workgroup_max_dim[0] <= RTLDeviceInfoTy::HardTeamLimit) {
    Plugin.GroupsPerDevice[device_id] = workgroup_max_dim[0];
    DP("Using %d ROCm blocks per grid\n", Plugin.GroupsPerDevice[device_id]);
  } else {
    Plugin.GroupsPerDevice[device_id] = RTLDeviceInfoTy::HardTeamLimit;
    DP("Max ROCm blocks per grid %d exceeds the hard team limit %d, capping "
       "at the hard limit\n",
       workgroup_max_dim[0], RTLDeviceInfoTy::HardTeamLimit);
  }

  // Get thread limit
  hsa_dim3_t grid_max_dim;
  Err = hsa_agent_get_info(agent, HSA_AGENT_INFO_GRID_MAX_DIM, &grid_max_dim);
  if (Err == Status::SUCCESS) {
    Plugin.ThreadsPerGroup[device_id] =
        reinterpret_cast<uint32_t *>(&grid_max_dim)[0] /
        Plugin.GroupsPerDevice[device_id];

    if (Plugin.ThreadsPerGroup[device_id] == 0) {
      Plugin.ThreadsPerGroup[device_id] = RTLDeviceInfoTy::Max_WG_Size;
      DP("Default thread limit: %d\n", RTLDeviceInfoTy::Max_WG_Size);
    } else if (enforce_upper_bound(&Plugin.ThreadsPerGroup[device_id],
                                   RTLDeviceInfoTy::Max_WG_Size)) {
      DP("Capped thread limit: %d\n", RTLDeviceInfoTy::Max_WG_Size);
    } else {
      DP("Using ROCm Queried thread limit: %d\n",
         Plugin.ThreadsPerGroup[device_id]);
    }
  } else {
    Plugin.ThreadsPerGroup[device_id] = RTLDeviceInfoTy::Max_WG_Size;
    DP("Error getting max block dimension, use default:%d \n",
       RTLDeviceInfoTy::Max_WG_Size);
  }

  // Get wavefront size
  uint32_t wavefront_size = 0;
  Err =
      hsa_agent_get_info(agent, HSA_AGENT_INFO_WAVEFRONT_SIZE, &wavefront_size);
  if (Err == Status::SUCCESS) {
    DP("Queried wavefront size: %d\n", wavefront_size);
    Plugin.WarpSize[device_id] = wavefront_size;
  } else {
    // TODO: Burn the wavefront size into the code object
    DP("Warning: Unknown wavefront size, assuming 64\n");
    Plugin.WarpSize[device_id] = 64;
  }

  // Adjust teams to the env variables

  if (Plugin.Env.TeamLimit > 0 &&
      (enforce_upper_bound(&Plugin.GroupsPerDevice[device_id],
                           Plugin.Env.TeamLimit))) {
    DP("Capping max groups per device to OMP_TEAM_LIMIT=%d\n",
       Plugin.Env.TeamLimit);
  }

  // Set default number of teams
  if (Plugin.Env.NumTeams > 0) {
    Plugin.NumTeams[device_id] = Plugin.Env.NumTeams;
    DP("Default number of teams set according to environment %d\n",
       Plugin.Env.NumTeams);
  } else {
    char *TeamsPerCUEnvStr = getenv("OMP_TARGET_TEAMS_PER_PROC");
    int TeamsPerCU = DefaultTeamsPerCU;
    if (TeamsPerCUEnvStr) {
      TeamsPerCU = std::stoi(TeamsPerCUEnvStr);
    }

    Plugin.NumTeams[device_id] = TeamsPerCU * Plugin.ComputeUnits[device_id];
    DP("Default number of teams = %d * number of compute units %d\n",
       TeamsPerCU, Plugin.ComputeUnits[device_id]);
  }

  if (enforce_upper_bound(&Plugin.NumTeams[device_id],
                          Plugin.GroupsPerDevice[device_id])) {
    DP("Default number of teams exceeds device limit, capping at %d\n",
       Plugin.GroupsPerDevice[device_id]);
  }

  // Adjust threads to the env variables
  if (Plugin.Env.TeamThreadLimit > 0 &&
      (enforce_upper_bound(&Plugin.NumThreads[device_id],
                           Plugin.Env.TeamThreadLimit))) {
    DP("Capping max number of threads to OMP_TEAMS_THREAD_LIMIT=%d\n",
       Plugin.Env.TeamThreadLimit);
  }

  // Set default number of threads
  Plugin.NumThreads[device_id] = RTLDeviceInfoTy::Default_WG_Size;
  DP("Default number of threads set according to library's default %d\n",
     RTLDeviceInfoTy::Default_WG_Size);
  if (enforce_upper_bound(&Plugin.NumThreads[device_id],
                          Plugin.ThreadsPerGroup[device_id])) {
    DP("Default number of threads exceeds device limit, capping at %d\n",
       Plugin.ThreadsPerGroup[device_id]);
  }

  DP("Device %d: default limit for groupsPerDevice %d & threadsPerGroup %d\n",
     device_id, Plugin.GroupsPerDevice[device_id],
     Plugin.ThreadsPerGroup[device_id]);

  DP("Device %d: wavefront size %d, total threads %d x %d = %d\n", device_id,
     Plugin.WarpSize[device_id], Plugin.ThreadsPerGroup[device_id],
     Plugin.GroupsPerDevice[device_id],
     Plugin.GroupsPerDevice[device_id] * Plugin.ThreadsPerGroup[device_id]);

  return OFFLOAD_SUCCESS;
}

namespace {
template <typename C>
hsa_status_t module_register_from_memory_to_place(
    KernelInfoTableTy &KernelInfoTable, SymbolInfoTableTy &SymbolInfoTable,
    void *module_bytes, size_t module_size, int DeviceId, C cb,
    std::vector<hsa_executable_t> &HSAExecutables) {
  auto L = [](void *data, size_t size, void *cb_state) -> hsa_status_t {
    C *unwrapped = static_cast<C *>(cb_state);
    return (*unwrapped)(data, size);
  };
  return core::RegisterModuleFromMemory(
      KernelInfoTable, SymbolInfoTable, module_bytes, module_size,
      Plugin.HSAAgents[DeviceId], L, static_cast<void *>(&cb), HSAExecutables);
}
} // namespace

static hsa_status_t impl_calloc(void **ret_ptr, size_t size, int DeviceId) {
  uint64_t rounded = 4 * ((size + 3) / 4);
  void *ptr;
  hsa_amd_memory_pool_t MemoryPool = Plugin.getDeviceMemoryPool(DeviceId);
  hsa_status_t Err = hsa_amd_memory_pool_allocate(MemoryPool, rounded, 0, &ptr);
  if (Err != Status::SUCCESS) {
    return Err;
  }

  hsa_status_t rc = hsa_amd_memory_fill(ptr, 0, rounded / 4);
  if (rc != Status::SUCCESS) {
    DP("zero fill device_state failed with %u\n", rc);
    core::Runtime::Memfree(ptr);
    return HSA_STATUS_ERROR;
  }

  *ret_ptr = ptr;
  return Status::SUCCESS;
}

// Determine launch values for kernel.
struct launchVals {
  int WorkgroupSize;
  int GridSize;
};
launchVals getLaunchVals(int WarpSize, EnvironmentVariables Env,
                         int ConstWGSize,
                         llvm::omp::OMPTgtExecModeFlags ExecutionMode,
                         int num_teams, int thread_limit,
                         uint64_t loop_tripcount, int DeviceNumTeams) {

  int threadsPerGroup = RTLDeviceInfoTy::Default_WG_Size;
  int num_groups = 0;

  int Max_Teams =
      Env.MaxTeamsDefault > 0 ? Env.MaxTeamsDefault : DeviceNumTeams;
  if (Max_Teams > RTLDeviceInfoTy::HardTeamLimit)
    Max_Teams = RTLDeviceInfoTy::HardTeamLimit;

  if (print_kernel_trace & STARTUP_DETAILS) {
    DP("RTLDeviceInfoTy::Max_Teams: %d\n", RTLDeviceInfoTy::Max_Teams);
    DP("Max_Teams: %d\n", Max_Teams);
    DP("RTLDeviceInfoTy::Warp_Size: %d\n", WarpSize);
    DP("RTLDeviceInfoTy::Max_WG_Size: %d\n", RTLDeviceInfoTy::Max_WG_Size);
    DP("RTLDeviceInfoTy::Default_WG_Size: %d\n",
       RTLDeviceInfoTy::Default_WG_Size);
    DP("thread_limit: %d\n", thread_limit);
    DP("threadsPerGroup: %d\n", threadsPerGroup);
    DP("ConstWGSize: %d\n", ConstWGSize);
  }
  // check for thread_limit() clause
  if (thread_limit > 0) {
    threadsPerGroup = thread_limit;
    DP("Setting threads per block to requested %d\n", thread_limit);
    // Add master warp for GENERIC
    if (ExecutionMode ==
        llvm::omp::OMPTgtExecModeFlags::OMP_TGT_EXEC_MODE_GENERIC) {
      threadsPerGroup += WarpSize;
      DP("Adding master wavefront: +%d threads\n", WarpSize);
    }
    if (threadsPerGroup > RTLDeviceInfoTy::Max_WG_Size) { // limit to max
      threadsPerGroup = RTLDeviceInfoTy::Max_WG_Size;
      DP("Setting threads per block to maximum %d\n", threadsPerGroup);
    }
  }
  // check flat_max_work_group_size attr here
  if (threadsPerGroup > ConstWGSize) {
    threadsPerGroup = ConstWGSize;
    DP("Reduced threadsPerGroup to flat-attr-group-size limit %d\n",
       threadsPerGroup);
  }
  if (print_kernel_trace & STARTUP_DETAILS)
    DP("threadsPerGroup: %d\n", threadsPerGroup);
  DP("Preparing %d threads\n", threadsPerGroup);

  // Set default num_groups (teams)
  if (Env.TeamLimit > 0)
    num_groups = (Max_Teams < Env.TeamLimit) ? Max_Teams : Env.TeamLimit;
  else
    num_groups = Max_Teams;
  DP("Set default num of groups %d\n", num_groups);

  if (print_kernel_trace & STARTUP_DETAILS) {
    DP("num_groups: %d\n", num_groups);
    DP("num_teams: %d\n", num_teams);
  }

  // Reduce num_groups if threadsPerGroup exceeds RTLDeviceInfoTy::Max_WG_Size
  // This reduction is typical for default case (no thread_limit clause).
  // or when user goes crazy with num_teams clause.
  // FIXME: We cant distinguish between a constant or variable thread limit.
  // So we only handle constant thread_limits.
  if (threadsPerGroup >
      RTLDeviceInfoTy::Default_WG_Size) //  256 < threadsPerGroup <= 1024
    // Should we round threadsPerGroup up to nearest WarpSize
    // here?
    num_groups = (Max_Teams * RTLDeviceInfoTy::Max_WG_Size) / threadsPerGroup;

  // check for num_teams() clause
  if (num_teams > 0) {
    num_groups = (num_teams < num_groups) ? num_teams : num_groups;
  }
  if (print_kernel_trace & STARTUP_DETAILS) {
    DP("num_groups: %d\n", num_groups);
    DP("Env.NumTeams %d\n", Env.NumTeams);
    DP("Env.TeamLimit %d\n", Env.TeamLimit);
  }

  if (Env.NumTeams > 0) {
    num_groups = (Env.NumTeams < num_groups) ? Env.NumTeams : num_groups;
    DP("Modifying teams based on Env.NumTeams %d\n", Env.NumTeams);
  } else if (Env.TeamLimit > 0) {
    num_groups = (Env.TeamLimit < num_groups) ? Env.TeamLimit : num_groups;
    DP("Modifying teams based on Env.TeamLimit%d\n", Env.TeamLimit);
  } else {
    if (num_teams <= 0) {
      if (loop_tripcount > 0) {
        if (ExecutionMode ==
            llvm::omp::OMPTgtExecModeFlags::OMP_TGT_EXEC_MODE_SPMD) {
          // round up to the nearest integer
          num_groups = ((loop_tripcount - 1) / threadsPerGroup) + 1;
        } else if (ExecutionMode ==
                   llvm::omp::OMPTgtExecModeFlags::OMP_TGT_EXEC_MODE_GENERIC) {
          num_groups = loop_tripcount;
        } else /* OMP_TGT_EXEC_MODE_GENERIC_SPMD */ {
          // This is a generic kernel that was transformed to use SPMD-mode
          // execution but uses Generic-mode semantics for scheduling.
          num_groups = loop_tripcount;
        }
        DP("Using %d teams due to loop trip count %" PRIu64 " and number of "
           "threads per block %d\n",
           num_groups, loop_tripcount, threadsPerGroup);
      }
    } else {
      num_groups = num_teams;
    }
    if (num_groups > Max_Teams) {
      num_groups = Max_Teams;
      if (print_kernel_trace & STARTUP_DETAILS)
        DP("Limiting num_groups %d to Max_Teams %d \n", num_groups, Max_Teams);
    }
    if (num_groups > num_teams && num_teams > 0) {
      num_groups = num_teams;
      if (print_kernel_trace & STARTUP_DETAILS)
        DP("Limiting num_groups %d to clause num_teams %d \n", num_groups,
           num_teams);
    }
  }

  // num_teams clause always honored, no matter what, unless DEFAULT is active.
  if (num_teams > 0) {
    num_groups = num_teams;
    // Cap num_groups to EnvMaxTeamsDefault if set.
    if (Env.MaxTeamsDefault > 0 && num_groups > Env.MaxTeamsDefault)
      num_groups = Env.MaxTeamsDefault;
  }
  if (print_kernel_trace & STARTUP_DETAILS) {
    DP("threadsPerGroup: %d\n", threadsPerGroup);
    DP("num_groups: %d\n", num_groups);
    DP("loop_tripcount: %ld\n", loop_tripcount);
  }
  DP("Final %d num_groups and %d threadsPerGroup\n", num_groups,
     threadsPerGroup);

  launchVals res;
  res.WorkgroupSize = threadsPerGroup;
  res.GridSize = threadsPerGroup * num_groups;
  return res;
}

namespace core {
hsa_status_t allow_access_to_all_gpu_agents(void *ptr) {
  return hsa_amd_agents_allow_access(Plugin.HSAAgents.size(),
                                     &Plugin.HSAAgents[0], NULL, ptr);
}

} // namespace core
