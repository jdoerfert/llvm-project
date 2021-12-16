//===- DeviceInterface.h - Target independent plugin device interface -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OPENMP_LIBOMPTARGET_PLUGINS_COMMON_DEVICEINTERFACE_DEVICEINTERFACE_H
#define LLVM_OPENMP_LIBOMPTARGET_PLUGINS_COMMON_DEVICEINTERFACE_DEVICEINTERFACE_H

#include <cstddef>
#include <cstdint>
#include <list>
#include <map>
#include <vector>

#include "Debug.h"
#include "DeviceEnvironment.h"
#include "GlobalHandler.h"
#include "MemoryManager.h"
#include "omptarget.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/Frontend/OpenMP/OMPGridValues.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBufferRef.h"

namespace llvm {
namespace omp {
namespace plugin {

struct GenericPluginTy;
struct GenericDeviceTy;

enum Status {
  SUCCESS = 0,
};
template <typename Ty>
Ty getErrorString(Ty ErrorCode, const char **ErrorString);

struct StatusCode {
  enum Tag { OK_TAG };
  StatusCode(Tag) : V(0), Checked(true) {}
  explicit StatusCode(int32_t V) : V(V), Checked(false) {}
  ~StatusCode() { assert(Checked && "Unchecked status code\n"); }

  static StatusCode OK;
  static StatusCode FAIL;

  StatusCode &operator=(const StatusCode &OtherSC) {
    if (this != &OtherSC) {
      assert(checked() && "Overwriting unchecked status code!");
      *this = OtherSC;
    }
    return *this;
  }

  bool failedAndChecked() const { return V && Checked; }
  bool failedAndUnchecked() const { return V && !Checked; }
  bool checked() const { return Checked; }
  bool isOK() const { return V == 0; }

  int32_t get() {
    Checked = true;
    return V;
  }
  operator bool() {
    Checked = true;
    return V;
  }

private:
  int32_t V;
  bool Checked;
};

template <typename Ty> struct EnvironmentVariable {
  EnvironmentVariable(const char *Name, int32_t DeviceId,
                      uint32_t Default = -1) {
    if (const char *EnvStr = getenv(Name)) {
      Data = std::stoi(EnvStr);
      INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId, "Parsed %s=%d\n", Name,
           (int)Data);
    } else {
      Data = Default;
      INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId, "Default %s=%d\n", Name,
           (int)Data);
    }
  }
  EnvironmentVariable(const char *Name, int32_t DeviceId,
                      function_ref<int32_t(Ty &)> Getter,
                      function_ref<int32_t(Ty)> Setter) {
    if (const char *EnvStr = getenv(Name)) {
      Data = std::stoi(EnvStr);
      INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId, "Parsed %s=%d\n", Name,
           (int)Data);
      if (Setter(Data)) {
        Getter(Data);
        INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
             "Setter failed, reset %s=%d\n", Name, (int)Data);
      }
    } else {
      Getter(Data);
      INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId, "Default %s=%d\n", Name,
           (int)Data);
    }
  }

  Ty get() const { return Data; }
  operator Ty() const { return get(); }

private:
  Ty Data;
};
using Int32EnvironmentVariable = EnvironmentVariable<uint32_t>;
using Int64EnvironmentVariable = EnvironmentVariable<uint64_t>;

struct AsyncInfoWrapperTy {
  AsyncInfoWrapperTy(StatusCode &SC, GenericDeviceTy &Device,
                     __tgt_async_info *AsyncInfoPtr)
      : SC(SC), Device(Device),
        AsyncInfoPtr(AsyncInfoPtr ? AsyncInfoPtr : &LocalAsyncInfo) {}
  ~AsyncInfoWrapperTy();

  operator __tgt_async_info *() const { return AsyncInfoPtr; }

  template <typename Ty> Ty &getQueueAs() const {
    static_assert(sizeof(Ty) == sizeof(AsyncInfoPtr->Queue),
                  "Queue is not of the same size as target type!");
    return reinterpret_cast<Ty &>(AsyncInfoPtr->Queue);
  }

private:
  StatusCode &SC;
  GenericDeviceTy &Device;
  __tgt_async_info LocalAsyncInfo;
  __tgt_async_info *const AsyncInfoPtr;
};

struct GenericKernelTy {
  GenericKernelTy(const char *Name, OMPTgtExecModeFlags ExecutionMode)
      : Name(Name), ExecutionMode(ExecutionMode) {}
  virtual ~GenericKernelTy() {}

  void init(GenericDeviceTy &GenericDevice);
  virtual void initImpl(GenericDeviceTy &GenericDevice) = 0;

  StatusCode launch(GenericDeviceTy &GenericDevice, void **ArgPtrs,
                    ptrdiff_t *ArgOffsets, int32_t NumArgs,
                    int32_t NumTeamsClause, int32_t ThreadLimitClause,
                    int32_t LoopTripCount,
                    AsyncInfoWrapperTy &AsyncInfoWrapper);

  virtual void *argumentPrepareImpl(GenericDeviceTy &GenericDevice,
                                    void **ArgPtrs, ptrdiff_t *ArgOffsets,
                                    int32_t NumArgs,
                                    AsyncInfoWrapperTy &AsyncInfoWrapper) = 0;
  virtual StatusCode launchImpl(GenericDeviceTy &GenericDevice,
                                int32_t NumThreads, int32_t NumBlocks,
                                int32_t DynamicMemorySize, void *KernelArgs,
                                AsyncInfoWrapperTy &AsyncInfoWrapper) = 0;

  virtual int32_t
  getDefaultThreadCount(GenericDeviceTy &GenericDevice) const = 0;
  int32_t getNumThreads(GenericDeviceTy &GenericDevice,
                        int32_t ThreadLimitClause) const;

  virtual int32_t
  getDefaultBlockCount(GenericDeviceTy &GenericDevice) const = 0;
  int32_t getNumBlocks(GenericDeviceTy &GenericDevice, int32_t BlockLimitClause,
                       int32_t LoopTripCount, int32_t NumThreads) const;

  const char *getName() const { return Name; }

  bool isGenericSPMDMode() const {
    return ExecutionMode == OMP_TGT_EXEC_MODE_GENERIC_SPMD;
  }
  bool isGenericMode() const {
    return ExecutionMode == OMP_TGT_EXEC_MODE_GENERIC;
  }
  bool isSPMDMode() const { return ExecutionMode == OMP_TGT_EXEC_MODE_SPMD; }

  const char *getExecutionModeName() const {
    switch (ExecutionMode) {
    case OMP_TGT_EXEC_MODE_SPMD:
      return "SPMD";
    case OMP_TGT_EXEC_MODE_GENERIC:
      return "Generic";
    case OMP_TGT_EXEC_MODE_GENERIC_SPMD:
      return "Generic-SPMD";
    }
    llvm_unreachable("Unknown execution mode!");
  }

private:
  const char *Name;
  bool Initialized = false;
  OMPTgtExecModeFlags ExecutionMode = OMP_TGT_EXEC_MODE_SPMD;

protected:
  int32_t DynamicMemorySize = -1;
  int32_t PreferredThreadCount = -1;
  int32_t MaxThreadCount = -1;
};

/// Stream pool manager.
template <typename StreamTy> struct StreamManagerTy {

  StreamManagerTy(int32_t DeviceId, int32_t DefaultStreamPoolSize = 32)
      : InitialStreamPoolSize("LIBOMPTARGET_NUM_INITIAL_STREAMS", DeviceId,
                              DefaultStreamPoolSize) {}
  virtual ~StreamManagerTy() {}

  /// Get a stream from pool. Per-device next stream id always points to the
  /// next available stream. That means, streams [0, id-1] have been
  /// assigned, and [id,] are still available. If there is no stream left, we
  /// will ask more streams from the RT. Each time a stream is assigned,
  /// the id will increase one.
  /// xxxxxs+++++++++
  ///      ^
  ///      id
  /// After assignment, the pool becomes the following and s is assigned.
  /// xxxxxs+++++++++
  ///       ^
  ///       id
  StreamTy getStream() {
    const std::lock_guard<std::mutex> Lock(Mtx);
    if (NextStreamId == StreamPool.size()) {
      // By default we double the stream pool every time.
      if (resizeStreamPool(NextStreamId * 2))
        return StreamTy();
    }
    return StreamPool[NextStreamId++];
  }

  /// Return a stream back to pool. As mentioned above, per-device next
  /// stream is always points to the next available stream, so when we return
  /// a stream, we need to first decrease the id, and then copy the stream
  /// back.
  /// It is worth noting that, the order of streams return might be different
  /// from that they're assigned, that saying, at some point, there might be
  /// two identical streams.
  /// xxax+a+++++
  ///     ^
  ///     id
  /// However, it doesn't matter, because they're always on the two sides of
  /// id. The left one will in the end be overwritten by another stream.
  /// Therefore, after several execution, the order of pool might be different
  /// from its initial state.
  void returnStream(StreamTy Stream) {
    const std::lock_guard<std::mutex> Lock(Mtx);
    StreamPool[--NextStreamId] = Stream;
  }

  /// Initialize the stream pool.
  StatusCode init() {
    assert(StreamPool.empty() && "stream pool has been initialized");
    return resizeStreamPool(std::max(uint32_t(1), InitialStreamPoolSize.get()));
  }

private:
  /// The subclass needs to provide this method in which the streams between
  /// \p OldSize and \p NewSize need to be initialized. The mutex is locked
  /// when called.
  virtual StatusCode resizeStreamPoolImpl(int32_t OldSize, int32_t NewSize) = 0;

  /// If there is no stream left in the pool, we will resize the pool to
  /// allocate more stream. This function should be called with the mutex
  /// locked and only to grow the pool.
  StatusCode resizeStreamPool(int32_t NewSize) {
    int32_t OldSize = StreamPool.size();
    assert(NewSize > OldSize && "new size is not larger than current size");

    StreamPool.resize(NewSize);

    return resizeStreamPoolImpl(OldSize, NewSize);
  }

  /// Mutex for the stream pool.
  std::mutex Mtx;

  /// The next available stream in the pool.
  uint32_t NextStreamId = 0;

  /// The initial stream pool size, potentially defined by an environment
  /// variable.
  Int32EnvironmentVariable InitialStreamPoolSize;

protected:
  /// The actual stream pool.
  std::deque<StreamTy> StreamPool;
};

struct GenericDeviceTy : public DeviceAllocatorTy {
  const int32_t DeviceId;
  GenericDeviceTy(int32_t DeviceId, const llvm::omp::GV &GridValues);

  struct OffloadEntryTableTy {

    void clear() {
      Entries.clear();
      TTTablePtr.EntriesBegin = TTTablePtr.EntriesEnd = nullptr;
    }

    void addEntry(const __tgt_offload_entry &Entry) {
      Entries.push_back(Entry);
      TTTablePtr.EntriesBegin = &Entries[0];
      TTTablePtr.EntriesEnd = TTTablePtr.EntriesBegin + Entries.size();
    }

    __tgt_offload_entry *getEntry(void *Addr) {
      auto It = EntryMap.find(Addr);
      if (It == EntryMap.end())
        return nullptr;
      return &Entries[It->second];
    }

    operator __tgt_target_table *() {
      if (Entries.empty())
        return nullptr;
      return &TTTablePtr;
    }

  private:
    __tgt_target_table TTTablePtr;
    std::map<void *, unsigned> EntryMap;
    std::vector<__tgt_offload_entry> Entries;
  };

  StatusCode init(GenericPluginTy &Plugin);
  virtual StatusCode initImpl(GenericPluginTy &Plugin) = 0;

  __tgt_target_table *loadBinary(const __tgt_device_image *Image);
  virtual StatusCode loadBinaryImpl() = 0;

  StatusCode setupDeviceEnvironment(GenericPluginTy &Plugin);

  StatusCode registerOffloadEntries();

  StatusCode synchronize(__tgt_async_info *AsyncInfo);
  virtual StatusCode synchronizeImpl(__tgt_async_info &AsyncInfo) = 0;

  void *dataAlloc(int64_t Size, void *HostPtr, TargetAllocTy Kind);
  StatusCode dataDelete(void *TgtPtr);

  StatusCode dataSubmit(void *TgtPtr, const void *HstPtr, int64_t Size,
                        __tgt_async_info *AsyncInfo);
  virtual StatusCode dataSubmitImpl(void *TgtPtr, const void *HstPtr,
                                    int64_t Size,
                                    AsyncInfoWrapperTy &AsyncInfoWrapper) = 0;

  StatusCode dataRetrieve(void *HstPtr, const void *TgtPtr, int64_t Size,
                          __tgt_async_info *AsyncInfo);
  virtual StatusCode dataRetrieveImpl(void *HstPtr, const void *TgtPtr,
                                      int64_t Size,
                                      AsyncInfoWrapperTy &AsyncInfoWrapper) = 0;

  StatusCode dataExchange(const void *SrcPtr, int32_t DstDevId, void *DstPtr,
                          int64_t Size, __tgt_async_info *AsyncInfo);
  virtual StatusCode dataExchangeImpl(const void *SrcPtr, int32_t DstDevId,
                                      void *DstPtr, int64_t Size,
                                      AsyncInfoWrapperTy &AsyncInfoWrapper) = 0;

  StatusCode runTargetTeamRegion(void *EntryPtr, void **ArgPtrs,
                                 ptrdiff_t *ArgOffsets, int32_t NumArgs,
                                 int32_t NumTeamsClause,
                                 int32_t ThreadLimitClause,
                                 uint64_t LoopTripCount,
                                 __tgt_async_info *AsyncInfo);

  /// Event API
  /// {
  StatusCode createEvent(void **EventPtrStorage);
  virtual StatusCode createEventImpl(void **EventPtrStorage) = 0;
  StatusCode destroyEvent(void *EventPtr);
  virtual StatusCode destroyEventImpl(void *EventPtr) = 0;
  StatusCode recordEvent(void *EventPtr, __tgt_async_info *AsyncInfo);
  virtual StatusCode recordEventImpl(void *EventPtr,
                                     AsyncInfoWrapperTy &AsyncInfoWrapper) = 0;
  StatusCode waitEvent(void *EventPtr, __tgt_async_info *AsyncInfo);
  virtual StatusCode waitEventImpl(void *EventPtr,
                                   AsyncInfoWrapperTy &AsyncInfoWrapper) = 0;
  StatusCode syncEvent(void *EventPtr);
  virtual StatusCode syncEventImpl(void *EventPtr) = 0;
  ///}

  const char *getErrorStr(StatusCode &SC);

  void printInfo();
  virtual void printInfoImpl() = 0;

  int32_t getWarpSize() const { return GridValues.GV_Warp_Size; }
  int32_t getBlockLimit() const { return GridValues.GV_Max_Teams; }
  int32_t getThreadLimit() const { return GridValues.GV_Max_WG_Size; }
  int32_t getDefaultThreadCount() const {
    return GridValues.GV_Default_WG_Size;
  }
  int32_t getDefaultBlockCount() const {
    // TODO: Introduce a default num blocks value.
    return GridValues.GV_Default_WG_Size;
  }
  int32_t getDynamicMemorySize() const {
    return DeviceEnvironment.DynamicMemSize;
  }

  const __tgt_device_image *getImage() const {
    assert(Image && "Device image was not loaded!");
    return Image;
  }
  size_t getImageSize() const {
    assert(Image && "Device image was not loaded!");
    return ((char *)Image->ImageEnd) - ((char *)Image->ImageStart);
  }

  MemoryBufferRef getImageBuffer() const {
    assert(Image && "Device image was not loaded!");
    return MemoryBufferRef(
        StringRef((const char *)Image->ImageStart, getImageSize()), "Image");
  }

private:
  StatusCode registerGlobalOffloadEntry(const __tgt_offload_entry *GlobalEntry);
  StatusCode registerKernelOffloadEntry(const __tgt_offload_entry *KernelEntry);
  virtual GenericKernelTy *
  constructKernelEntry(const __tgt_offload_entry *KernelEntry) = 0;

  StatusCode getDeviceStackSize(uint64_t &V);
  StatusCode setDeviceStackSize(uint64_t V);

  StatusCode getDeviceHeapSize(uint64_t &V);
  StatusCode setDeviceHeapSize(uint64_t V);

  Int32EnvironmentVariable OMP_TeamLimit;
  Int32EnvironmentVariable OMP_NumTeams;
  Int32EnvironmentVariable OMPX_DebugKind;
  Int32EnvironmentVariable OMPX_SharedMemorySize;
  Int32EnvironmentVariable OMP_TeamsThreadLimit;
  Int64EnvironmentVariable OMPX_TargetStackSize;
  Int64EnvironmentVariable OMPX_TargetHeapSize;

  OffloadEntryTableTy OffloadEntryTable;

  DeviceEnvironmentTy DeviceEnvironment;

  std::unique_ptr<MemoryManagerTy> MemoryManager;

  const __tgt_device_image *Image = nullptr;

protected:
  llvm::omp::GV GridValues;
};

struct GenericPluginTy {
  virtual ~GenericPluginTy() {}

  virtual GenericDeviceTy &getDevice(int32_t DeviceId) = 0;

  int32_t getNumDevices() const { return NumDevices; }

  virtual uint16_t getMagicElfBits() const = 0;

  template <typename Ty> Ty *allocate() {
    return reinterpret_cast<Ty *>(Allocator.Allocate(sizeof(Ty), alignof(Ty)));
  }

  GlobalHandlerTy &getGlobalHandler() { return GlobalHandler; }

  int64_t getRequiresFlags() { return RequiresFlags; }

  void setRequiresFlag(int64_t Flags) { RequiresFlags = Flags; }

  virtual bool isDataExchangable(int32_t SrcDeviceId, int32_t DstDeviceId) {
    return isValidDeviceId(SrcDeviceId) && isValidDeviceId(DstDeviceId);
  }

protected:
  void init(int32_t NumDevices) { this->NumDevices = NumDevices; }

  virtual void setStackAndHeapSize();
  virtual StatusCode queryNumDevices(int32_t &NumDevices) const;

  bool isValidDeviceId(int32_t DeviceId) const {
    return DeviceId >= 0 && DeviceId < getNumDevices();
  }

private:
  int32_t NumDevices;

  /// OpenMP requires flags
  int64_t RequiresFlags = OMP_REQ_UNDEFINED;

  GlobalHandlerTy GlobalHandler;

  BumpPtrAllocator Allocator;
};

extern GenericPluginTy &getPlugin();

} // namespace plugin
} // namespace omp
} // namespace llvm

#endif // LLVM_OPENMP_LIBOMPTARGET_PLUGINS_COMMON_DEVICEINTERFACE_DEVICEINTERFACE_H
