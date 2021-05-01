//===------ State.cpp - OpenMP State & ICV interface ------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "State.h"
#include "Configuration.h"
#include "Debug.h"
#include "Interface.h"
#include "Mapping.h"
#include "Synchronization.h"
#include "Types.h"
#include "Utils.h"
#include <stdio.h>

using namespace _OMP;

/// Memory implementation
///
///{

namespace {

#pragma omp declare target

/// Fallback implementations are missing to trigger a link time error.
/// Implementations for new devices, including the host, should go into a
/// dedicated begin/end declare variant.
///
///{

extern "C" {
void *malloc(uint64_t Size);
void free(void *Ptr);
}

///}

/// AMDGCN implementations of the shuffle sync idiom.
///
///{
#pragma omp begin declare variant match(device = {arch(amdgcn)})

extern "C" {
void *malloc(uint64_t Size) {
  // TODO: Use some preallocated space for dynamic malloc.
  return 0;
}

void free(void *Ptr) {}
}

#pragma omp end declare variant
///}

/// Add worst-case padding so that future allocations are properly aligned.
constexpr const uint32_t Alignment = 8;

/// We do expose a malloc/free interface but instead should take the number of
/// bytes also in the pop call. Until clang is updated we hide the allocation
/// size "under" the allocated pointer.
constexpr const uint32_t StorageTrackingBytes = 8;

static_assert((StorageTrackingBytes % Alignment) == 0,
              "Storage tracker should preserve alignment.");

/// A "smart" stack in shared memory.
///
/// The stack exposes a malloc/free interface but works like a stack internally.
/// In fact, it is a separate stack *per warp*. That means, each warp must push
/// and pop symmetrically or this breaks, badly. The implementation will (aim
/// to) detect non-lock-step warps and fallback to malloc/free. The same will
/// happen if a warp runs out of memory. The master warp in generic memory is
/// special and is given more memory than the rest.
///
struct SharedMemorySmartStackTy {
  /// Initialize the stack. Must be called by all threads.
  void init(bool IsSPMD);

  /// Allocate \p Bytes on the stack for the encountering thread. Each thread
  /// can call this function.
  void *push(uint64_t Bytes);

  /// Deallocate the last allocation made by the encountering thread and pointed
  /// to by \p Ptr from the stack. Each thread can call this function.
  void pop(void *Ptr);

private:
  /// Compute the size of the storage space reserved for the warp of the
  /// encountering thread.
  uint32_t computeWarpStorageTotal(bool IsSPMD);

  /// Return the bottom address of the warp data stack, that is the first
  /// address this warp allocated memory at, or will.
  char *getWarpDataBottom(bool IsSPMD);

  /// Same as \p getWarpDataBottom() but it can access other warp's bottom. No
  /// lock is used so it's user's responsibility to make sure the correctness.
  char *getWarpDataBottom(bool IsSPMD, uint32_t WarpId);

  /// Return the top address of the warp data stack, that is the first address
  /// this warp will allocate memory at next.
  char *getWarpDataTop(bool IsSPMD);

  /// Return the location of the usage tracker which keeps track of the amount
  /// of memory used by this warp.
  /// TODO: We could use the next warp bottom to avoid this tracker completely.
  uint32_t *getWarpStorageTracker(bool IsSPMD);

  /// Same as \p getWarpStorageTracker() but it can access other warp's bottom.
  /// No lock is used so it's user's responsibility to make sure the
  /// correctness.
  uint32_t *getWarpStorageTracker(bool IsSPMD, uint32_t WarpId);

  /// The actual storage, shared among all warps.
  char Data[state::SharedScratchpadSize] __attribute__((aligned(Alignment)));
};

/// The allocation of a single shared memory scratchpad.
static SharedMemorySmartStackTy SHARED(SharedMemorySmartStack);

void SharedMemorySmartStackTy::init(bool IsSPMD) {
  static_assert(
      StorageTrackingBytes >= (sizeof(*getWarpStorageTracker(0))),
      "Storage tracker bytes should cover the size of the storage tracker.");

  // Initialize the tracker for all warps if in non-spmd mode because this
  // function will be only called by master thread.
  if (mapping::isLeaderInWarp()) {
    uint32_t *WarpStorageTracker = getWarpStorageTracker(IsSPMD);
    *WarpStorageTracker = StorageTrackingBytes;
  }
}

void *SharedMemorySmartStackTy::push(uint64_t BytesPerLane) {
  // First align the number of requested bytes.
  BytesPerLane = (BytesPerLane + (Alignment - 1)) / Alignment * Alignment;

  // The pointer we eventually return.
  void *Ptr = nullptr;

  LaneMaskTy Active = mapping::activemask();
  uint32_t NumActive = utils::popc(Active);

  // Only the leader allocates, the rest, if any, waits for the result at the
  // shfl_sync below.
  //printf("push %lu : %i\n", BytesPerLane, (mapping::isLeaderInWarp()));
  if (mapping::isLeaderInWarp()) {
    uint32_t BytesTotal = BytesPerLane * NumActive;

    // Ensure the warp is complete or it is the sole main thread in generic
    // mode. In any other situation we might not be able to preserve balanced
    // push/pop accesses even if the code looked that way because the hardware
    // could independently schedule the warp parts. If an incomplete warp
    // arrives here we fallback to the slow path, namely malloc.
    if (NumActive < mapping::getWarpSize() &&
        !mapping::isMainThreadInGenericMode()) {
      Ptr = memory::allocGlobal(
          BytesTotal, "Slow path shared memory allocation, incomplete warp!");
      //printf("alloc global1 %p\n", Ptr);
    } else {
      // If warp is "complete" determine if we have sufficient space.
      bool IsSPMD = mapping::isSPMDMode();
      uint32_t *WarpStorageTracker = getWarpStorageTracker(IsSPMD);
      uint32_t BytesTotalAdjusted = BytesTotal + StorageTrackingBytes;
      uint32_t BytesTotalAdjustedAligned =
          (BytesTotalAdjusted + (Alignment - 1)) / Alignment * Alignment;

      uint32_t BytesInUse = *WarpStorageTracker;
      if (BytesInUse + BytesTotalAdjustedAligned > computeWarpStorageTotal(IsSPMD)) {
        Ptr = memory::allocGlobal(
            BytesTotal,
            "Slow path shared memory allocation, insufficient memory!");
        //printf("alloc global2 %p\n", Ptr);
      } else {
        // We have enough memory, put the new allocation on the top of the stack
        // preceded by the size of the allocation.
        Ptr = getWarpDataTop(IsSPMD);
        *WarpStorageTracker += BytesTotalAdjustedAligned;
        *((uint64_t *)Ptr) = BytesTotalAdjustedAligned;
        Ptr = ((char *)Ptr) + StorageTrackingBytes;
        //printf("alloc shared %p\n", Ptr);
      }
    }
  }

  //printf("allocated %p %i\n", Ptr, omp_get_num_threads());
  // Skip the shfl_sync if the thread is alone.
  if (omp_get_num_threads() == 1)
    return Ptr;

  // Get the address of the allocation from the leader.
  uint32_t Leader = utils::ffs(Active) - 1;
  int *FP = reinterpret_cast<int *>(&Ptr);
  FP[0] = utils::shuffle(Active, FP[0], Leader);
  if (sizeof(Ptr) == 8)
    FP[1] = utils::shuffle(Active, FP[1], Leader);

  // Compute the thread position into the allocation, which we did for the
  // entire warp.
  LaneMaskTy LaneMaskLT = mapping::lanemaskLT();
  uint32_t WarpPosition = utils::popc(Active & LaneMaskLT);
  return reinterpret_cast<char *>(Ptr) + (BytesPerLane * WarpPosition);
}

void SharedMemorySmartStackTy::pop(void *Ptr) {
  // Only the leader deallocates, the rest, if any, waits at the synchwarp
  // below.
  if (mapping::isLeaderInWarp()) {
    // memory::freeGlobal(Ptr, "Slow path shared memory deallocation");
    // // Check if the pointer is from a malloc or from within the stack.
    if (Ptr < &Data[0] || Ptr >= &Data[state::SharedScratchpadSize]) {
      //printf("Free global %p\n", Ptr);
      memory::freeGlobal(Ptr, "Slow path shared memory deallocation");
    } else {
      // Lookup the allocation size "below" the allocation (=Ptr).
      Ptr = reinterpret_cast<char *>(Ptr) - StorageTrackingBytes;
      //printf("Free shared %p\n", Ptr);
      uint64_t BytesTotalAdjustedAligned = *reinterpret_cast<uint64_t *>(Ptr);

      // Free the memory by adjusting the storage tracker accordingly.
      bool IsSPMD = mapping::isSPMDMode();
      uint32_t *WarpStorageTracker = getWarpStorageTracker(IsSPMD);
      *WarpStorageTracker -= BytesTotalAdjustedAligned;
    }
  }
  // Ensure the entire warp waits until the pop is done.
  //synchronize::warp(mapping::activemask());
}

uint32_t SharedMemorySmartStackTy::computeWarpStorageTotal(bool IsSPMD) {
  if (!IsSPMD && mapping::getThreadIdInBlock() == 0)
    return config::getGenericModeMainThreadSharedMemoryStorage();

  // In generic mode we reserve parts of the storage for the main thread.
  uint32_t StorageTotal = state::SharedScratchpadSize;
  if (!IsSPMD)
    StorageTotal -= config::getGenericModeMainThreadSharedMemoryStorage();

  uint32_t NumWarps = mapping::getNumberOfWarpsInBlock();
  uint32_t WarpStorageTotal = StorageTotal / NumWarps;

  // Align the size
  WarpStorageTotal = WarpStorageTotal / Alignment * Alignment;

  return WarpStorageTotal;
}

char *SharedMemorySmartStackTy::getWarpDataBottom(bool IsSPMD, uint32_t WarpId) {
  if (!IsSPMD && mapping::getThreadIdInBlock() == 0)
    return &Data[0];

  uint32_t PriorWarpStorageTotal = 0;
  if (!IsSPMD)
    PriorWarpStorageTotal +=  config::getGenericModeMainThreadSharedMemoryStorage();

  PriorWarpStorageTotal += computeWarpStorageTotal(IsSPMD) * WarpId;

  return &Data[PriorWarpStorageTotal];
}

char *SharedMemorySmartStackTy::getWarpDataBottom(bool IsSPMD) {
  return getWarpDataBottom(IsSPMD, mapping::getWarpId());
}

char *SharedMemorySmartStackTy::getWarpDataTop(bool IsSPMD) {
  uint32_t *WarpStorageTracker = getWarpStorageTracker(IsSPMD);
  return getWarpDataBottom(IsSPMD) + (*WarpStorageTracker);
}

uint32_t *SharedMemorySmartStackTy::getWarpStorageTracker(bool IsSPMD, uint32_t WarpId) {
  return ((uint32_t *)getWarpDataBottom(IsSPMD, WarpId));
}

uint32_t *SharedMemorySmartStackTy::getWarpStorageTracker(bool IsSPMD) {
  return getWarpStorageTracker(IsSPMD, mapping::getWarpId());
}

#pragma omp end declare target
} // namespace

// TODO: Clang should accept namespaces inside the declare target range.
#pragma omp declare target

void *memory::allocShared(uint64_t Bytes, const char *Reason) {
  return SharedMemorySmartStack.push(Bytes);
}

void memory::freeShared(void *Ptr, const char *Reason) {
  SharedMemorySmartStack.pop(Ptr);
}

void *memory::allocGlobal(uint64_t Bytes, const char *Reason) {
  return malloc(Bytes);
}

void memory::freeGlobal(void *Ptr, const char *Reason) { free(Ptr); }

#pragma omp end declare target

///}

namespace {

#pragma omp declare target

struct ICVStateTy {
  uint32_t NThreadsVar;
  uint32_t LevelVar;
  uint32_t ActiveLevelVar;
  uint32_t MaxActiveLevelsVar;
  uint32_t RunSchedVar;
  uint32_t RunSchedChunkVar;

  bool operator==(const ICVStateTy &Other) const;

  void assertEqual(const ICVStateTy &Other) const;
};

bool ICVStateTy::operator==(const ICVStateTy &Other) const {
  return (NThreadsVar == Other.NThreadsVar) & (LevelVar == Other.LevelVar) &
         (ActiveLevelVar == Other.ActiveLevelVar) &
         (MaxActiveLevelsVar == Other.MaxActiveLevelsVar) &
         (RunSchedVar == Other.RunSchedVar) &
         (RunSchedChunkVar == Other.RunSchedChunkVar);
}

void ICVStateTy::assertEqual(const ICVStateTy &Other) const {
  ASSERT(NThreadsVar == Other.NThreadsVar);
  ASSERT(LevelVar == Other.LevelVar);
  ASSERT(ActiveLevelVar == Other.ActiveLevelVar);
  ASSERT(MaxActiveLevelsVar == Other.MaxActiveLevelsVar);
  ASSERT(RunSchedVar == Other.RunSchedVar);
  ASSERT(RunSchedChunkVar == Other.RunSchedChunkVar);
}

struct TeamStateTy {
  /// TODO: provide a proper init function.
  void init(bool IsSPMD);

  bool operator==(const TeamStateTy &) const;

  void assertEqual(TeamStateTy &Other) const;

  /// ICVs
  ///
  /// Preallocated storage for ICV values that are used if the threads have not
  /// set a custom default. The latter is supported but unlikely and slow(er).
  ///
  ///{
  ICVStateTy ICVState;
  ///}

  uint32_t ParallelTeamSize;
  ParallelRegionFnTy ParallelRegionFnVar;
};

TeamStateTy SHARED(TeamState);

void TeamStateTy::init(bool IsSPMD) {
  if (IsSPMD) {
    ICVState.NThreadsVar = 1;
    ICVState.LevelVar = 1;
    ICVState.ActiveLevelVar = 1;
    ICVState.MaxActiveLevelsVar = 1;
    ICVState.RunSchedVar = omp_sched_static;
    ICVState.RunSchedChunkVar = 1;
    ParallelTeamSize = mapping::getBlockSize();
  } else {
    ICVState.NThreadsVar = mapping::getBlockSize();
    ICVState.LevelVar = 0;
    ICVState.ActiveLevelVar = 0;
    ICVState.MaxActiveLevelsVar = 1;
    ICVState.RunSchedVar = omp_sched_static;
    ICVState.RunSchedChunkVar = 1;
    ParallelTeamSize = 1;
  }
}

bool TeamStateTy::operator==(const TeamStateTy &Other) const {
  return (ICVState == Other.ICVState) &
         (ParallelTeamSize == Other.ParallelTeamSize);
}

void TeamStateTy::assertEqual(TeamStateTy &Other) const {
  ICVState.assertEqual(Other.ICVState);
  ASSERT(ParallelTeamSize == Other.ParallelTeamSize);
}

struct ThreadStateTy {

  /// ICVs have preallocated storage in the TeamStateTy which is used if a
  /// thread has not set a custom value. The latter is supported but unlikely.
  /// When it happens we will allocate dynamic memory to hold the values of all
  /// ICVs. Thus, the first time an ICV is set by a thread we will allocate an
  /// ICV struct to hold them all. This is slower than alternatives but allows
  /// users to pay only for what they use.
  ///
  ICVStateTy ICVState;

  ThreadStateTy *PreviousThreadState;

  void init() {
    ICVState = TeamState.ICVState;
    PreviousThreadState = nullptr;
  }

  void init(ThreadStateTy &PreviousTS) {
    ICVState = PreviousTS.ICVState;
    PreviousThreadState = &PreviousTS;
  }
};

__attribute__((loader_uninitialized))
ThreadStateTy *ThreadStates[mapping::MaxThreadsPerTeam];
#pragma omp allocate(ThreadStates) allocator(omp_pteam_mem_alloc)

uint32_t &lookupForModify32Impl(uint32_t ICVStateTy::*Var) {
  if (TeamState.ICVState.LevelVar == 0)
    return TeamState.ICVState.*Var;
  uint32_t TId = mapping::getThreadIdInBlock();
  //printf("ThreadState %i\n", TId);
  //__builtin_trap();
  if (!ThreadStates[TId]) {
    ThreadStates[TId] = reinterpret_cast<ThreadStateTy *>(memory::allocGlobal(
        sizeof(ThreadStateTy), "ICV modification outside data environment"));
    ThreadStates[TId]->init();
  }
  return ThreadStates[TId]->ICVState.*Var;
}

uint32_t &lookup32Impl(uint32_t ICVStateTy::*Var) {
  uint32_t TId = mapping::getThreadIdInBlock();
  if (ThreadStates[TId])
    return ThreadStates[TId]->ICVState.*Var;
  return TeamState.ICVState.*Var;
}
uint64_t &lookup64Impl(uint64_t ICVStateTy::*Var) {
  uint64_t TId = mapping::getThreadIdInBlock();
  if (ThreadStates[TId])
    return ThreadStates[TId]->ICVState.*Var;
  return TeamState.ICVState.*Var;
}

int returnValIfLevelIsActive(int Level, int Val, int DefaultVal,
                             int OutOfBoundsVal = -1) {
  if (Level == 0)
    return DefaultVal;
  int LevelVar = omp_get_level();
  if (Level < 0 || Level > LevelVar)
    return OutOfBoundsVal;
  int ActiveLevel = icv::ActiveLevel;
  if (Level != ActiveLevel)
    return DefaultVal;
  return Val;
}

#pragma omp end declare target

} // namespace

#pragma omp declare target

uint32_t &state::lookup32(ValueKind Kind, bool IsReadonly) {
  //printf("lookup32 %i %i\n", Kind, IsReadonly);
  switch (Kind) {
  case state::VK_NThreads:
    if (IsReadonly)
      return lookup32Impl(&ICVStateTy::NThreadsVar);
    return lookupForModify32Impl(&ICVStateTy::NThreadsVar);
  case state::VK_Level:
    if (IsReadonly)
      return lookup32Impl(&ICVStateTy::LevelVar);
    return lookupForModify32Impl(&ICVStateTy::LevelVar);
  case state::VK_ActiveLevel:
    if (IsReadonly)
      return lookup32Impl(&ICVStateTy::ActiveLevelVar);
    return lookupForModify32Impl(&ICVStateTy::ActiveLevelVar);
  case state::VK_MaxActiveLevels:
    if (IsReadonly)
      return lookup32Impl(&ICVStateTy::MaxActiveLevelsVar);
    return lookupForModify32Impl(&ICVStateTy::MaxActiveLevelsVar);
  case state::VK_RunSched:
    if (IsReadonly)
      return lookup32Impl(&ICVStateTy::RunSchedVar);
    return lookupForModify32Impl(&ICVStateTy::RunSchedVar);
  case state::VK_RunSchedChunk:
    if (IsReadonly)
      return lookup32Impl(&ICVStateTy::RunSchedChunkVar);
    return lookupForModify32Impl(&ICVStateTy::RunSchedChunkVar);
  case state::VK_ParallelTeamSize:
    return TeamState.ParallelTeamSize;
  default:
    break;
  }
  __builtin_unreachable();
}

void *&state::lookupPtr(ValueKind Kind, bool IsReadonly) {
  switch (Kind) {
  case state::VK_ParallelRegionFn:
    return TeamState.ParallelRegionFnVar;
  default:
    break;
  }
  __builtin_unreachable();
}

void state::init(bool IsSPMD) {
  SharedMemorySmartStack.init(IsSPMD);
  if (!mapping::getThreadIdInBlock())
    TeamState.init(IsSPMD);

  ThreadStates[mapping::getThreadIdInBlock()] = nullptr;
}

void state::enterDataEnvironment() {
  unsigned TId = mapping::getThreadIdInBlock();
  //printf("enterDataEnv %i\n", TId);
  //__builtin_trap();

  ThreadStateTy *NewThreadState =
      static_cast<ThreadStateTy *>(__kmpc_alloc_shared(sizeof(ThreadStateTy)));
  NewThreadState->init(*ThreadStates[TId]);
  ThreadStates[TId] = NewThreadState;
}

void state::exitDataEnvironment() {
  // assert(ThreadStates[TId] && "exptected thread state");
  unsigned TId = mapping::getThreadIdInBlock();
  resetStateForThread(TId);
}

void state::resetStateForThread(uint32_t TId) {
  if (!ThreadStates[TId])
    return;

  ThreadStateTy *PreviousThreadState = ThreadStates[TId]->PreviousThreadState;
  __kmpc_free_shared(ThreadStates[TId]);
  ThreadStates[TId] = PreviousThreadState;
}

void state::runAndCheckState(void(Func(void))) {
  TeamStateTy OldTeamState = TeamState;
  OldTeamState.assertEqual(TeamState);

  Func();

  OldTeamState.assertEqual(TeamState);
}

void state::assumeInitialState(bool IsSPMD) {
  TeamStateTy InitialTeamState;
  InitialTeamState.init(IsSPMD);
  InitialTeamState.assertEqual(TeamState);
  ASSERT(!ThreadStates[mapping::getThreadIdInBlock()]);
  ASSERT(mapping::isSPMDMode() == IsSPMD);
}

extern "C" {
void omp_set_dynamic(int V) {}

int omp_get_dynamic(void) { return 0; }

void omp_set_num_threads(int V) { icv::NThreads = V; }

int omp_get_max_threads(void) { return icv::NThreads; }

int omp_get_level(void) {
  int LevelVar = icv::Level;
  __builtin_assume(LevelVar >= 0);
  return LevelVar;
}

int omp_get_active_level(void) { return !!icv::ActiveLevel; }

int omp_in_parallel(void) { return !!icv::ActiveLevel; }

void omp_get_schedule(omp_sched_t *ScheduleKind, int *ChunkSize) {
  *ScheduleKind = static_cast<omp_sched_t>((int)icv::RunSched);
  *ChunkSize = state::RunSchedChunk;
}

void omp_set_schedule(omp_sched_t ScheduleKind, int ChunkSize) {
  icv::RunSched = (int)ScheduleKind;
  state::RunSchedChunk = ChunkSize;
}

int omp_get_ancestor_thread_num(int Level) {
  return returnValIfLevelIsActive(Level, mapping::getThreadIdInBlock(), 0);
}

__attribute__((flatten, always_inline))
int omp_get_thread_num(void) {
  return omp_get_ancestor_thread_num(omp_get_level());
}

__attribute__((flatten, always_inline))
int omp_get_team_size(int Level) {
  return returnValIfLevelIsActive(Level, state::ParallelTeamSize, 1);
}

int omp_get_num_threads(void) { return state::ParallelTeamSize; }

int omp_get_thread_limit(void) { return mapping::getKernelSize(); }

int omp_get_num_procs(void) { return mapping::getNumberOfProcessorElements(); }

void omp_set_nested(int) {}

int omp_get_nested(void) { return false; }

void omp_set_max_active_levels(int Levels) {
  icv::MaxActiveLevels = Levels > 0 ? 1 : 0;
}

int omp_get_max_active_levels(void) { return icv::MaxActiveLevels; }

omp_proc_bind_t omp_get_proc_bind(void) { return omp_proc_bind_false; }

int omp_get_num_places(void) { return 0; }

int omp_get_place_num_procs(int) { return omp_get_num_procs(); }

void omp_get_place_proc_ids(int, int *) {
  // TODO
}

int omp_get_place_num(void) { return 0; }

int omp_get_partition_num_places(void) { return 0; }

void omp_get_partition_place_nums(int *) {
  // TODO
}

int omp_get_cancellation(void) { return 0; }

void omp_set_default_device(int) {}

int omp_get_default_device(void) { return -1; }

int omp_get_num_devices(void) { return config::getNumDevices(); }

int omp_get_num_teams(void) { return mapping::getNumberOfBlocks(); }

int omp_get_team_num() { return mapping::getBlockId(); }

int omp_get_initial_device(void) { return -1; }
}

extern "C" {
// TODO: The noinline is a workaround until we run OpenMP opt before the
// inliner.
__attribute__((flatten, noinline))
void *__kmpc_alloc_shared(uint64_t Bytes) {
  return memory::allocShared(Bytes, "Frontend alloc shared");
}

// TODO: The noinline is a workaround until we run OpenMP opt before the
// inliner.
__attribute__((flatten, noinline))
void __kmpc_free_shared(void *Ptr) {
  memory::freeShared(Ptr, "Frontend free shared");
}

/// The shared variable used by the main thread to communicate with the workers.
/// It will contain the location of the memory allocated for the actually shared
/// values.
///
/// Workaround until the interface is changed.
static void **SHARED(GlobalArgsPtr);

__attribute__((flatten, always_inline))
void __kmpc_begin_sharing_variables(void ***GlobalArgs, uint64_t NumArgs) {
  // TODO: To mimic the old behavior we allocate in `sizeof(void*)` chunks. We
  //       should pass the required bytes instead.
  *GlobalArgs = GlobalArgsPtr = static_cast<decltype(GlobalArgsPtr)>(
    __kmpc_alloc_shared(NumArgs * sizeof(GlobalArgsPtr[0])));
}

__attribute__((flatten, always_inline))
void __kmpc_end_sharing_variables() {
  __kmpc_free_shared(GlobalArgsPtr);
}

void __kmpc_get_shared_variables(void ***GlobalArgs) {
  *GlobalArgs = GlobalArgsPtr;
}
}
#pragma omp end declare target
