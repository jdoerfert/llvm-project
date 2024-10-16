//===------ Sanitizer.cpp - Track allocation for sanitizer checks ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "DeviceTypes.h"
#include "DeviceUtils.h"
#include "Mapping.h"
#include "Shared/Environment.h"
#include "Synchronization.h"

using namespace ompx;

#define _INLINE [[gnu::always_inline]]
#define _NOINLINE [[gnu::noinline]]
#define _FLATTEN [[gnu::flatten]]
#define _KEEP [[gnu::used, gnu::retain]]
#define _SAN_ATTRS [[clang::disable_sanitizer_instrumentation]]
#define _SAN_ENTRY_ATTRS _FLATTEN _INLINE _SAN_ATTRS _KEEP

#pragma omp begin declare target device_type(nohost)

enum {
  GlobalAS = 1,
  SharedAS = 3,
  AllocaAS = 5,
};

template <uint32_t AS> using ASPtrTy = char [[clang::address_space(AS)]] *;
using GlobalPtrTy = char [[clang::address_space(GlobalAS)]] *;
using SharedPtrTy = char [[clang::address_space(SharedAS)]] *;
using AllocaPtrTy = char [[clang::address_space(AllocaAS)]] *;

[[gnu::visibility(
    "protected")]] _KEEP SanitizerEnvironmentTy *__sanitizer_environment_ptr;
[[gnu::visibility("protected")]] _KEEP int64_t
    [[clang::address_space(SharedAS)]] *__offload_san_ambiguous_calls_info_ptr =
        nullptr;
struct __attribute__((packed)) PtrInfoTy {
  char *Base;
  uint64_t Size;
};
struct __attribute__((packed)) PtrASInfoTy {
  PtrInfoTy PI;
  uint32_t AS;
};
struct __attribute__((packed)) GlobalInfoTy {
  char *Base;
  uint64_t LocationId;
};

static constexpr uint32_t __san_num_ptr_infos = 1 << 16;
static PtrInfoTy __san_ptr_infos[__san_num_ptr_infos];
static uint32_t __san_ptr_info_cnt = 0;

static constexpr uint32_t __san_num_global_infos = 128;
static GlobalInfoTy __san_global_infos[__san_num_global_infos];
static uint32_t __san_global_info_cnt = 0;

namespace {

int64_t getAmbiguouesCallId() {
  auto GTId = ompx_global_thread_id();
  return __offload_san_ambiguous_calls_info_ptr[GTId];
}

/// Helper to lock the sanitizer environment. While we never unlock it, this
/// allows us to have a no-op "side effect" in the spin-wait function below.
_SAN_ATTRS bool
getSanitizerEnvironmentLock(SanitizerEnvironmentTy &SE,
                            SanitizerEnvironmentTy::ErrorCodeTy ErrorCode) {
  return atomic::cas(SE.getErrorCodeLocation(), SanitizerEnvironmentTy::NONE,
                     ErrorCode, atomic::OrderingTy::seq_cst,
                     atomic::OrderingTy::seq_cst);
}

/// The spin-wait function should not be inlined, it's a catch all to give one
/// thread time to setup the sanitizer environment.
_SAN_ATTRS void spinWait(SanitizerEnvironmentTy &SE) {
  while (!atomic::load(&SE.IsInitialized, atomic::OrderingTy::aquire))
    ;
  //  __builtin_trap();
}

struct FakePtrTy;
_SAN_ATTRS void setAccessInfo(SanitizerEnvironmentTy &SE, const FakePtrTy &FP,
                              uint32_t AccessSize);

_SAN_ATTRS void setLocation(SanitizerEnvironmentTy &SE, uint64_t PC,
                            uint64_t LocationId) {
  for (int I = 0; I < 3; ++I) {
    SE.ThreadId[I] = mapping::getThreadIdInBlock(I);
    SE.BlockId[I] = mapping::getBlockIdInKernel(I);
  }
  SE.PC = PC;
  SE.LocationId = LocationId;
  SE.CallId = getAmbiguouesCallId();

  // This is the last step to initialize the sanitizer environment, time to
  // trap via the spinWait. Flush the memory writes and signal for the end.
  fence::system(atomic::OrderingTy::release);
  atomic::store(&SE.IsInitialized, 1, atomic::OrderingTy::release);
}

_SAN_ATTRS _FLATTEN _NOINLINE void
raiseExecutionError(SanitizerEnvironmentTy::ErrorCodeTy ErrorCode, uint64_t PC,
                    uint64_t LocationId, const FakePtrTy &FP,
                    uint32_t AccessSize = 0) {
  SanitizerEnvironmentTy &SE = *__sanitizer_environment_ptr;
  bool HasLock = getSanitizerEnvironmentLock(SE, ErrorCode);

  // If no thread of this warp has the lock, end execution gracefully.
  // bool AnyThreadHasLock = utils::ballotSync(lanes::All, HasLock);
  //  if (!AnyThreadHasLock)
  //    utils::terminateWarp();

  // One thread will set the location information and signal that the rest of
  // the wapr that the actual trap can be executed now.
  if (HasLock) {
    setAccessInfo(SE, FP, AccessSize);
    setLocation(SE, PC, LocationId);
  }

  synchronize::warp(lanes::All);

  // This is not the first thread that encountered the trap, to avoid a race
  // on the sanitizer environment, this thread is simply going to spin-wait.
  // The trap above will end the program for all threads.
  spinWait(SE);
}

struct FakePtrTy {

  FakePtrUnionEncodingTy U;

  static_assert(sizeof(U) == sizeof(void *), "Encoding is too large!");

  _SAN_ATTRS
  FakePtrTy() { U.VPtr = nullptr; }

  _SAN_ATTRS
  FakePtrTy(void *FakePtr) { U.VPtr = FakePtr; }

  _SAN_ATTRS
  FakePtrTy(void *FakePtr, int32_t AS, bool Checked, uint64_t PC,
            uint64_t LocationId)
      : FakePtrTy(FakePtr) {
    if (Checked)
      return;
    bool BadAS = U.Enc32.RealAS != AS;
    bool BadMagic =
        (AS == GlobalAS ? U.Enc64.Magic : U.Enc32.Magic) != FAKE_PTR_MAGIC;
    if (BadMagic | BadAS)
      raiseExecutionError(BadMagic ? SanitizerEnvironmentTy::BAD_PTR
                                   : SanitizerEnvironmentTy::AS_MISMATCH,
                          uint64_t(PC), LocationId, *this);
  }

  _SAN_ATTRS
  uint64_t getMaxSize() { return (1UL << 40) - 1; }

  template <uint32_t AS>
  _SAN_ATTRS static FakePtrTy create(uint64_t PC, uint64_t LocationId,
                                     ASPtrTy<AS> Ptr, uint64_t Size) {
    FakePtrTy FP;

    FP.U.Enc32.RealAS = AS;
    FP.U.Enc32.RealPtr = uint64_t(Ptr);
    FP.U.Enc32.Size = Size;
    FP.U.Enc32.Magic = FAKE_PTR_MAGIC;

    return FP;
  }

  template <>
  _SAN_ATTRS FakePtrTy create<GlobalAS>(uint64_t PC, uint64_t LocationId,
                                        GlobalPtrTy Ptr, uint64_t Size) {
    FakePtrTy FP;
    if (Size > FP.getMaxSize())
      raiseExecutionError(SanitizerEnvironmentTy::ALLOCATION_TOO_LARGE, PC,
                          LocationId, FakePtrTy((void *)(uint64_t)Size));

    uint32_t SlotId = atomic::inc(&__san_ptr_info_cnt, __san_num_ptr_infos,
                                  atomic::acq_rel, atomic::device);
    __san_ptr_infos[SlotId].Base = decltype(PtrInfoTy::Base)(Ptr);
    __san_ptr_infos[SlotId].Size = Size;
    FP.U.Enc64.RealAS = GlobalAS;
    FP.U.Enc64.SlotId = SlotId;
    FP.U.Enc64.Magic = FAKE_PTR_MAGIC;

    return FP;
  }

  template <uint32_t AS>
  _SAN_ATTRS static FakePtrTy registerGlobal(uint64_t PC, uint64_t LocationId,
                                             ASPtrTy<AS> Ptr, uint64_t Size) {

    uint32_t GlobalSlotId = __san_global_info_cnt++;
    if (GlobalSlotId < __san_num_global_infos) {
      __san_global_infos[GlobalSlotId].Base = decltype(GlobalInfoTy::Base)(Ptr);
      __san_global_infos[GlobalSlotId].LocationId = LocationId;
    }

    uint32_t PtrSlotId = __san_ptr_info_cnt++;
    __san_ptr_infos[PtrSlotId].Base = decltype(PtrInfoTy::Base)(Ptr);
    __san_ptr_infos[PtrSlotId].Size = Size;

    FakePtrTy FP;
    FP.U.Enc64.RealAS = AS;
    FP.U.Enc64.SlotId = PtrSlotId;
    FP.U.Enc64.Magic = FAKE_PTR_MAGIC;
    return FP;
  }

  template <>
  _SAN_ATTRS FakePtrTy registerGlobal<SharedAS>(uint64_t PC,
                                                uint64_t LocationId,
                                                ASPtrTy<SharedAS> Ptr,
                                                uint64_t Size) {
    uint32_t GlobalSlotId = __san_global_info_cnt++;
    if (GlobalSlotId < __san_num_global_infos) {
      __san_global_infos[GlobalSlotId].Base =
          decltype(GlobalInfoTy::Base)(uint64_t(Ptr));
      __san_global_infos[GlobalSlotId].LocationId = LocationId;
    }

    return create<SharedAS>(PC, LocationId, Ptr, Size);
  }

  _SAN_ATTRS static void registerHost(void *Ptr, uint64_t Size,
                                      uint32_t SlotId) {
    __san_ptr_infos[SlotId].Base = decltype(PtrInfoTy::Base)(Ptr);
    __san_ptr_infos[SlotId].Size = Size;
  }

  _SAN_ATTRS static void unregisterHost(void *Ptr) {
    FakePtrTy FP(Ptr);
    __san_ptr_infos[FP.U.Enc64.SlotId].Size = 0;
  }

  _SAN_ATTRS static void getHostPtrInfo(uint32_t SlotId, void **BasePtrPtr,
                                        uint32_t *SizePtr) {
    *BasePtrPtr = (void *)(uint64_t)__san_ptr_infos[SlotId].Base;
    *SizePtr = __san_ptr_infos[SlotId].Size;
  }

  _SAN_ATTRS static void getGlobalPtrInfo(void *BasePtr,
                                          uint64_t *LocationIdPtr) {
    for (int32_t I = 0, E = __san_global_info_cnt; I != E; ++I) {
      if (__san_global_infos[I].Base == BasePtr) {
        *LocationIdPtr = __san_global_infos[I].LocationId;
        return;
      }
    }
    *LocationIdPtr = -1;
  }

  _SAN_ATTRS
  operator void *() const { return U.VPtr; }

  template <uint32_t AS>
  _SAN_ATTRS ASPtrTy<AS> check(uint64_t PC, uint64_t LocationId,
                               uint32_t SizeAndKind) {
    uint32_t Size = (SizeAndKind << 3) >> 3;
    uint64_t MaxOffset = int64_t(U.Enc32.Offset) + uint64_t(Size);
    if (MaxOffset < Size || MaxOffset > uint64_t(U.Enc32.Size))
      raiseExecutionError(SanitizerEnvironmentTy::OUT_OF_BOUNDS, PC, LocationId,
                          *this, SizeAndKind);
    return unpack<AS>(PC);
  }

  template <>
  _SAN_ATTRS GlobalPtrTy check<GlobalAS>(uint64_t PC, uint64_t LocationId,
                                         uint32_t SizeAndKind) {
    uint32_t SlotId = U.Enc64.SlotId;
    return checkWithBase<GlobalAS>(PC, LocationId, SizeAndKind,
                                   __san_ptr_infos[SlotId]);
  }

  template <uint32_t AS>
  _SAN_ATTRS ASPtrTy<AS> checkWithBase(uint64_t PC, uint64_t LocationId,
                                       uint32_t SizeAndKind, PtrInfoTy PI) {
    static_assert(AS == GlobalAS, "");
    int64_t Offset = int64_t(U.Enc64.Offset);
    uint32_t Size = (SizeAndKind << 3) >> 3;
    uint64_t MaxOffset = Offset + uint64_t(Size);
    if (MaxOffset < Size || MaxOffset > PI.Size)
      raiseExecutionError(SanitizerEnvironmentTy::OUT_OF_BOUNDS, PC, LocationId,
                          *this, SizeAndKind);
    auto *Base = (ASPtrTy<AS>)PI.Base;
    return Base + Offset;
  }

  _SAN_ATTRS PtrInfoTy getPtrInfo() {
    uint32_t SlotId = U.Enc64.SlotId;
    return __san_ptr_infos[SlotId];
  }

  template <uint32_t AS> _SAN_ATTRS ASPtrTy<AS> unpack(uint64_t PC) {
    auto *PtrBase = (ASPtrTy<AS>)(uint64_t(U.Enc32.RealPtr));
    return PtrBase + uint64_t(U.Enc32.Offset);
  }

  template <> _SAN_ATTRS GlobalPtrTy unpack<GlobalAS>(uint64_t PC) {
    uint64_t Offset = uint64_t(U.Enc64.Offset);
    uint32_t SlotId = U.Enc64.SlotId;
    auto *Base = (GlobalPtrTy)__san_ptr_infos[SlotId].Base;
    return Base + Offset;
  }

  _SAN_ATTRS
  uint32_t getAS() {
    switch (U.Enc32.RealAS) {
    case GlobalAS:
      return U.Enc64.Magic == FAKE_PTR_MAGIC ? GlobalAS : ~0;
    case SharedAS:
      return U.Enc32.Magic == FAKE_PTR_MAGIC ? SharedAS : ~0;
    case AllocaAS:
      return U.Enc32.Magic == FAKE_PTR_MAGIC ? AllocaAS : ~0;
    }
    return ~0;
  }
};

_SAN_ATTRS void setAccessInfo(SanitizerEnvironmentTy &SE, const FakePtrTy &FP,
                              uint32_t AccessSize) {
  SE.FP.VPtr = FP;
  SE.AccessSize = AccessSize;
}

#pragma omp begin declare variant match(device = {arch(amdgcn)})
// TODO: On NVIDIA we currently use 64 bit pointers, see -fcuda-short-ptr
static_assert(sizeof(AllocaPtrTy) == 4,
              "Can only handle 32 bit pointers for now!");
#pragma omp end declare variant

} // namespace

extern "C" {
_SAN_ENTRY_ATTRS void __offload_san_trap_info(uint64_t PC,
                                              uint64_t LocationId) {
  raiseExecutionError(SanitizerEnvironmentTy::TRAP, PC, LocationId,
                      FakePtrTy());
}

_SAN_ENTRY_ATTRS void __offload_san_unreachable_info(uint64_t PC,
                                                     uint64_t LocationId) {
  raiseExecutionError(SanitizerEnvironmentTy::UNREACHABLE, PC, LocationId,
                      FakePtrTy());
}

_SAN_ENTRY_ATTRS void *__offload_san_register_alloca(uint64_t PC,
                                                     uint64_t LocationId,
                                                     AllocaPtrTy Ptr,
                                                     uint64_t Size) {
  return FakePtrTy::create<AllocaAS>(PC, LocationId, Ptr, Size);
}

_SAN_ENTRY_ATTRS void *__offload_san_register_malloc(uint64_t PC,
                                                     uint64_t LocationId,
                                                     GlobalPtrTy Ptr,
                                                     uint64_t Size) {
  return FakePtrTy::create<GlobalAS>(PC, LocationId, Ptr, Size);
}

_SAN_ENTRY_ATTRS void __offload_san_register_host(void *Ptr, uint64_t Size,
                                                  uint32_t SlotId) {
  FakePtrTy::registerHost(Ptr, Size, SlotId);
}

_SAN_ENTRY_ATTRS void __offload_san_unregister_host(void *Ptr) {
  FakePtrTy::unregisterHost(Ptr);
}

_SAN_ENTRY_ATTRS void __offload_san_get_global_info(void *Ptr,
                                                    uint64_t *LocationIdPtr) {
  FakePtrTy::getGlobalPtrInfo(Ptr, LocationIdPtr);
}

_SAN_ENTRY_ATTRS void __offload_san_get_ptr_info(uint32_t SlotId,
                                                 void **BasePtrPtr,
                                                 uint32_t *SizePtr,
                                                 uint64_t *LocationIdPtr) {
  FakePtrTy::getHostPtrInfo(SlotId, BasePtrPtr, SizePtr);
  __offload_san_get_global_info(*BasePtrPtr, LocationIdPtr);
}

#define CHECK_FOR_AS(AS)                                                       \
  _SAN_ENTRY_ATTRS ASPtrTy<AS> __offload_san_unpack_as##AS(                    \
      uint64_t PC, uint64_t LocationId, void *FakePtr) {                       \
    FakePtrTy FP(FakePtr, AS, /* Checked */ AS == GlobalAS, PC, LocationId);   \
    if constexpr (AS == GlobalAS)                                              \
      return FP.unpack<AS>(PC);                                                \
    if constexpr (AS == SharedAS)                                              \
      return FP.unpack<AS>(PC);                                                \
    if constexpr (AS == AllocaAS)                                              \
      return FP.unpack<AS>(PC);                                                \
    return (ASPtrTy<AS>)FakePtr;                                               \
  }                                                                            \
                                                                               \
  _SAN_ENTRY_ATTRS PtrASInfoTy __offload_san_get_as##AS##_info(                \
      uint64_t PC, uint64_t LocationId, void *FakePtr) {                       \
    FakePtrTy FP(FakePtr, AS, true, PC, LocationId);                           \
    if constexpr (AS == GlobalAS)                                              \
      return {FP.getPtrInfo(), GlobalAS};                                      \
    return {{}, AS};                                                           \
  }                                                                            \
                                                                               \
  _SAN_ENTRY_ATTRS ASPtrTy<AS> __offload_san_check_as##AS##_access_with_info(  \
      uint64_t PC, uint64_t LocationId, void *FakePtr, uint32_t Size,          \
      uint32_t AllocAS, char *AllocBase, uint64_t AllocSize) {                 \
    if constexpr (AS == GlobalAS) {                                            \
      FakePtrTy FP(FakePtr, AS, /* Checked */ false, PC, LocationId);          \
      return FP.checkWithBase<AS>(PC, LocationId, Size,                        \
                                  {AllocBase, AllocSize});                     \
    }                                                                          \
    if constexpr (AS == SharedAS) {                                            \
      FakePtrTy FP(FakePtr, AS, /* Checked */ true, PC, LocationId);           \
      return FP.check<AS>(PC, LocationId, Size);                               \
    }                                                                          \
    if constexpr (AS == AllocaAS) {                                            \
      FakePtrTy FP(FakePtr, AS, /* Checked */ true, PC, LocationId);           \
      return FP.check<AS>(PC, LocationId, Size);                               \
    }                                                                          \
    return (ASPtrTy<AS>)FakePtr;                                               \
  }                                                                            \
  _SAN_ENTRY_ATTRS void *__offload_san_register_as##AS##_global(               \
      uint64_t PC, uint64_t LocationId, ASPtrTy<AS> Ptr, uint64_t Size) {      \
    return (void *)FakePtrTy::registerGlobal<AS>(PC, LocationId, Ptr, Size);   \
  }

CHECK_FOR_AS(1)
CHECK_FOR_AS(3)
CHECK_FOR_AS(4)
CHECK_FOR_AS(5)
#undef CHECK_FOR_AS

_SAN_ENTRY_ATTRS void *
__offload_san_unpack_as0(uint64_t PC, uint64_t LocationId, void *FakePtr) {
  FakePtrTy FP(FakePtr);
  if (FP.getAS() == GlobalAS)
    return (void *)FP.unpack<GlobalAS>(PC);
  if (FP.getAS() == SharedAS)
    return (void *)FP.unpack<SharedAS>(PC);
  if (FP.getAS() == AllocaAS)
    return (void *)FP.unpack<AllocaAS>(PC);
  return FakePtr;
}

_SAN_ENTRY_ATTRS PtrASInfoTy __offload_san_get_as0_info(uint64_t PC,
                                                        uint64_t LocationId,
                                                        void *FakePtr) {
  FakePtrTy FP(FakePtr);
  if (FP.getAS() == GlobalAS)
    return __offload_san_get_as1_info(PC, LocationId, FakePtr);
  if (FP.getAS() == SharedAS)
    return __offload_san_get_as3_info(PC, LocationId, FakePtr);
  if (FP.getAS() == AllocaAS)
    return __offload_san_get_as5_info(PC, LocationId, FakePtr);
  return {{}, 0};
}

_SAN_ENTRY_ATTRS void *__offload_san_check_as0_access_with_info(
    uint64_t PC, uint64_t LocationId, void *FakePtr, uint32_t Size,
    uint32_t InfoAS, char *InfoBase, uint64_t InfoSize) {
  if (InfoAS == GlobalAS) {
    FakePtrTy FP(FakePtr, GlobalAS, false, PC, LocationId);
    return (void *)FP.checkWithBase<GlobalAS>(PC, LocationId, Size,
                                              {InfoBase, InfoSize});
  }
  if (InfoAS == SharedAS) {
    FakePtrTy FP(FakePtr, SharedAS, false, PC, LocationId);
    return (void *)FP.check<SharedAS>(PC, LocationId, Size);
  }
  if (InfoAS == AllocaAS) {
    FakePtrTy FP(FakePtr, AllocaAS, false, PC, LocationId);
    return (void *)FP.check<AllocaAS>(PC, LocationId, Size);
  }
  return FakePtr;
}

_SAN_ENTRY_ATTRS void __offload_san_leak_check() {}
}

#pragma omp end declare target
