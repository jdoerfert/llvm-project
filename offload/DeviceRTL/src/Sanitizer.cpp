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
#define _SAN_ATTRS                                                             \
  [[clang::disable_sanitizer_instrumentation, gnu::used, gnu::retain]]
#define _SAN_ENTRY_ATTRS _FLATTEN _INLINE _SAN_ATTRS

#pragma omp begin declare target device_type(nohost)

[[gnu::visibility("protected")]] _SAN_ATTRS SanitizerEnvironmentTy
    *__sanitizer_environment_ptr;
namespace {

enum {
  MallocAS = 1,
  AllocaAS = 5,
};

struct __attribute__((packed)) PtrInfoTy {
  char [[clang::address_space(MallocAS)]] * Base;
  uint64_t Size;
};
static PtrInfoTy PtrInfos[1 << 14];
static uint32_t PtrInfoCnt = 0;

struct __attribute__((packed)) InfoTy {
  PtrInfoTy PI;
  uint32_t AS;
};

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

_SAN_ATTRS void setLocation(SanitizerEnvironmentTy &SE, uint64_t PC) {
  for (int I = 0; I < 3; ++I) {
    SE.ThreadId[I] = mapping::getThreadIdInBlock(I);
    SE.BlockId[I] = mapping::getBlockIdInKernel(I);
  }
  SE.PC = PC;

  // This is the last step to initialize the sanitizer environment, time to
  // trap via the spinWait. Flush the memory writes and signal for the end.
  fence::system(atomic::OrderingTy::release);
  atomic::store(&SE.IsInitialized, 1, atomic::OrderingTy::release);
}

_SAN_ATTRS _FLATTEN _INLINE void
raiseExecutionError(SanitizerEnvironmentTy::ErrorCodeTy ErrorCode,
                    uint64_t PC) {
  SanitizerEnvironmentTy &SE = *__sanitizer_environment_ptr;
  bool HasLock = getSanitizerEnvironmentLock(SE, ErrorCode);

  // If no thread of this warp has the lock, end execution gracefully.
  // bool AnyThreadHasLock = utils::ballotSync(lanes::All, HasLock);
  //  if (!AnyThreadHasLock)
  //    utils::terminateWarp();

  // One thread will set the location information and signal that the rest of
  // the wapr that the actual trap can be executed now.
  if (HasLock)
    setLocation(SE, PC);

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
  FakePtrTy(void *FakePtr, int32_t AS, bool Checked = false, uint64_t PC = 0)
      : FakePtrTy(FakePtr) {
    if (Checked)
      return;
    bool BadAS = U.Enc32.RealAS != AS;
    bool BadMagic =
        (AS == MallocAS ? U.Enc64.Magic : U.Enc32.Magic) != FAKE_PTR_MAGIC;
    if (BadMagic | BadAS)
      raiseExecutionError(BadMagic ? SanitizerEnvironmentTy::BAD_PTR
                                   : SanitizerEnvironmentTy::AS_MISMATCH,
                          uint64_t(PC));
  }

  _SAN_ATTRS
  uint32_t getMaxSize() { return (1u << FAKE_PTR_BASE_BITS_OFFSET) - 1; }

  template <uint32_t AS>
  _SAN_ATTRS static FakePtrTy create(void [[clang::address_space(AS)]] * Ptr,
                                     uint32_t Size) {
    FakePtrTy FP;
    if (Size > FP.getMaxSize())
      raiseExecutionError(SanitizerEnvironmentTy::ALLOCATION_TOO_LARGE, Size);

    FP.U.Enc32.RealAS = AS;
    FP.U.Enc32.RealPtr = uint64_t(Ptr);
    FP.U.Enc32.Size = Size;
    FP.U.Enc32.Magic = FAKE_PTR_MAGIC;

    return FP;
  }

  template <>
  _SAN_ATTRS FakePtrTy create<MallocAS>(
      void [[clang::address_space(MallocAS)]] * Ptr, uint32_t Size) {
    FakePtrTy FP;

    uint32_t SlotId = PtrInfoCnt++;
    PtrInfos[SlotId].Base = decltype(PtrInfoTy::Base)(Ptr);
    PtrInfos[SlotId].Size = Size;
    FP.U.Enc64.RealAS = MallocAS;
    FP.U.Enc64.SlotId = SlotId;
    FP.U.Enc64.Magic = FAKE_PTR_MAGIC;

    return FP;
  }

  _SAN_ATTRS static void registerHost(void *Ptr, uint32_t Size,
                                      uint32_t SlotId) {
    if (SlotId >= (1 << 14))
      raiseExecutionError(SanitizerEnvironmentTy::ALLOCATION_TOO_LARGE, SlotId);

    PtrInfos[SlotId].Base = decltype(PtrInfoTy::Base)(Ptr);
    PtrInfos[SlotId].Size = Size;
  }

  _SAN_ATTRS static void unregisterHost(void *Ptr) {
    FakePtrTy FP(Ptr);
    PtrInfos[FP.U.Enc64.SlotId].Size = 0;
  }

  _SAN_ATTRS
  operator void *() { return U.VPtr; }

  template <uint32_t AS>
      _SAN_ATTRS void [[clang::address_space(AS)]] *
      check(uint64_t PC, uint32_t Size) {
    uint64_t MaxOffset = int64_t(U.Enc32.Offset) + uint64_t(Size);
    if (MaxOffset < Size || MaxOffset > uint64_t(U.Enc32.Size))
      raiseExecutionError(SanitizerEnvironmentTy::OUT_OF_BOUNDS, PC);
    return unpack<AS>(PC);
  }

  template <>
      _SAN_ATTRS void [[clang::address_space(MallocAS)]] *
      check<MallocAS>(uint64_t PC, uint32_t Size) {
    uint32_t SlotId = U.Enc64.SlotId;
    return checkWithBase<MallocAS>(PC, Size, PtrInfos[SlotId]);
  }

  template <uint32_t AS>
      _SAN_ATTRS void [[clang::address_space(AS)]] *
      checkWithBase(uint64_t PC, uint32_t Size, PtrInfoTy PI) {
    static_assert(AS == MallocAS, "");
    int64_t Offset = int64_t(U.Enc64.Offset);
    uint64_t MaxOffset = Offset + uint64_t(Size);
    if (MaxOffset < Size || MaxOffset > PI.Size)
      raiseExecutionError(SanitizerEnvironmentTy::OUT_OF_BOUNDS, PC);
    return PI.Base + Offset;
  }

  _SAN_ATTRS void advance(int64_t Offset) {
    //    if constexpr (AS == MallocAS) {
    U.Enc64.Offset += Offset;
    //    } else {
    //      U.Enc32.Offset += Offset;
    //    }
  }

  _SAN_ATTRS PtrInfoTy getPtrInfo() {
    uint32_t SlotId = U.Enc64.SlotId;
    return PtrInfos[SlotId];
  }

  template <uint32_t AS>
      _SAN_ATTRS void [[clang::address_space(AS)]] * unpack(uint64_t PC) {
    auto *PtrBase =
        (char [[clang::address_space(AS)]] *)(uint64_t(U.Enc32.RealPtr));
    return PtrBase + uint64_t(U.Enc32.Offset);
  }

  template <>
      _SAN_ATTRS void [[clang::address_space(MallocAS)]] *
      unpack<MallocAS>(uint64_t PC) {
    uint64_t Offset = uint64_t(U.Enc64.Offset);
    uint32_t SlotId = U.Enc64.SlotId;
    auto *Base = PtrInfos[SlotId].Base;
    return Base + Offset;
  }

  _SAN_ATTRS
  uint32_t getAS() {
    switch (U.Enc32.RealAS) {
    case MallocAS:
      return U.Enc64.Magic == FAKE_PTR_MAGIC ? MallocAS : ~0;
    case AllocaAS:
      return U.Enc32.Magic == FAKE_PTR_MAGIC ? AllocaAS : ~0;
    }
    return ~0;
  }
};

#pragma omp begin declare variant match(device = {arch(amdgcn)})
// TODO: On NVIDIA we currently use 64 bit pointers, see -fcuda-short-ptr
static_assert(sizeof(void [[clang::address_space(AllocaAS)]] *) == 4,
              "Can only handle 32 bit pointers for now!");
#pragma omp end declare variant

} // namespace

extern "C" {
_SAN_ENTRY_ATTRS void __offload_san_trap_info(uint64_t PC) {
  raiseExecutionError(SanitizerEnvironmentTy::TRAP, PC);
}

_SAN_ENTRY_ATTRS void __offload_san_unreachable_info(uint64_t PC) {
  raiseExecutionError(SanitizerEnvironmentTy::UNREACHABLE, PC);
}

_SAN_ENTRY_ATTRS void *__offload_san_register_alloca(
    uint64_t PC, void [[clang::address_space(AllocaAS)]] * Ptr, uint32_t Size) {
  return FakePtrTy::create<AllocaAS>(Ptr, Size);
}

_SAN_ENTRY_ATTRS void *__offload_san_register_malloc(
    uint64_t PC, void [[clang::address_space(MallocAS)]] * Ptr, uint32_t Size) {
  return FakePtrTy::create<MallocAS>(Ptr, Size);
}

_SAN_ENTRY_ATTRS void __offload_san_register_host(void *Ptr, uint32_t Size,
                                                  uint32_t Slot) {
  FakePtrTy::registerHost(Ptr, Size, Slot);
}
_SAN_ENTRY_ATTRS void __offload_san_unregister_host(void *Ptr) {
  FakePtrTy::unregisterHost(Ptr);
}

#define CHECK_FOR_AS(AS)                                                       \
  _SAN_ENTRY_ATTRS void [[clang::address_space(AS)]] *                         \
      __offload_san_unpack_as##AS(uint64_t PC, void *FakePtr) {                \
    FakePtrTy FP(FakePtr, AS, true, PC);                                       \
    if constexpr (AS == MallocAS)                                              \
      return FP.unpack<AS>(PC);                                                \
    if constexpr (AS == AllocaAS)                                              \
      return FP.unpack<AS>(PC);                                                \
    return (void [[clang::address_space(AS)]] *)FakePtr;                       \
  }                                                                            \
                                                                               \
  _SAN_ENTRY_ATTRS InfoTy __offload_san_get_as##AS##_info(uint64_t PC,         \
                                                          void *FakePtr) {     \
    FakePtrTy FP(FakePtr, AS, true, PC);                                       \
    if constexpr (AS == MallocAS)                                              \
      return {FP.getPtrInfo(), MallocAS};                                      \
    return {{}, AS};                                                           \
  }                                                                            \
                                                                               \
  _SAN_ENTRY_ATTRS void [[clang::address_space(AS)]] *                         \
      __offload_san_check_as##AS##_access_with_info(                           \
          uint64_t PC, void *FakePtr, uint32_t Size, uint32_t AllocAS,         \
          char [[clang::address_space(MallocAS)]] * AllocBase,                 \
          uint64_t AllocSize) {                                                \
    if constexpr (AS == MallocAS) {                                            \
      FakePtrTy FP(FakePtr, AS, false, PC);                                    \
      return FP.checkWithBase<AS>(PC, Size, {AllocBase, AllocSize});           \
    }                                                                          \
    if constexpr (AS == AllocaAS) {                                            \
      FakePtrTy FP(FakePtr, AS, false, PC);                                    \
      return FP.check<AS>(PC, Size);                                           \
    }                                                                          \
    return (void [[clang::address_space(AS)]] *)FakePtr;                       \
  }

CHECK_FOR_AS(1)
CHECK_FOR_AS(3)
CHECK_FOR_AS(4)
CHECK_FOR_AS(5)
#undef CHECK_FOR_AS

_SAN_ENTRY_ATTRS void *__offload_san_unpack_as0(uint64_t PC, void *FakePtr) {
  FakePtrTy FP(FakePtr);
  if (FP.getAS() == MallocAS)
    return (void *)FP.unpack<MallocAS>(PC);
  if (FP.getAS() == AllocaAS)
    return (void *)FP.unpack<AllocaAS>(PC);
  return FakePtr;
}

_SAN_ENTRY_ATTRS InfoTy __offload_san_get_as0_info(uint64_t PC, void *FakePtr) {
  FakePtrTy FP(FakePtr);
  if (FP.getAS() == MallocAS)
    return __offload_san_get_as1_info(PC, FakePtr);
  if (FP.getAS() == AllocaAS)
    return __offload_san_get_as5_info(PC, FakePtr);
  return {{}, 0};
}

_SAN_ENTRY_ATTRS void *__offload_san_check_as0_access_with_info(
    uint64_t PC, void *FakePtr, uint32_t Size, uint32_t InfoAS,
    char [[clang::address_space(MallocAS)]] * InfoBase, uint64_t InfoSize) {
  if (InfoAS == MallocAS) {
    FakePtrTy FP(FakePtr, MallocAS, false, PC);
    return (void *)FP.checkWithBase<MallocAS>(PC, Size, {InfoBase, InfoSize});
  }
  if (InfoAS == AllocaAS) {
    FakePtrTy FP(FakePtr, AllocaAS, false, PC);
    return (void *)FP.check<AllocaAS>(PC, Size);
  }
  return FakePtr;
}

_SAN_ENTRY_ATTRS void *__offload_san_gep(void *FakePtr, int64_t Offset) {
  FakePtrTy FP(FakePtr);
  FP.advance(Offset);
  return FP;
}
}

#pragma omp end declare target
