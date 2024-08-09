//===------ Sanitizer.cpp - Track allocation for sanitizer checks ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "Mapping.h"
#include "Shared/Environment.h"
#include "Synchronization.h"
#include "Types.h"
#include "Utils.h"

using namespace ompx;

#define _SAN_ATTRS                                                             \
  [[clang::disable_sanitizer_instrumentation, gnu::used, gnu::retain]]
#define _SAN_ENTRY_ATTRS [[gnu::flatten, gnu::always_inline]] _SAN_ATTRS

#pragma omp begin declare target device_type(nohost)

[[gnu::visibility("protected")]] _SAN_ATTRS SanitizerEnvironmentTy
    *__sanitizer_environment_ptr;
namespace {

enum {
  MallocAS = 1,
  AllocaAS = 5,
};

struct PtrInfoTy {
  char [[clang::address_space(MallocAS)]] * Base;
  uint64_t Size;
};
static PtrInfoTy PtrInfos[1024];
static uint32_t PtrInfoCnt = 0;

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
[[clang::noinline]] _SAN_ATTRS void spinWait(SanitizerEnvironmentTy &SE) {
  while (!atomic::load(&SE.IsInitialized, atomic::OrderingTy::aquire))
    ;
  //  __builtin_trap();
}

_SAN_ATTRS
void setLocation(SanitizerEnvironmentTy &SE, uint64_t PC) {
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

_SAN_ATTRS
void raiseExecutionError(SanitizerEnvironmentTy::ErrorCodeTy ErrorCode,
                         uint64_t PC) {
  SanitizerEnvironmentTy &SE = *__sanitizer_environment_ptr;
  bool HasLock = getSanitizerEnvironmentLock(SE, ErrorCode);

  // If no thread of this warp has the lock, end execution gracefully.
  bool AnyThreadHasLock = utils::ballotSync(lanes::All, HasLock);
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

  static constexpr uint32_t MAGIC = 0b1111000;
  static constexpr uint32_t BITS_OFFSET = 11;

  union {
    void *VPtr;
    struct {
      uint32_t Offset : BITS_OFFSET;
      uint32_t Size : BITS_OFFSET;
      uint32_t RealAS : 3;
      uint32_t Magic : 7;
      uint32_t RealPtr : 32;
    } Enc32;
    struct {
      uint32_t Offset : BITS_OFFSET * 2;
      uint32_t RealAS : 3;
      uint32_t Magic : 7;
      uint32_t SlotId : 32;
    } Enc64;
  } U;

  static_assert(sizeof(U) == sizeof(void *), "Encoding is too large!");

  _SAN_ATTRS
  FakePtrTy() { U.VPtr = nullptr; }

  _SAN_ATTRS
  FakePtrTy(void *FakePtr) { U.VPtr = FakePtr; }

  _SAN_ATTRS
  FakePtrTy(void *FakePtr, int32_t AS) : FakePtrTy(FakePtr) {
    if (U.Enc32.Magic != MAGIC)
      raiseExecutionError(SanitizerEnvironmentTy::BAD_PTR, 0);
    if (U.Enc32.RealAS != AS)
      raiseExecutionError(SanitizerEnvironmentTy::AS_MISMATCH, AS);
  }

  _SAN_ATTRS
  uint32_t getMaxSize() { return (1u << BITS_OFFSET) - 1; }

  template <uint32_t AS>
  _SAN_ATTRS static FakePtrTy create(void [[clang::address_space(AS)]] * Ptr,
                                     uint32_t Size) {
    FakePtrTy FP;
    if (Size > FP.getMaxSize())
      raiseExecutionError(SanitizerEnvironmentTy::ALLOCATION_TOO_LARGE, Size);

    FP.U.Enc32.Size = Size;
    FP.U.Enc32.RealAS = AS;
    FP.U.Enc32.Magic = MAGIC;
    FP.U.Enc32.RealPtr = uint64_t(Ptr);

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
    FP.U.Enc64.Magic = MAGIC;
    FP.U.Enc64.SlotId = SlotId;

    return FP;
  }

  _SAN_ATTRS
  operator void *() { return U.VPtr; }

  template <uint32_t AS>
      _SAN_ATTRS void [[clang::address_space(AS)]] *
      check(uint64_t PC, uint32_t Size) {
    uint64_t MaxOffset = uint64_t(U.Enc32.Offset) + uint64_t(Size);
    if (MaxOffset > uint64_t(U.Enc32.Size))
      raiseExecutionError(SanitizerEnvironmentTy::OUT_OF_BOUNDS, PC);
    return (char [[clang::address_space(AS)]] *)(uint64_t(U.Enc32.RealPtr)) +
           U.Enc32.Offset;
  }

  template <>
      _SAN_ATTRS void [[clang::address_space(MallocAS)]] *
      check<MallocAS>(uint64_t PC, uint32_t Size) {
    uint64_t Offset = uint64_t(U.Enc64.Offset);
    uint64_t MaxOffset = Offset + uint64_t(Size);
    uint32_t SlotId = U.Enc64.SlotId;
    auto *Base = PtrInfos[SlotId].Base;
    uint64_t AllocSize = PtrInfos[SlotId].Size;
    if (MaxOffset > AllocSize)
      raiseExecutionError(SanitizerEnvironmentTy::OUT_OF_BOUNDS, PC);
    return Base + Offset;
  }

  template <uint32_t AS>
      _SAN_ATTRS void [[clang::address_space(AS)]] * unpack(uint64_t PC) {
    return (char [[clang::address_space(AS)]] *)(uint64_t(U.Enc32.RealPtr)) +
           U.Enc32.Offset;
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
    if (U.Enc32.Magic == MAGIC)
      return U.Enc32.RealAS;
    return ~0;
  }

  _SAN_ATTRS
  uint32_t getMagic() { return U.Enc32.Magic; }
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

_SAN_ENTRY_ATTRS void *__offload_san_unpack(uint64_t PC, void *FakePtr) {
  FakePtrTy FP(FakePtr);
  /* Only AllocaAS is actually sanitized for now */
  if (FP.getAS() == AllocaAS)
    return (void *)FP.unpack<AllocaAS>(PC);
  if (FP.getAS() == MallocAS)
    return (void *)FP.unpack<MallocAS>(PC);
  return FakePtr;
}

#define CHECK_FOR_AS(AS)                                                       \
  _SAN_ENTRY_ATTRS void [[clang::address_space(AS)]] *                         \
      __offload_san_check_as##AS##_access(uint64_t PC, void *FakePtr,          \
                                          uint32_t Size) {                     \
    if constexpr (AS == AllocaAS) {                                            \
      FakePtrTy FP(FakePtr, AS);                                               \
      return FP.check<AS>(PC, Size);                                           \
    }                                                                          \
    if constexpr (AS == MallocAS) {                                            \
      FakePtrTy FP(FakePtr, AS);                                               \
      return FP.check<AS>(PC, Size);                                           \
    }                                                                          \
    return (void [[clang::address_space(AS)]] *)FakePtr;                       \
  }

CHECK_FOR_AS(1)
CHECK_FOR_AS(3)
CHECK_FOR_AS(4)
CHECK_FOR_AS(5)

_SAN_ENTRY_ATTRS void *
__offload_san_check_as0_access(uint64_t PC, void *FakePtr, uint32_t Size) {
  FakePtrTy FP(FakePtr);
  if (FP.getAS() == AllocaAS)
    return (void *)__offload_san_check_as5_access(PC, FakePtr, Size);
  if (FP.getAS() == MallocAS)
    return (void *)__offload_san_check_as1_access(PC, FakePtr, Size);
  return FakePtr;
}
}

#pragma omp end declare target
