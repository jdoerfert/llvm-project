//===- Synchronization.cpp - OpenMP Device synchronization API ---- c++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Include all synchronization.
//
//===----------------------------------------------------------------------===//

#include "Synchronization.h"

#include "Mapping.h"
#include "State.h"
#include "Types.h"
#include "Utils.h"

using namespace _OMP;

//namespace {

/// Fallback implementations are partially missing to trigger a link time error.
/// Implementations for new devices, including the host, should go into a
/// dedicated begin/end declare variant.
///
///{

void namedBarrierImpl();
void fenceTeamImpl(int Ordering);
void fenceKernelImpl(int Ordering);
void fenceSystemImpl(int Ordering);
void syncThreadsImpl();
void syncWarpImpl(LaneMaskTy Mask);

void namedBarrierInitImpl() {}

uint32_t atomicReadImpl(uint32_t *Address, int Ordering) {
  return __atomic_load_n(Address, Ordering);
}

uint32_t atomicIncImpl(uint32_t *Address, uint32_t Val, int Ordering);

uint32_t atomicAddImpl(uint32_t *Address, uint32_t Val, int Ordering) {
  return __atomic_fetch_add(Address, Val, Ordering);
}
uint32_t atomicMaxImpl(uint32_t *Address, uint32_t Val, int Ordering) {
  return __atomic_fetch_max(Address, Val, Ordering);
}

uint32_t atomicExchangeImpl(uint32_t *Address, uint32_t Val, int Ordering) {
  uint32_t R;
  __atomic_exchange(Address, &Val, &R, Ordering);
  return R;
}
uint32_t atomicCASImpl(uint32_t *Address, uint32_t Compare, uint32_t Val,
                       int Ordering) {
  (void)__atomic_compare_exchange(Address, &Compare, &Val, false, Ordering,
                                  __ATOMIC_ACQUIRE);
  return Compare;
}

uint64_t atomicAddImpl(uint64_t *Address, uint64_t Val, int Ordering) {
  return __atomic_fetch_add(Address, Val, Ordering);
}

void initLockImpl(omp_lock_t *Lock);

void destoryLockImpl(omp_lock_t *Lock);

void setLockImpl(omp_lock_t *Lock);

void unsetLockImpl(omp_lock_t *Lock);

int testLockImpl(omp_lock_t *Lock);

///}

/// AMDGCN implementations of the shuffle sync idiom.
///
///{
#pragma omp begin declare variant match(device = {arch(amdgcn)})

uint32_t SHARD(namedBarrierTracker);

void namedBarrierInitImpl() {
  // Don't have global ctors, and shared memory is not zero init
  atomic::store(&namedBarrierTracker, 0u, __ATOMIC_RELEASE);
}

void namedBarrierImpl() {
  uint32_t NumThreads = omp_get_num_threads();
  // assert(NumThreads % 32 == 0);

  uint32_t WarpSize = maping::getWarpSize();
  uint32_t NumWaves = NumThreads / WarpSize;

  fence::team(__ATOMIC_ACQUIRE);

  // named barrier implementation for amdgcn.
  // Uses two 16 bit unsigned counters. One for the number of waves to have
  // reached the barrier, and one to count how many times the barrier has been
  // passed. These are packed in a single atomically accessed 32 bit integer.
  // Low bits for the number of waves, assumed zero before this call.
  // High bits to count the number of times the barrier has been passed.

  // precondition: NumWaves != 0;
  // invariant: NumWaves * WarpSize == NumThreads;
  // precondition: NumWaves < 0xffffu;

  // Increment the low 16 bits once, using the lowest active thread.
  if (mapping::isLeaderInWarp()) {
    uint32_t load = atomic::add(&namedBarrierTracker, 1,
                                __ATOMIC_RELAXED); // commutative

    // Record the number of times the barrier has been passed
    uint32_t generation = load & 0xffff0000u;

    if ((load & 0x0000ffffu) == (NumWaves - 1)) {
      // Reached NumWaves in low bits so this is the last wave.
      // Set low bits to zero and increment high bits
      load += 0x00010000u; // wrap is safe
      load &= 0xffff0000u; // because bits zeroed second

      // Reset the wave counter and release the waiting waves
      atomic::store(&namedBarrierTracker, load, __ATOMIC_RELAXED);
    } else {
      // more waves still to go, spin until generation counter changes
      do {
        __builtin_amdgcn_s_sleep(0);
        load = atomi::load(&namedBarrierTracker, __ATOMIC_RELAXED);
      } while ((load & 0xffff0000u) == generation);
    }
  }
  fence::team(__ATOMIC_RELEASE);
}

void syncWarpImpl(__kmpc_impl_lanemask_t) {
  // AMDGCN doesn't need to sync threads in a warp
}

void syncThreadsImpl() { __builtin_amdgcn_s_barrier(); }

void fenceTeamImpl(int Ordering) {
  __builtin_amdgcn_fence(Ordering, "workgroup");
}

void fenceKernel(int Ordering) { __builtin_amdgcn_fence(Ordering, "agent"); }

void fenceSystemImpl(int Ordering) { __builtin_amdgcn_fence(Ordering, ""); }

uint32_t atomicIncImpl(uint32_t *Address, uint32_t Val, int Ordering) {
  return __builtin_amdgcn_atomic_inc32(Address, Val, Ordering, "");
}

#pragma omp end declare variant
///}

/// NVPTX implementations of the shuffle and shuffle sync idiom.
///
///{
#pragma omp begin declare variant match(                                       \
    device = {arch(nvptx, nvptx64)}, implementation = {extension(match_any)})

void namedBarrierImpl() {
  uint32_t NumThreads = omp_get_num_threads();
  // assert(NumThreads % 32 == 0);

  // The named barrier for active parallel threads of a team in an L1 parallel
  // region to synchronize with each other.
  int BarrierNo = 7;
  asm volatile("bar.sync %0, %1;"
               :
               : "r"(BarrierNo), "r"(NumThreads)
               : "memory");
}

void fenceTeamImpl(int) { __nvvm_membar_cta(); }
void fenceKernelImpl(int) { __nvvm_membar_gl(); }
void fenceSystemImpl(int) { __nvvm_membar_sys(); }

void syncWarpImpl(__kmpc_impl_lanemask_t Mask) { __nvvm_bar_warp_sync(Mask); }
void syncThreadsImpl() { __syncthreads(); }

constexpr uint32_t OMP_SPIN = 1000;
constexpr uint32_t UNSET = 0;
constexpr uint32_t SET = 1;

// TODO: This seems to hide a bug in the declare variant handling. If it is called before it is defined
//       here the overload won't happen. Investigate lalter!
void unsetLockImpl(omp_lock_t *Lock) { (void)atomicExchangeImpl((uint32_t*)Lock, UNSET, __ATOMIC_SEQ_CST); }

int testLockImpl(omp_lock_t *Lock) { return atomicAddImpl((uint32_t*)Lock, 0u, __ATOMIC_SEQ_CST); }

void initLockImpl(omp_lock_t *Lock) { unsetLockImpl(Lock); }

void destoryLockImpl(omp_lock_t *Lock) { unsetLockImpl(Lock); }

void setLockImpl(omp_lock_t *Lock) {
  // TODO: not sure spinning is a good idea here..
  while (atomicCASImpl((uint32_t*)Lock, UNSET, SET, __ATOMIC_SEQ_CST) != UNSET) {
    int32_t start = __nvvm_read_ptx_sreg_clock();
    int32_t now;
    for (;;) {
      now = __nvvm_read_ptx_sreg_clock();
      int32_t cycles = now > start ? now - start : now + (0xffffffff - start);
      if (cycles >= OMP_SPIN * mapping::getBlockId()) {
        break;
      }
    }
  } // wait for 0 to be the read value
}

uint32_t atomicIncImpl(uint32_t *Address, uint32_t Val, int Ordering) {
  return __nvvm_atom_inc_gen_ui(Address, Val);
}

#pragma omp end declare variant

///}

//} // namespace

#pragma omp declare target

void __kmpc_ordered(IdentTy *Loc, int32_t TId) {}

void __kmpc_end_ordered(IdentTy *Loc, int32_t TId) {}

int32_t __kmpc_cancel_barrier(IdentTy *Loc, int32_t TId) {
  __kmpc_barrier(Loc, TId);
  return 0;
}

void __kmpc_barrier(IdentTy *Loc, int32_t TId) {
  if (mapping::isMainThreadInGenericMode())
    return __kmpc_flush(Loc);

  if (mapping::isSPMDMode())
    return __kmpc_barrier_simple_spmd(Loc, TId);

  namedBarrierImpl();
}

void __kmpc_barrier_simple_spmd(IdentTy *Loc, int32_t TId) {
  synchronize::threads();
}

int32_t __kmpc_master(IdentTy *Loc, int32_t TId) {
  return mapping::getThreadIdInBlock() == 0;
}

void __kmpc_end_master(IdentTy *Loc, int32_t TId) {}

int32_t __kmpc_single(IdentTy *Loc, int32_t TId) {
  return __kmpc_master(Loc, TId);
}

void __kmpc_end_single(IdentTy *Loc, int32_t TId) {
  // The barrier is explicitly called.
}

void __kmpc_flush(IdentTy *Loc) { fence::team(__ATOMIC_SEQ_CST); }

__kmpc_impl_lanemask_t __kmpc_warp_active_thread_mask() {
  return mapping::activemask();
}

void __kmpc_syncwarp(__kmpc_impl_lanemask_t Mask) { synchronize::warp(Mask); }

void __kmpc_critical(IdentTy *Loc, int32_t TId, CriticalNameTy *Name) {
  omp_set_lock(reinterpret_cast<omp_lock_t *>(Name));
}

void __kmpc_end_critical(IdentTy *Loc, int32_t TId, CriticalNameTy *Name) {
  omp_unset_lock(reinterpret_cast<omp_lock_t *>(Name));
}

void omp_init_lock(omp_lock_t *Lock) { initLockImpl(Lock); }

void omp_destroy_lock(omp_lock_t *Lock) { destoryLockImpl(Lock); }

void omp_set_lock(omp_lock_t *Lock) { setLockImpl(Lock); }

void omp_unset_lock(omp_lock_t *Lock) { unsetLockImpl(Lock); }

int omp_test_lock(omp_lock_t *Lock) { return testLockImpl(Lock); }

void synchronize::init(bool IsSPMD) {
  if (!IsSPMD && mapping::getThreadIdInBlock() == 0)
    namedBarrierInitImpl();
}

void synchronize::warp(LaneMaskTy Mask) { syncWarpImpl(Mask); }

void synchronize::threads() { syncThreadsImpl(); }

void fence::team(int Ordering) { fenceTeamImpl(Ordering); }

void fence::kernel(int Ordering) { fenceKernelImpl(Ordering); }

void fence::system(int Ordering) { fenceSystemImpl(Ordering); }

uint32_t atomic::read(uint32_t *Addr, int Ordering) {
  return atomicReadImpl(Addr, Ordering);
}
uint32_t atomic::inc(uint32_t *Addr, uint32_t V, int Ordering) {
  return atomicIncImpl(Addr, V, Ordering);
}
uint32_t atomic::add(uint32_t *Addr, uint32_t V, int Ordering) {
  return atomicAddImpl(Addr, V, Ordering);
}
uint64_t atomic::add(uint64_t *Addr, uint64_t V, int Ordering) {
  return atomicAddImpl(Addr, V, Ordering);
}

#pragma omp end declare target
