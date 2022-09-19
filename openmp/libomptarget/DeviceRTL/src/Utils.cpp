//===------- Utils.cpp - OpenMP device runtime utility functions -- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "Utils.h"

#include "Debug.h"
#include "Interface.h"
#include "Mapping.h"

#pragma omp begin declare target device_type(nohost)

using namespace _OMP;

namespace _OMP {
/// Helper to keep code alive without introducing a performance penalty.
__attribute__((weak, optnone, cold)) KEEP_ALIVE void keepAlive() {
  __kmpc_get_hardware_thread_id_in_block();
  __kmpc_get_hardware_num_threads_in_block();
  __kmpc_get_warp_size();
  __kmpc_barrier_simple_spmd(nullptr, 0);
  __kmpc_barrier_simple_generic(nullptr, 0);
}
} // namespace _OMP

namespace impl {

uint64_t ballotSync(uint64_t Mask, uint32_t Predicate);

uint32_t shuffle(uint64_t Mask, uint32_t Var, int32_t SrcLane);

uint32_t shuffleDown(uint64_t Mask, uint32_t Var, uint32_t LaneDelta,
                    int32_t Width);
uint64_t shuffleDown(uint64_t Mask, uint64_t Var, uint32_t LaneDelta,
                    int32_t Width);

void Unpack(uint64_t Val, uint32_t *LowBits, uint32_t *HighBits);
uint64_t Pack(uint32_t LowBits, uint32_t HighBits);

/// AMDGCN Implementation
///
///{
#pragma omp begin declare variant match(device = {arch(amdgcn)})

void Unpack(uint64_t Val, uint32_t *LowBits, uint32_t *HighBits) {
  static_assert(sizeof(unsigned long) == 8, "");
  *LowBits = (uint32_t)(Val & 0x00000000FFFFFFFFUL);
  *HighBits = (uint32_t)((Val & 0xFFFFFFFF00000000UL) >> 32);
}

uint64_t Pack(uint32_t LowBits, uint32_t HighBits) {
  return (((uint64_t)HighBits) << 32) | (uint64_t)LowBits;
}

#pragma omp end declare variant

/// NVPTX Implementation
///
///{
#pragma omp begin declare variant match(                                       \
    device = {arch(nvptx, nvptx64)}, implementation = {extension(match_any)})

void Unpack(uint64_t Val, uint32_t *LowBits, uint32_t *HighBits) {
  uint32_t LowBitsLocal, HighBitsLocal;
  asm("mov.b64 {%0,%1}, %2;"
      : "=r"(LowBitsLocal), "=r"(HighBitsLocal)
      : "l"(Val));
  *LowBits = LowBitsLocal;
  *HighBits = HighBitsLocal;
}

uint64_t Pack(uint32_t LowBits, uint32_t HighBits) {
  uint64_t Val;
  asm("mov.b64 %0, {%1,%2};" : "=l"(Val) : "r"(LowBits), "r"(HighBits));
  return Val;
}

#pragma omp end declare variant

/// AMDGCN Implementation
///
///{
#pragma omp begin declare variant match(device = {arch(amdgcn)})

uint64_t ballotSync(uint64_t Mask, uint32_t Predicate){
  __builtin_trap();
}

uint32_t shuffle(uint64_t Mask, uint32_t Var, int32_t SrcLane) {
  int Width = mapping::getWarpSize();
  int Self = mapping::getThreadIdInWarp();
  int Index = SrcLane + (Self & ~(Width - 1));
  return __builtin_amdgcn_ds_bpermute(Index << 2, Var);
}

uint32_t shuffleDown(uint64_t Mask, uint32_t Var, uint32_t LaneDelta,
                    int32_t Width) {
  int Self = mapping::getThreadIdInWarp();
  int Index = Self + LaneDelta;
  Index = (int)(LaneDelta + (Self & (Width - 1))) >= Width ? Self : Index;
  return __builtin_amdgcn_ds_bpermute(Index << 2, Var);
}

uint64_t shuffleDown(uint64_t Mask, uint64_t Var, uint32_t LaneDelta,
                    int32_t Width) {
  int Self = mapping::getThreadIdInWarp();
  int Index = Self + LaneDelta;
  Index = (int)(LaneDelta + (Self & (Width - 1))) >= Width ? Self : Index;
  return __builtin_amdgcn_ds_bpermute(Index << 2, Var);
}

#pragma omp end declare variant
///}

/// NVPTX Implementation
///
///{
#pragma omp begin declare variant match(                                       \
    device = {arch(nvptx, nvptx64)}, implementation = {extension(match_any)})

uint64_t ballotSync(uint64_t Mask, uint32_t Predicate){
  return __nvvm_vote_ballot_sync(Mask, Predicate);
}

uint32_t shuffle(uint64_t Mask, uint32_t Var, int32_t SrcLane) {
  return __nvvm_shfl_sync_idx_i32(Mask, Var, SrcLane, 0x1f);
}

uint32_t shuffleDown(uint64_t Mask, uint32_t Var, uint32_t Delta, int32_t Width) {
  int32_t T = ((mapping::getWarpSize() - Width) << 8) | 0x1f;
  return __nvvm_shfl_sync_down_i32(Mask, Var, Delta, T);
}

#pragma omp end declare variant

uint64_t shuffleDown(uint64_t Mask, uint64_t Var, uint32_t LaneDelta,
                    int32_t Width) {
  uint32_t lo, hi;
  utils::unpack(Var, lo, hi);
  hi = shuffleDown(Mask, hi, LaneDelta, Width);
  lo = shuffleDown(Mask, lo, LaneDelta, Width);
  return utils::pack(lo, hi);
}

} // namespace impl

uint64_t utils::pack(uint32_t LowBits, uint32_t HighBits) {
  return impl::Pack(LowBits, HighBits);
}

void utils::unpack(uint64_t Val, uint32_t &LowBits, uint32_t &HighBits) {
  impl::Unpack(Val, &LowBits, &HighBits);
}

uint64_t utils::ballotSync(uint64_t Mask, uint32_t Predicate) {
  return impl::ballotSync(Mask, Predicate);
}

uint32_t utils::shuffle(uint64_t Mask, uint32_t Var, int32_t SrcLane) {
  return impl::shuffle(Mask, Var, SrcLane);
}

uint32_t utils::shuffleDown(uint64_t Mask, uint32_t Var, uint32_t Delta,
                           int32_t Width) {
  return impl::shuffleDown(Mask, Var, Delta, Width);
}

int32_t utils::shuffleDown(uint64_t Mask, int32_t Var, uint32_t Delta,
                           int32_t Width) {
  return static_cast<int32_t> (impl::shuffleDown(Mask, static_cast<uint32_t>(Var), Delta, Width));
}


uint64_t utils::shuffleDown(uint64_t Mask, uint64_t Var, uint32_t Delta,
                           int32_t Width) {
  return impl::shuffleDown(Mask, Var, Delta, Width);
}

int64_t utils::shuffleDown(uint64_t Mask, int64_t Var, uint32_t Delta,
                           int32_t Width) {
  return static_cast<int64_t>(impl::shuffleDown(Mask, static_cast<uint64_t>(Var), Delta, Width));
}

extern "C" {
uint32_t __kmpc_shuffle_int32(uint32_t Val, int16_t Delta, int16_t SrcLane) {
  FunctionTracingRAII();
  return impl::shuffleDown(lanes::All, Val, Delta, SrcLane);
}

uint64_t __kmpc_shuffle_int64(uint64_t Val, int16_t Delta, int16_t Width) {
  FunctionTracingRAII();
  return impl::shuffleDown(lanes::All, Val, Delta, Width);
}
}

#pragma omp end declare target
