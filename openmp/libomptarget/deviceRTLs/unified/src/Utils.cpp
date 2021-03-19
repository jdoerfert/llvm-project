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
#include "Mapping.h"

using namespace _OMP;

namespace {

/// Fallback implementations are missing to trigger a link time error.
/// Implementations for new devices, including the host, should go into a
/// dedicated begin/end declare variant.
///
///{

uint64_t PackImpl(uint32_t LowBits, uint32_t HighBits);
void UnpackImpl(uint64_t Val, uint32_t *LowBits, uint32_t &HighBits);

///}


/// AMDGCN implementations of the shuffle sync idiom.
///
///{
#pragma omp begin declare variant match(device = {arch(amdgcn)})

void UnpackImpl(uint64_t Val, uint32_t *LowBits, uint32_t *HighBits) {
  *LowBits = (uint32_t)(Val & UINT64_C(0x00000000FFFFFFFF));
  *HighBits = (uint32_t)((Val & UINT64_C(0xFFFFFFFF00000000)) >> 32);
}

uint64_t PackImpl(uint32_t LowBits, uint32_t HighBits) {
  return (((uint64_t)HighBits) << 32) | (uint64_t)LowBits;
}

#pragma omp end declare variant
///}

/// NVPTX implementations of the shuffle and shuffle sync idiom.
///
///{
#pragma omp begin declare variant match(                                       \
    device = {arch(nvptx, nvptx64)}, implementation = {extension(match_any)})

void UnpackImpl(uint64_t Val, uint32_t *LowBits, uint32_t *HighBits) {
  asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(LowBits), "=r"(HighBits) : "l"(Val));
}

uint64_t PackImpl(uint32_t LowBits, uint32_t HighBits) {
  uint64_t Val;
  asm volatile("mov.b64 %0, {%1,%2};" : "=l"(Val) : "r"(LowBits), "r"(HighBits));
  return Val;
}

#pragma omp end declare variant

} // namespace


uint64_t utils::pack(uint32_t LowBits, uint32_t HighBits) {
  return PackImpl(LowBits, HighBits);
}

void utils::unpack(uint64_t Val, uint32_t &LowBits, uint32_t &HighBits) {
  UnpackImpl(Val, &LowBits, &HighBits);
}

///}


namespace {

#pragma omp declare target

/// Fallback implementations are missing to trigger a link time error.
/// Implementations for new devices, including the host, should go into a
/// dedicated begin/end declare variant.
///
///{

int32_t shuffleImpl(uint64_t Mask, int32_t Var, int32_t SrcLane);

int32_t shuffleDownImpl(uint64_t Mask, int32_t Var, uint32_t Delta,
                              int32_t Width);

///}

/// AMDGCN implementations of the shuffle sync idiom.
///
///{
#pragma omp begin declare variant match(device = {arch(amdgcn)})

int32_t shuffleImpl(uint64_t Mask, int32_t Var, int32_t SrcLane) {
  int Width = GetWarpSize();
  int Self = GetLaneId();
  int Index = SrcLane + (Self & ~(Width - 1));
  return __builtin_amdgcn_ds_bpermute(Index << 2, Var);
}

int32_t shuffleDownImpl(uint64_t Mask, int32_t Var,
                                     uint32_t LaneDelta, int32_t Width) {
  int Self = GetLaneId();
  int Index = Self + LaneDelta;
  Index = (int)(LaneDelta + (Self & (Width - 1))) >= Width ? Self : Index;
  return __builtin_amdgcn_ds_bpermute(Index << 2, Var);
}

#pragma omp end declare variant
///}

/// NVPTX implementations of the shuffle and shuffle sync idiom.
///
///{
#pragma omp begin declare variant match(                                       \
    device = {arch(nvptx, nvptx64)}, implementation = {extension(match_any)})

int32_t shuffleImpl(uint64_t Mask, int32_t Var, int32_t SrcLane) {
  return __nvvm_shfl_sync_idx_i32(Mask, Var, SrcLane, 0x1f);
}

int32_t shuffleDownImpl(uint64_t Mask, int32_t Var, uint32_t Delta,
                                     int32_t Width) {
  int32_t T = ((mapping::getWarpSize() - Width) << 8) | 0x1f;
  return __nvvm_shfl_sync_down_i32(Mask, Var, Delta, T);
}

#pragma omp end declare variant
///}

#pragma omp end declare target
} // namespace


#pragma omp declare target

int32_t __kmpc_shuffle_int32(int32_t Val, int16_t Delta, int16_t SrcLane) {
  return shuffleDownImpl(lanes::All, Val, Delta, SrcLane);
}

int64_t __kmpc_shuffle_int64(int64_t Val, int16_t Delta, int16_t Width) {
  uint32_t lo, hi;
  utils::unpack(Val, lo, hi);
  hi = shuffleDownImpl(lanes::All, hi, Delta, Width);
  lo = shuffleDownImpl(lanes::All, lo, Delta, Width);
  return utils::pack(lo, hi);
}

int32_t utils::shuffle(uint64_t Mask, int32_t Var, int32_t SrcLane) {
  return shuffleImpl(Mask, Var, SrcLane);
}

int32_t utils::shuffleDown(uint64_t Mask, int32_t Var, uint32_t Delta, int32_t Width) {
  return shuffleDownImpl(Mask, Var, Delta, Width);
}


#pragma omp end declare target


