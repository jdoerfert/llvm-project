//===--- target_impl.h - OpenMP device RTL target code impl. ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions of target specific functions needed in the generic part of the
// device RTL implementation.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_IMPL_H
#define TARGET_IMPL_H

/// Atomically increment the pointee of \p Ptr by \p Val and return the original
/// value of the pointee.
template <typename T> T __kmpc_impl_atomic_add(T *Ptr, T Val) {
  return atomicAdd(Ptr, Val);
}

/// Atomically exchange the pointee of \p Ptr with \p Val and return the
/// original value of the pointee.
template <typename T> T __kmpc_impl_atomic_exchange(T *Ptr, T Val) {
  return atomicExch(Ptr, Val);
}

#endif // TARGET_IMPL_H
