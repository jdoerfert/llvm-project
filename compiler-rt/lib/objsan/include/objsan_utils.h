//===- objsan/include/objsan_utils.h --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ObjSan.
//
//===----------------------------------------------------------------------===//

#ifndef OBJSAN_INCLUDE_OBJSAN_UTILS_H
#define OBJSAN_INCLUDE_OBJSAN_UTILS_H

namespace __objsan {

enum class AccessKind {
  None = 0,
  Load,
  Store,
};

struct StatusTy {
  bool Failed = false;
  AccessKind Access = AccessKind::None;

  bool hasFailed() const { return Failed; }
  bool isLoad() const { return Access == AccessKind::Load; }
  bool isStore() const { return Access == AccessKind::Store; }
  void setFailure(AccessKind A) { Failed = true; Access = A; }
};

} // namespace __objsan

#endif // OBJSAN_INCLUDE_OBJSAN_UTILS_H
