//===---- ProfileEnvironment.h - OpenMP GPU profile environment --- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef _OMPTARGET_PROFILE_ENVIRONMENT_H_
#define _OMPTARGET_PROFILE_ENVIRONMENT_H_

// deviceRTL uses <stdint> and DeviceRTL uses explicit definitions

#ifdef OMPTARGET_DEVICE_RUNTIME
#include "Types.h"
#else
#include <SourceInfo.h>
using IdentTy = ident_t;
#endif

#ifdef OMPTARGET_DEVICE_RUNTIME
namespace _OMP {
#endif

namespace profile {

class ProfileValueTy {
  uint32_t Value;
  IdentTy *FirstLoc;

public:
#ifndef OMPTARGET_DEVICE_RUNTIME
  /// Host implementation
  ///{

  ProfileValueTy() : Value(0), FirstLoc(nullptr) {}

  uint32_t getValue() { return Value; }
  IdentTy *getFirstLoc() { return FirstLoc; }

  ///}

#else

  /// Device implementation
  ///{

  void set(IdentTy *Loc = nullptr, uint32_t NewV = 1);
  void inc(IdentTy *Loc = nullptr);

  ///}

#endif
};

struct ProfileEnvironmentTy {
#define Element(NAME) ProfileValueTy NAME;
#include "ProfileEnvironmentElements.inc"
#undef Element
};

} // namespace profile

#ifdef OMPTARGET_DEVICE_RUNTIME
} // namespace _OMP
#endif

#endif
