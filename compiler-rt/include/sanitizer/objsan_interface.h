//===- sanitizer/objsan_interface.h ---------------------------------------===//
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

#ifndef SANITIZER_OBJSAN_INTERFACE_H
#define SANITIZER_OBJSAN_INTERFACE_H

#pragma GCC visibility push(default)

#ifdef __cplusplus
extern "C" {
#endif

/// Initialize the objsan runtime.
void __objsan_rt_init(void);

/// Finalize the objsan runtime.
void __objsan_rt_deinit(void);

#ifdef __cplusplus
} // extern C
#endif

#pragma GCC visibility pop

#endif // SANITIZER_OBJSAN_INTERFACE_H
