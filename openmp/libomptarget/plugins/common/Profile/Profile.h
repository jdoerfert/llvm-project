//===----------- Profile.h - Target independent profile support -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Target independent profile support.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OPENMP_LIBOMPTARGET_PLUGINS_COMMON_PROFILE_MEMORYMANAGER_H
#define LLVM_OPENMP_LIBOMPTARGET_PLUGINS_COMMON_PROFILE_MEMORYMANAGER_H

#include "omptarget.h"

#include "KernelEnvironment.h"
#include "ProfileEnvironment.h"

#include <functional>
#include <map>
#include <vector>

class ProfilerTy {
  using CopyBackCallbackTy = std::function<bool(int, void *, const void *,
                                                int64_t, __tgt_async_info *)>;
  CopyBackCallbackTy CopyBackCB;

  using SynchronizeCallbackTy = std::function<bool(int, __tgt_async_info *)>;
  SynchronizeCallbackTy SynchronizeCallback;

  std::map<int, __tgt_async_info> AsyncInfoMap;
  std::vector<void *> MallocedMemory;
  std::vector<kernel::KernelEnvironmentTy> KernelEnvironments;

  void copyBackIdent(int DeviceId, IdentTy *Ident,
                     __tgt_async_info &AsyncInfo) {
    if (!Ident)
      return;
    if (!Ident->psource || !Ident->SourceLocationSize) {
      Ident->psource = nullptr;
      Ident->SourceLocationSize = 0;
      return;
    }

    void *Tgt = malloc(sizeof(char) * Ident->SourceLocationSize);
    MallocedMemory.push_back(Tgt);
    CopyBackCB(DeviceId, Tgt, Ident->psource, Ident->SourceLocationSize,
               &AsyncInfo);
    Ident->psource = static_cast<const char *>(Tgt);
  }

public:
  ProfilerTy(const CopyBackCallbackTy &&CopyBackCB,
             const SynchronizeCallbackTy &&SynchronizeCallback)
      : CopyBackCB(CopyBackCB), SynchronizeCallback(SynchronizeCallback) {}

  ~ProfilerTy() {
    for (auto &It : AsyncInfoMap)
      SynchronizeCallback(It.first, &It.second);

    printf("Kernel profiles:\n");
    for (kernel::KernelEnvironmentTy &KernelEnvironment : KernelEnvironments) {
      SourceInfo KernelSI(&KernelEnvironment.Ident);
      printf("\t%s:%i:%i - %s\n", KernelSI.getFilename(), KernelSI.getLine(),
             KernelSI.getColumn(), KernelSI.getName());
#define Element(NAME)                                                          \
  {                                                                            \
    IdentTy *ElementIdent =                                                    \
        KernelEnvironment.ProfileEnvironment.NAME.getFirstLoc();               \
    SourceInfo ElementSI(ElementIdent);                                        \
    printf("\t%s:%i:%i - %s: %d\n", ElementSI.getFilename(),                   \
           ElementSI.getLine(), ElementSI.getColumn(), #NAME,                  \
           KernelEnvironment.ProfileEnvironment.NAME.getValue());              \
  }
#include "ProfileEnvironmentElements.inc"
#undef Element
    }

    for (void *It : MallocedMemory)
      free(It);
  }

  void registerKernelProfile(int DeviceId,
                             const kernel::KernelEnvironmentTy &KE) {
    // Copy the kernel environment as it needs to be owned by the Profiler now.
    KernelEnvironments.push_back(KE);

    __tgt_async_info &AsyncInfo = AsyncInfoMap[DeviceId];
    copyBackIdent(DeviceId, &KernelEnvironments.back().Ident, AsyncInfo);

#define Element(NAME)                                                          \
  copyBackIdent(                                                               \
      DeviceId,                                                                \
      KernelEnvironments.back().ProfileEnvironment.NAME.getFirstLoc(),         \
      AsyncInfo);
#include "ProfileEnvironmentElements.inc"
#undef Element
  }
};

#endif // LLVM_OPENMP_LIBOMPTARGET_PLUGINS_COMMON_PROFILE_MEMORYMANAGER_H
