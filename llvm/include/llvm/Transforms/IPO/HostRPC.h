//===- Transform/IPO/HostRPC.h - Code of automatic host rpc -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_HOSTRPC_H
#define LLVM_TRANSFORMS_IPO_HOSTRPC_H

#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/IR/PassManager.h"

namespace llvm {
class HostRPCPass : public PassInfoMixin<HostRPCPass> {
public:
  HostRPCPass() : LTOPhase(ThinOrFullLTOPhase::None) {}
  HostRPCPass(ThinOrFullLTOPhase LTOPhase) : LTOPhase(LTOPhase) {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

private:
  const ThinOrFullLTOPhase LTOPhase = ThinOrFullLTOPhase::None;
};
} // namespace llvm

#endif // LLVM_TRANSFORMS_IPO_HOSTRPC_H
