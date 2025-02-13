//===- Transforms/IPO/InputGen.h ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A pass to create executables that generate inputs for static code.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_INPUTGEN_H
#define LLVM_TRANSFORMS_IPO_INPUTGEN_H

#include "llvm/IR/PassManager.h"

namespace llvm {

enum class IGIMode : unsigned {
  Disabled,
  Record,
  Generate,
  ReplayGenerated,
  ReplayRecorded
};

class InputGenInstrumentEntriesPass
    : public PassInfoMixin<InputGenInstrumentEntriesPass> {
public:
  InputGenInstrumentEntriesPass(){};
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};
class InputGenInstrumentMemoryPass
    : public PassInfoMixin<InputGenInstrumentMemoryPass> {
public:
  InputGenInstrumentMemoryPass(){};
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_IPO_INPUTGEN_H
