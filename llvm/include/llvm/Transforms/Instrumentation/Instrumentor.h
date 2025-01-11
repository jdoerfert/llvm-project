//===- Transforms/Instrumentation/Instrumentor.h --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A highly configurable instrumentation pass.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_INSTRUMENTOR_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_INSTRUMENTOR_H

#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalObject.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/PassManager.h"

#include <functional>
#include <string>

namespace llvm {

enum InstrumentorKindTy {
  PLAIN,
  STRING,
  BOOLEAN,
  INT8,
  INT32,
  INT64,
  INT32_PTR,
  PTR_PTR,
  TYPE_ID,
  INITIALIZER_KIND,
  PRE_ONLY = 1 << 19,
  POST_ONLY = 1 << 20,
  POTENTIALLY_INDIRECT = 1 << 21,
  REPLACABLE_PRE = 1 << 28,
  REPLACABLE_POST = 1 << 29,
  LLVM_MARK_AS_BITMASK_ENUM(/* LargestValue = */ REPLACABLE_POST)
};

/// An optional callback that takes the instruction that is about to be
/// instrumented and can return false if it should be skipped.
using CallbackTy = std::function<bool(Instruction &)>;

/// An optional callback that takes the global object that is about to be
/// instrumented and can return false if it should be skipped.
using GlobalCallbackTy = std::function<bool(GlobalObject &)>;

/// Configuration for the Instrumentor. First generic configuration, followed by
/// the selection of what instruction classes and instructions should be
/// instrumented and how.
struct InstrumentorConfig {
  static Type *getType(LLVMContext &Ctx, StringRef S) {
    auto *Ty = StringSwitch<Type *>(S)
                   .Case("int8_t", Type::getInt8Ty(Ctx))
                   .Case("int16_t", Type::getInt16Ty(Ctx))
                   .Case("int32_t", Type::getInt32Ty(Ctx))
                   .Case("int64_t", Type::getInt64Ty(Ctx))
                   .Case("void*", PointerType::getUnqual(Ctx))
                   .Default(nullptr);
    assert(Ty && "Unsupported type string!");
    return Ty;
  }

  enum Position {
    NONE,
    PRE,
    POST,
    PRE_AND_POST = PRE | POST,
  };

  struct ConfigSection {
    ConfigSection(const char *SectionName, Position SP)
        : SectionName(SectionName), SP(SP) {}
    bool EnabledPre = true;
    bool EnabledPost = true;
    std::string SectionName;
    Position SP;
    bool canRunPre() const { return SP & PRE; }
    bool canRunPost() const { return SP & POST; }
    bool isEnabled(Position P) const {
      return (P & SP) &&
             ((EnabledPre && (P & PRE)) || (EnabledPost && (P & POST)));
    }
  };

  struct ConfigValue {
    ConfigValue(bool Enabled, const std::string &ValName,
                const std::string &ValTypeStr, InstrumentorKindTy ValKind)
        : Enabled(Enabled), ValName(ValName), ValTypeStr(ValTypeStr),
          ValKind(ValKind) {};

    StringRef getName() const { return ValName; };
    InstrumentorKindTy getKind() const { return ValKind; };

    Type *getType(LLVMContext &Ctx) {
      if (!ValType)
        ValType = InstrumentorConfig::getType(Ctx, ValTypeStr);
      return ValType;
    }

    bool Enabled;
    operator bool() { return Enabled; }

  private:
    std::string ValName;
    std::string ValTypeStr;
    Type *ValType = nullptr;
    InstrumentorKindTy ValKind;
  };

#define SECTION_START(SECTION, POSITION)                                       \
  struct SECTION##Obj : public ConfigSection {                                 \
    SECTION##Obj() : ConfigSection(#SECTION, POSITION) {}

#define CVALUE_INTERNAL(SECTION, TYPE, NAME, DEFAULT_VALUE)                    \
  TYPE NAME = DEFAULT_VALUE;

#define CVALUE(SECTION, TYPE, NAME, DEFAULT_VALUE) TYPE NAME = DEFAULT_VALUE;

#define RTVALUE(SECTION, NAME, ENABLED, VALUE_TYPE_STR, PROPERTIES)            \
  struct NAME##Obj : public InstrumentorConfig::ConfigValue {                  \
    NAME##Obj()                                                                \
        : InstrumentorConfig::ConfigValue(ENABLED, #NAME, VALUE_TYPE_STR,      \
                                          PROPERTIES){};                       \
  } NAME;

#define SECTION_END(SECTION)                                                   \
  }                                                                            \
  SECTION;

#include "llvm/Transforms/Instrumentation/InstrumentorConfig.def"
};

class InstrumentorPass : public PassInfoMixin<InstrumentorPass> {
  InstrumentorConfig IC;

public:
  InstrumentorPass(InstrumentorConfig IC = InstrumentorConfig{}) : IC(IC) {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};
} // end namespace llvm

#endif // LLVM_TRANSFORMS_INSTRUMENTATION_INSTRUMENTOR_H
