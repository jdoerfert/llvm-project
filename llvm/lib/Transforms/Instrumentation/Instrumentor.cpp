//===-- Instrumentor.cpp - Highly configurable instrumentation pass -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/Instrumentor.h"

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/ConstantFolder.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <type_traits>

using namespace llvm;

#define DEBUG_TYPE "instrumentor"

cl::opt<std::string> WriteJSONConfig(
    "instrumentor-write-config-file",
    cl::desc(
        "Write the instrumentor configuration into the specified JSON file"),
    cl::init(""));
cl::opt<std::string> ReadJSONConfig(
    "instrumentor-read-config-file",
    cl::desc(
        "Read the instrumentor configuration from the specified JSON file"),
    cl::init(""));

namespace {

template <typename... Targs>
void dumpObject(json::OStream &J, Targs... Fargs) {}

void writeInstrumentorConfig(InstrumentorConfig &IC) {
  if (WriteJSONConfig.empty())
    return;

  std::error_code EC;
  raw_fd_stream OS(WriteJSONConfig, EC);
  if (EC) {
    errs() << "WARNING: Failed to open instrumentor configuration file for "
              "writing: "
           << EC.message() << "\n";
    return;
  }

  json::OStream J(OS, 2);
  J.objectBegin();

#define SECTION_START(SECTION, CLASS)                                          \
  J.attributeBegin(#SECTION);                                                  \
  J.objectBegin();
#define CONFIG_INTERNAL(SECTION, TYPE, NAME, DEFAULT_VALUE)
#define CONFIG(SECTION, TYPE, NAME, DEFAULT_VALUE)                             \
  J.attribute(#NAME, IC.SECTION.NAME);
#define SECTION_END(SECTION)                                                   \
  J.objectEnd();                                                               \
  J.attributeEnd();

#include "llvm/Transforms/Instrumentation/InstrumentorConfig.def"

  J.objectEnd();
}

bool readInstrumentorConfigFromJSON(InstrumentorConfig &IC) {
  if (ReadJSONConfig.empty())
    return true;

  std::error_code EC;
  auto BufferOrErr = MemoryBuffer::getFileOrSTDIN(ReadJSONConfig);
  if (std::error_code EC = BufferOrErr.getError()) {
    errs() << "WARNING: Failed to open instrumentor configuration file for "
              "reading: "
           << EC.message() << "\n";
    return false;
  }
  auto Buffer = std::move(BufferOrErr.get());
  json::Path::Root NullRoot;
  auto Parsed = json::parse(Buffer->getBuffer());
  if (!Parsed) {
    errs() << "WARNING: Failed to parse the instrumentor configuration file: "
           << Parsed.takeError() << "\n";
    return false;
  }
  auto *Config = Parsed->getAsObject();
  if (!Config) {
    errs() << "WARNING: Failed to parse the instrumentor configuration file: "
              "Expected "
              "an object '{ ... }'\n";
    return false;
  }

  auto End = Config->end(), It = Config->begin();

#define CONFIG(SECTION, TYPE, NAME, DEFAULT_VALUE)                             \
  It = Config->find(#SECTION);                                                 \
  if (It != End) {                                                             \
    if (auto *InstObj = It->second.getAsObject()) {                            \
      if (auto *Val = InstObj->get(#NAME)) {                                   \
        if (!json::fromJSON(*Val, IC.SECTION.NAME, NullRoot))                  \
          errs() << "WARNING: Failed to read " #SECTION "." #NAME " as " #TYPE \
                 << "\n";                                                      \
      }                                                                        \
    }                                                                          \
  }

#define SECTION_START(SECTION, CLASS)
#define CONFIG_INTERNAL(SECTION, TYPE, NAME, DEFAULT_VALUE)
#define SECTION_END(SECTION)

#include "llvm/Transforms/Instrumentation/InstrumentorConfig.def"

  return true;
}

raw_ostream &printAsCType(raw_ostream &OS, Type *T) {
  if (T->isPointerTy())
    return OS << "void* ";
  if (T->isIntegerTy())
    return OS << "int" << T->getIntegerBitWidth() << "_t ";
  return OS << *T << " ";
}

template <typename IRBTy>
Value *tryToCast(IRBTy &IRB, Value *V, Type *Ty, const DataLayout &DL) {
  auto *VTy = V->getType();
  if (VTy == Ty)
    return V;
  if (CastInst::isBitOrNoopPointerCastable(VTy, Ty, DL))
    return IRB.CreateBitOrPointerCast(V, Ty);
  if (VTy->isPointerTy() && Ty->isPointerTy())
    return IRB.CreatePointerBitCastOrAddrSpaceCast(V, Ty);
  if (VTy->isPointerTy() && Ty->isIntegerTy())
    return IRB.CreateIntToPtr(V, Ty);
  if (VTy->isIntegerTy() && Ty->isIntegerTy())
    return IRB.CreateIntCast(V, Ty, /* isSigned */ false);
  if (VTy->isFloatingPointTy() && Ty->isIntOrPtrTy()) {
    switch (DL.getTypeSizeInBits(VTy)) {
    case 64:
      V = IRB.CreateBitCast(V, IRB.getInt64Ty());
      break;
    case 32:
      V = IRB.CreateBitCast(V, IRB.getInt32Ty());
      break;
    case 16:
      V = IRB.CreateBitCast(V, IRB.getInt16Ty());
      break;
    case 8:
      V = IRB.CreateBitCast(V, IRB.getInt8Ty());
      break;
    default:
      return Constant::getAllOnesValue(Ty);
    }
    return tryToCast(IRB, V, Ty, DL);
  }
  return Constant::getAllOnesValue(Ty);
}

class InstrumentorImpl final {
public:
  InstrumentorImpl(const InstrumentorConfig &IC, Module &M)
      : IC(IC), M(M), Ctx(M.getContext()),
        IRB(Ctx, ConstantFolder(),
            IRBuilderCallbackInserter(
                [&](Instruction *I) { NewInsts[I] = Epoche; })) {}

  /// Instrument the module, public entry point.
  bool instrument();

private:
  bool shouldInstrumentFunction(Function *Fn);
  bool instrumentFunction(Function &Fn);
  bool instrument(AllocaInst &I);
  bool instrument(StoreInst &I);

  /// Mapping to remember temporary allocas for reuse.
  DenseMap<std::pair<Function *, unsigned>, AllocaInst *> AllocaMap;

  /// Return a temporary alloca to communicate (large) values with the runtime.
  AllocaInst *getAlloca(Function *Fn, Type *Ty) {
    AllocaInst *&AI = AllocaMap[{Fn, DL.getTypeAllocSize(Ty)}];
    if (!AI)
      AI = new AllocaInst(Ty, DL.getAllocaAddrSpace(), "",
                          Fn->getEntryBlock().getFirstInsertionPt());
    return AI;
  }

  template <typename Ty> Constant *getCI(Type *IT, Ty Val) {
    return ConstantInt::get(IT, Val);
  }

  std::string getRTName(StringRef Prefix, StringRef Position,
                        StringRef Suffix = "") {
    return (IC.Base.RuntimeName + Prefix + Position + Suffix).str();
  }

  /// Map to remember instrumentation functions for a specific opcode and
  /// pre/post position.
  DenseMap<int, FunctionCallee> InstrumentationFunctions;

  FunctionCallee getCallee(Instruction &I, SmallVectorImpl<Type *> &RTArgTypes,
                           SmallVectorImpl<std::string> &RTArgNames, bool After,
                           bool Indirection, Type *RT = nullptr) {
    FunctionCallee &FC =
        InstrumentationFunctions[4 * I.getOpcode() + 2 * After + Indirection];
    if (!FC.getFunctionType()) {
      FC = M.getOrInsertFunction(
          getRTName(After ? "post_" : "pre_", I.getOpcodeName(),
                    Indirection ? "_ind" : ""),
          FunctionType::get(RT ? RT : VoidTy, RTArgTypes, /*IsVarArgs*/ false));

      if (IC.Base.PrintRuntimeSignatures) {
        printAsCType(outs(), FC.getFunctionType()->getReturnType());
        outs() << FC.getCallee()->getName() << "(";
        auto *FT = FC.getFunctionType();
        for (int I = 0, E = RTArgNames.size(); I != E; ++I) {
          if (I != 0)
            outs() << ", ";
          printAsCType(outs(), FT->getParamType(I)) << RTArgNames[I];
        }
        outs() << ");\n";
      }
    }
    return FC;
  }

  /// Each instrumentation, i.a., of an instruction, is happening in a dedicated
  /// epoche. The epoche allows to determine if instrumentation instructions
  /// were already around, due to prior instrumentations, or have been
  /// introduced to support the current instrumentation, i.a., compute
  /// information about the current instruction.
  unsigned Epoche = 0;

  /// A mapping from instrumentation instructions to the epoche they have been
  /// created.
  DenseMap<Instruction *, unsigned> NewInsts;

  /// The instrumentor configuration.
  const InstrumentorConfig &IC;

  /// The module and the LLVM context.
  Module &M;
  LLVMContext &Ctx;

  /// A special IR builder that keeps track of the inserted instructions.
  IRBuilder<ConstantFolder, IRBuilderCallbackInserter> IRB;

  /// Commonly used values for IR inspection and creation.
  ///{

  const DataLayout &DL = M.getDataLayout();

  Type *VoidTy = Type::getVoidTy(Ctx);
  Type *IntptrTy = M.getDataLayout().getIntPtrType(Ctx);
  PointerType *PtrTy = PointerType::getUnqual(Ctx);
  IntegerType *Int8Ty = Type::getInt8Ty(Ctx);
  IntegerType *Int32Ty = Type::getInt32Ty(Ctx);
  IntegerType *Int64Ty = Type::getInt64Ty(Ctx);
  ///}
};

} // end anonymous namespace

bool InstrumentorImpl::shouldInstrumentFunction(Function *Fn) {
  if (!Fn || Fn->isDeclaration())
    return false;
  return true;
}

bool InstrumentorImpl::instrument(AllocaInst &I) {
  if (IC.Alloca.CB && !IC.Alloca.CB(I))
    return false;

  Instruction *IP = I.getNextNonDebugInstruction();
  while (isa<AllocaInst>(IP))
    IP = IP->getNextNonDebugInstruction();
  IRB.SetInsertPoint(IP);

  SmallVector<Type *> RTArgTypes;
  SmallVector<Value *> RTArgs;
  SmallVector<std::string> RTArgNames;

  if (IC.Alloca.Value) {
    auto *ArgTy = PtrTy;
    RTArgTypes.push_back(ArgTy);
    RTArgs.push_back(tryToCast(IRB, &I, ArgTy, DL));
    RTArgNames.push_back("Value");
  }

  if (IC.Alloca.AllocationSize) {
    auto *ArgTy = Int64Ty;
    RTArgTypes.push_back(ArgTy);
    Value *SizeValue = nullptr;
    TypeSize TypeSize = DL.getTypeAllocSize(I.getAllocatedType());
    if (TypeSize.isFixed())
      SizeValue = getCI(ArgTy, TypeSize.getFixedValue());
    if (!SizeValue) {
      SizeValue = IRB.CreateSub(
          IRB.CreatePtrToInt(
              IRB.CreateGEP(I.getAllocatedType(), &I, {getCI(Int32Ty, 1)}),
              ArgTy),
          IRB.CreatePtrToInt(&I, ArgTy));
    }
    if (I.isArrayAllocation())
      SizeValue = IRB.CreateMul(
          SizeValue, IRB.CreateZExtOrBitCast(I.getArraySize(), ArgTy));
    RTArgs.push_back(SizeValue);
    RTArgNames.push_back("AllocationSize");
  }

  if (IC.Alloca.Alignment) {
    auto *ArgTy = Int64Ty;
    RTArgTypes.push_back(ArgTy);
    RTArgs.push_back(getCI(ArgTy, I.getAlign().value()));
    RTArgNames.push_back("Alignment");
  }

  Type *RetTy = IC.Alloca.ReplaceValue ? PtrTy : nullptr;
  FunctionCallee FC = getCallee(I, RTArgTypes, RTArgNames, /*After=*/true,
                                /*Indirection=*/false, RetTy);
  auto *CI = IRB.CreateCall(FC, RTArgs);
  if (IC.Alloca.ReplaceValue) {
    IRB.SetInsertPoint(CI->getNextNonDebugInstruction());
    I.replaceUsesWithIf(tryToCast(IRB, CI, I.getType(), DL), [&](Use &U) {
      return NewInsts.lookup(cast<Instruction>(U.getUser())) != Epoche;
    });
  }

  return true;
}

bool InstrumentorImpl::instrument(StoreInst &I) {
  if (IC.Store.CB && !IC.Store.CB(I))
    return false;

  IRB.SetInsertPoint(&I);

  AllocaInst *IndirectionAI = nullptr;
  SmallVector<Type *> RTArgTypes;
  SmallVector<Value *> RTArgs;
  SmallVector<std::string> RTArgNames;

  if (IC.Store.PointerOperand) {
    auto *ArgTy = PtrTy;
    RTArgTypes.push_back(ArgTy);
    RTArgs.push_back(
        IRB.CreatePointerBitCastOrAddrSpaceCast(I.getPointerOperand(), ArgTy));
    RTArgNames.push_back("PointerOperand");
  }

  if (IC.Store.PointerOperandAddressSpace) {
    auto *ArgTy = Int32Ty;
    RTArgTypes.push_back(ArgTy);
    RTArgs.push_back(getCI(ArgTy, I.getPointerAddressSpace()));
    RTArgNames.push_back("PointerOperandAddressSpace");
  }

  if (IC.Store.ValueOperand) {
    Type *ArgTy = Int64Ty;
    if (DL.getTypeSizeInBits(I.getValueOperand()->getType()) > 64) {
      IndirectionAI =
          getAlloca(I.getFunction(), I.getValueOperand()->getType());
      IRB.CreateStore(I.getValueOperand(), IndirectionAI);
      ArgTy = PtrTy;
    }

    RTArgTypes.push_back(ArgTy);
    if (IndirectionAI)
      RTArgs.push_back(IndirectionAI);
    else
      RTArgs.push_back(tryToCast(IRB, I.getValueOperand(), ArgTy, DL));
    RTArgNames.push_back(IndirectionAI ? "ValueOperandStorage"
                                       : "ValueOperand");
  }

  if (IC.Store.ValueOperandSize) {
    auto *ArgTy = Int64Ty;
    RTArgTypes.push_back(ArgTy);
    RTArgs.push_back(
        getCI(ArgTy, DL.getTypeStoreSize(I.getValueOperand()->getType())));
    RTArgNames.push_back("ValueOperandSize");
  }

  if (IC.Store.ValueOperandTypeId) {
    auto *ArgTy = Int32Ty;
    RTArgTypes.push_back(ArgTy);
    RTArgs.push_back(getCI(ArgTy, I.getValueOperand()->getType()->getTypeID()));
    RTArgNames.push_back("ValueOperandTypeId");
  };

  if (IC.Store.Alignment) {
    auto *ArgTy = Int64Ty;
    RTArgTypes.push_back(ArgTy);
    RTArgs.push_back(getCI(ArgTy, I.getAlign().value()));
    RTArgNames.push_back("Alignment");
  }

  if (IC.Store.AtomicityOrdering) {
    auto *ArgTy = Int32Ty;
    RTArgTypes.push_back(ArgTy);
    RTArgs.push_back(getCI(ArgTy, uint64_t(I.getOrdering())));
    RTArgNames.push_back("AtomicityOrdering");
  }

  if (IC.Store.SyncScopeId) {
    auto *ArgTy = Int8Ty;
    RTArgTypes.push_back(ArgTy);
    RTArgs.push_back(getCI(ArgTy, uint64_t(I.getSyncScopeID())));
    RTArgNames.push_back("SyncScopeId");
  }

  if (IC.Store.IsVolatile) {
    auto *ArgTy = Int8Ty;
    RTArgTypes.push_back(ArgTy);
    RTArgs.push_back(getCI(ArgTy, I.isVolatile()));
    RTArgNames.push_back("IsVolatile");
  }

  FunctionCallee FC =
      getCallee(I, RTArgTypes, RTArgNames, /*After=*/false, IndirectionAI);
  IRB.CreateCall(FC, RTArgs);

  return true;
}

bool InstrumentorImpl::instrumentFunction(Function &Fn) {
  bool Changed = false;
  if (!shouldInstrumentFunction(&Fn))
    return Changed;

  ReversePostOrderTraversal<Function *> RPOT(&Fn);
  for (auto &It : RPOT) {
    for (auto &I : *It) {
      // Skip instrumentation instructions.
      if (NewInsts.contains(&I))
        continue;

      // Count epochs eagerly.
      ++Epoche;

      switch (I.getOpcode()) {
      case Instruction::Alloca:
        if (IC.Alloca.Instrument)
          instrument(cast<AllocaInst>(I));
        break;
      case Instruction::Store:
        if (IC.Store.Instrument)
          instrument(cast<StoreInst>(I));
        break;
      default:
        break;
      }
    }
  }

  return Changed;
}

bool InstrumentorImpl::instrument() {
  bool Changed = false;

  for (Function &Fn : M)
    Changed |= instrumentFunction(Fn);

  return Changed;
}

PreservedAnalyses InstrumentorPass::run(Module &M, ModuleAnalysisManager &MAM) {
  InstrumentorImpl Impl(IC, M);
  if (!readInstrumentorConfigFromJSON(IC))
    return PreservedAnalyses::all();
  writeInstrumentorConfig(IC);
  if (!Impl.instrument())
    return PreservedAnalyses::all();
  assert(!verifyModule(M, &errs()));
  return PreservedAnalyses::none();
}
