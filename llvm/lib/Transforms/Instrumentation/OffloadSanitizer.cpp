//===-- OffloadSanitizer.cpp - Offload sanitizer --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/OffloadSanitizer.h"

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Frontend/OpenMP/OMP.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include <llvm/IR/GlobalVariable.h>
#include <string>

using namespace llvm;

#define DEBUG_TYPE "offload-sanitizer"

namespace llvm {

struct LocationInfoTy {
  uint64_t LineNo = 0;
  uint64_t ColumnNo = 0;
  uint64_t ParentIdx = -1;
  StringRef FileName;
  StringRef FunctionName;
  bool operator==(const LocationInfoTy &RHS) const {
    return LineNo == RHS.LineNo && ColumnNo == RHS.ColumnNo &&
           FileName == RHS.FileName && FunctionName == RHS.FunctionName;
  }
};
template <> struct DenseMapInfo<LocationInfoTy *> {
  static LocationInfoTy EmptyKey;
  static LocationInfoTy TombstoneKey;
  static inline LocationInfoTy *getEmptyKey() { return &EmptyKey; }

  static inline LocationInfoTy *getTombstoneKey() { return &TombstoneKey; }

  static unsigned getHashValue(const LocationInfoTy *LI) {
    unsigned Hash = DenseMapInfo<uint64_t>::getHashValue(LI->LineNo);
    Hash = detail::combineHashValue(
        Hash, DenseMapInfo<uint64_t>::getHashValue(LI->ColumnNo));
    Hash = detail::combineHashValue(
        Hash, DenseMapInfo<StringRef>::getHashValue(LI->FileName));
    Hash = detail::combineHashValue(
        Hash, DenseMapInfo<StringRef>::getHashValue(LI->FunctionName));
    return Hash;
  }

  static bool isEqual(const LocationInfoTy *LHS, const LocationInfoTy *RHS) {
    return *LHS == *RHS;
  }
};
LocationInfoTy DenseMapInfo<LocationInfoTy *>::EmptyKey =
    LocationInfoTy{(uint64_t)-1};
LocationInfoTy DenseMapInfo<LocationInfoTy *>::TombstoneKey =
    LocationInfoTy{(uint64_t)-2};
} // namespace llvm

namespace {

constexpr uint32_t SHARED_ADDRSPACE = 3;
bool isSharedGlobal(const GlobalValue &G) {
  return G.getAddressSpace() == SHARED_ADDRSPACE;
}

class OffloadSanitizerImpl final {
public:
  OffloadSanitizerImpl(Module &M, FunctionAnalysisManager &FAM)
      : M(M), FAM(FAM), Ctx(M.getContext()) {
    if (auto *Fn = M.getFunction("__offload_san_get_as0_info"))
      InfoTy = Fn->getReturnType();
  }

  bool instrument();

private:
  /// We support address space 0 to 5 right now.
  static constexpr int NumSupportedAddressSpaces = 6;

  bool isASType(Type &T) {
    return T.isPointerTy() && T.getPointerAddressSpace();
  };
  Type *getWithoutAS(Type &T) {
    return isASType(T) ? PointerType::get(T.getContext(), 0) : &T;
  };

  bool shouldInstrumentFunction(Function *Fn);

  struct AccessInfoTy {
    Instruction *I;
    unsigned PtrOpIdx;
    unsigned AS;
  };

  struct PtrInfoTy {
    Value *Start;
    Value *Length;
    Value *AS;
  };
  DenseMap<Value *, PtrInfoTy> AllocationInfoMap;
  StringMap<Value *> GlobalStringMap;

  DenseMap<LocationInfoTy *, uint64_t, DenseMapInfo<LocationInfoTy *>>
      LocationMap;

  const std::pair<LocationInfoTy *, uint64_t>
  addLocationInfo(LocationInfoTy *LI, bool &IsNew) {
    auto It = LocationMap.insert({LI, LocationMap.size()});
    IsNew = It.second;
    if (!IsNew)
      delete LI;
    return {It.first->first, It.first->second};
  }

  uint64_t addString(StringRef S) {
    const auto &It = UniqueStrings.insert({S, ConcatenatedString.size()});
    if (It.second) {
      ConcatenatedString += S;
      ConcatenatedString.push_back('\0');
    }
    return It.first->second;
  };

  void encodeLocationInfo(LocationInfoTy &LI, uint64_t Idx) {
    StringRef FunctionName = LI.FunctionName;
    auto PN = omp::prettifyFunctionName(FunctionName);
    FunctionName = SS.save(PN);

    auto FuncIdx = addString(FunctionName);
    auto FileIdx = addString(LI.FileName);
    if (LocationEncoding.size() < (Idx + 1) * 5)
      LocationEncoding.resize((Idx + 1) * 5);
    LocationEncoding[Idx * 5 + 0] = ConstantInt::get(Int64Ty, FuncIdx);
    LocationEncoding[Idx * 5 + 1] = ConstantInt::get(Int64Ty, FileIdx);
    LocationEncoding[Idx * 5 + 2] = ConstantInt::get(Int64Ty, LI.LineNo);
    LocationEncoding[Idx * 5 + 3] = ConstantInt::get(Int64Ty, LI.ColumnNo);
    LocationEncoding[Idx * 5 + 4] = ConstantInt::get(Int64Ty, LI.ParentIdx);
  }

  ConstantInt *getSourceIndex(Instruction &I,
                              LocationInfoTy *LastLI = nullptr) {
    LocationInfoTy *LI = new LocationInfoTy();
    auto *DILoc = I.getDebugLoc().get();

    auto FillLI = [&](LocationInfoTy &LI, DILocation &DIL) {
      LI.FileName = DIL.getFilename();
      if (LI.FileName.empty())
        LI.FileName = I.getFunction()->getSubprogram()->getFilename();
      LI.FunctionName = DIL.getSubprogramLinkageName();
      if (LI.FunctionName.empty())
        LI.FunctionName = I.getFunction()->getName();
      LI.LineNo = DIL.getLine();
      LI.ColumnNo = DIL.getColumn();
    };

    DILocation *ParentDILoc = nullptr;
    if (DILoc) {
      FillLI(*LI, *DILoc);
      ParentDILoc = DILoc->getInlinedAt();
    } else {
      LI->FunctionName = I.getFunction()->getName();
    }

    bool IsNew;
    uint64_t Idx;
    std::tie(LI, Idx) = addLocationInfo(LI, IsNew);
    if (LastLI)
      LastLI->ParentIdx = Idx;
    if (!IsNew)
      return ConstantInt::get(Int64Ty, Idx);

    uint64_t CurIdx = Idx;
    LocationInfoTy *CurLI = LI;
    while (ParentDILoc) {
      auto *ParentLI = new LocationInfoTy();
      FillLI(*ParentLI, *ParentDILoc);
      uint64_t ParentIdx;
      std::tie(ParentLI, ParentIdx) = addLocationInfo(ParentLI, IsNew);
      CurLI->ParentIdx = ParentIdx;
      if (!IsNew)
        break;
      encodeLocationInfo(*CurLI, CurIdx);
      CurLI = ParentLI;
      CurIdx = ParentIdx;
      ParentDILoc = ParentDILoc->getInlinedAt();
    }

    Function &Fn = *I.getFunction();
    buildCallTreeInfo(Fn, *CurLI);

    encodeLocationInfo(*CurLI, CurIdx);

    return ConstantInt::get(Int64Ty, Idx);
  }

  ConstantInt *getSourceIndex(const GlobalVariable *G) {
    SmallVector<DIGlobalVariableExpression *, 1> GlobalLocations;
    G->getDebugInfo(GlobalLocations);

    if (GlobalLocations.empty())
      return ConstantInt::get(Int64Ty, 0); // Fallback

    const auto *DLVar = GlobalLocations.front()->getVariable();

    LocationInfoTy *LI = new LocationInfoTy();
    LI->FileName = DLVar->getFilename();
    LI->LineNo = DLVar->getLine();
    LI->FunctionName = DLVar->getName();
    LI->ColumnNo = 0;

    bool IsNew;
    uint64_t Idx;
    std::tie(LI, Idx) = addLocationInfo(LI, IsNew);

    if (IsNew)
      encodeLocationInfo(*LI, Idx);

    return ConstantInt::get(Int64Ty, Idx);
  }

  void buildCallTreeInfo(Function &Fn, LocationInfoTy &LI) {
    if (Fn.hasFnAttribute("kernel"))
      return;
    SmallVector<CallBase *> Calls;
    for (auto &U : Fn.uses()) {
      auto *CB = dyn_cast<CallBase>(U.getUser());
      if (!CB)
        continue;
      if (!CB->isCallee(&U))
        continue;
      Calls.push_back(CB);
    }
    if (Calls.size() == 1) {
      getSourceIndex(*Calls.back(), &LI);
      return;
    }
    LI.ParentIdx = -2;
    AmbiguousCalls.insert(Calls.begin(), Calls.end());
  }

  SmallVector<Constant *> LocationEncoding;
  std::string ConcatenatedString;
  DenseMap<uint64_t, uint64_t> StringIndexMap;
  DenseMap<StringRef, uint64_t> UniqueStrings;

  SmallVector<Function *> Kernels;
  GlobalVariable *LocationsArray = nullptr;
  SmallSetVector<CallBase *, 16> AmbiguousCalls;
  int AllocationId = 1;

  BumpPtrAllocator BPA;
  StringSaver SS = StringSaver(BPA);

  bool handleAmbiguousCalls();
  bool handleCallStackSupport();
  bool finalizeKernels();

  bool addCtor();
  bool addDtor();

  Function *createSanitizerInitKernel();

  Value *getFunctionName(IRBuilder<> &IRB);
  Value *getFileName(IRBuilder<> &IRB);
  Value *getLineNo(IRBuilder<> &IRB);

  PtrInfoTy getPtrInfoTy(Value &Obj, const AccessInfoTy &AI) {
    if (AI.AS > 1)
      return {
          PoisonValue::get(InfoTy->getContainedType(0)->getContainedType(0)),
          PoisonValue::get(InfoTy->getContainedType(0)->getContainedType(1)),
          PoisonValue::get(InfoTy->getContainedType(1))};
    PtrInfoTy &AllocationInfo = AllocationInfoMap[&Obj];
    if (!AllocationInfo.Start) {
      Instruction *IP;
      if (auto *PHI = dyn_cast<PHINode>(&Obj)) {
        IP = PHI->getParent()->getFirstNonPHIOrDbgOrLifetime();
      } else if (auto *I = dyn_cast<Instruction>(&Obj)) {
        IP = I->getNextNode();
      } else if (isa<Argument>(Obj)) {
        IP = &*cast<Argument>(Obj)
                   .getParent()
                   ->getEntryBlock()
                   .getFirstNonPHIOrDbgOrAlloca();
      } else {
        IP = &*AI.I->getFunction()
                   ->getEntryBlock()
                   .getFirstNonPHIOrDbgOrAlloca();
      }
      IRBuilder<> IRB(IP);
      auto *AllocationInfoStruct =
          createCall(IRB, getGetAllocationInfoFn(AI.AS),
                     {getPC(IRB), getSourceIndex(*AI.I),
                      IRB.CreateAddrSpaceCast(&Obj, PtrTy)});
      AllocationInfo = PtrInfoTy{
          IRB.CreateExtractValue(AllocationInfoStruct, {0, 0}, "obj.base"),
          IRB.CreateExtractValue(AllocationInfoStruct, {0, 1}, "obj.size"),
          IRB.CreateExtractValue(AllocationInfoStruct, {1}, "obj.as")};
    }
    return AllocationInfo;
  }

  void removeAS(Function &Fn, SmallVectorImpl<Instruction *> &ASInsts);

  bool instrumentGlobals();
  bool instrumentFunction(Function &Fn);
  void instrumentTrapInstructions(SmallVectorImpl<IntrinsicInst *> &TrapCalls);
  void instrumentCallInsts(SmallVectorImpl<CallInst *> &CallInsts);
  void instrumentLifetimeIntrinsics(
      SmallVectorImpl<LifetimeIntrinsic *> &LifetimeInsts);
  void instrumentUnreachableInstructions(
      SmallVectorImpl<UnreachableInst *> &UnreachableInsts);
  void instrumentAccesses(SmallVectorImpl<AccessInfoTy> &Accesses);
  void instrumentAllocaInstructions(SmallVectorImpl<AllocaInst *> &AllocaInsts);

  FunctionCallee getOrCreateFn(FunctionCallee &FC, StringRef Name, Type *RetTy,
                               ArrayRef<Type *> ArgTys) {
    if (!FC) {
      auto *NewAllocationFnTy = FunctionType::get(RetTy, ArgTys, false);
      FC = M.getOrInsertFunction(Name, NewAllocationFnTy);
    }
    return FC;
  }

  /// int32_t ompx_global_thread_id();
  FunctionCallee ThreadIDFn;
  FunctionCallee getThreadIdFn() {
    return getOrCreateFn(ThreadIDFn, "ompx_global_thread_id", Int32Ty, {});
  }

  /// void ompx_sync_block_acq_rel();
  FunctionCallee SyncBlockFn;
  FunctionCallee getSyncBlockFn() {
    return getOrCreateFn(SyncBlockFn, "ompx_sync_block_acq_rel", VoidTy, {});
  }

  /// void __offload_san_leak_check();
  FunctionCallee getLeakCheckFn() {
    FunctionCallee LeakCheckFn;
    return getOrCreateFn(LeakCheckFn, "__offload_san_leak_check", VoidTy, {});
  }

  /// void __offload_san_trap_info(Int64Ty);
  FunctionCallee TrapInfoFn;
  FunctionCallee getTrapInfoFn() {
    return getOrCreateFn(TrapInfoFn, "__offload_san_trap_info", VoidTy,
                         {/*PC*/ Int64Ty, /*LocationId*/ Int64Ty});
  }

  /// void __offload_san_unreachable_info(Int64Ty);
  FunctionCallee UnreachableInfoFn;
  FunctionCallee getUnreachableInfoFn() {
    return getOrCreateFn(UnreachableInfoFn, "__offload_san_unreachable_info",
                         VoidTy, {/*PC*/ Int64Ty, /*LocationId*/ Int64Ty});
  }

  /// PtrTy __offload_san_unpack(Int64Ty, PtrTy);
  FunctionCallee UnpackFns[NumSupportedAddressSpaces];
  FunctionCallee getUnpackFn(uint32_t AS) {
    assert(AS < NumSupportedAddressSpaces && "Unexpected address space!");
    return getOrCreateFn(
        UnpackFns[AS], "__offload_san_unpack_as" + std::to_string(AS),
        ASPtrTy[AS], {/*PC*/ Int64Ty, /*LocationId*/ Int64Ty, PtrTy});
  }

  /// InfoTy __offload_san_get_as<AS>_info(Int64Ty, Int64Ty, PtrTy);
  FunctionCallee GetAllocationInfoFn[NumSupportedAddressSpaces];
  FunctionCallee getGetAllocationInfoFn(unsigned AS) {
    assert(AS < NumSupportedAddressSpaces && "Unexpected address space!");
    return getOrCreateFn(GetAllocationInfoFn[AS],
                         "__offload_san_get_as" + std::to_string(AS) + "_info",
                         InfoTy, {Int64Ty, Int64Ty, PtrTy});
  }

  /// ptr(0) __offload_san_check_as0_access_with_info(/* PC */Int64Ty,
  /// 						     /*LocationId*/ Int64Ty,
  /// 					             /* FakePtr */ PtrTy,
  /// 				                     /* Size */Int32Ty,
  /// 				                     /* AS */Int32Ty,
  /// 				                     /* PI.Base */ ptr(1),
  /// 				                     /* PI.Size */ Int64Ty);
  /// ptr(AS) __offload_san_check_as<AS>_access_with_info(/* PC */Int64Ty,
  /// 					                  /* FakePtr */ PtrTy,
  /// 				                          /* Size */Int32Ty,
  /// 				                          /* AS */Int32Ty,
  /// 				                          /* PI.Base */ ptr(1),
  /// 				                          /* PI.Size */
  /// Int64Ty);
  FunctionCallee CheckAccessWithInfoFn[NumSupportedAddressSpaces];
  FunctionCallee getCheckAccessWithInfoFn(unsigned AS) {
    assert(AS < NumSupportedAddressSpaces && "Unexpected address space!");
    return getOrCreateFn(CheckAccessWithInfoFn[AS],
                         "__offload_san_check_as" + std::to_string(AS) +
                             "_access_with_info",
                         ASPtrTy[AS],
                         {/*PC*/ Int64Ty, /*LocationId*/ Int64Ty, PtrTy,
                          Int32Ty, Int32Ty, ASPtrTy[1], Int64Ty});
  }

  /// PtrTy __offload_san_register_alloca(/* PC */ Int64Ty, /*LocationId*/
  /// Int64Ty,
  /// 						/* RealPtr */ AllocaPtrTy,
  /// 						/* Size */ Int32Ty);
  FunctionCallee AllocaRegisterFn;
  FunctionCallee getAllocaRegisterFn() {
    getOrCreateFn(
        AllocaRegisterFn, "__offload_san_register_alloca", PtrTy,
        {/*PC*/ Int64Ty, /*LocationId*/ Int64Ty, AllocaPtrTy, Int32Ty});
    //    cast<Function>(AllocaRegisterFn.getCallee())
    //        ->addRetAttr(Attribute::NoAlias);
    return AllocaRegisterFn;
  }

  CallInst *createCall(IRBuilder<> &IRB, FunctionCallee Callee,
                       ArrayRef<Value *> Args = std::nullopt,
                       const Twine &Name = "") {
    RTCalls.push_back(IRB.CreateCall(Callee, Args, Name));
    return RTCalls.back();
  }
  SmallVector<CallInst *> RTCalls;

  Value *getPC(IRBuilder<> &IRB) {
    static int X = 0;
    return ConstantInt::get(Int64Ty, X++);
    return IRB.CreateIntrinsic(Int64Ty, Intrinsic::amdgcn_s_getpc, {}, nullptr,
                               "PC");
  }

  Module &M;
  const DataLayout &DL = M.getDataLayout();
  FunctionAnalysisManager &FAM;
  LLVMContext &Ctx;

  Type *VoidTy = Type::getVoidTy(Ctx);
  Type *IntptrTy = M.getDataLayout().getIntPtrType(Ctx);
  PointerType *PtrTy = PointerType::getUnqual(Ctx);
  IntegerType *Int8Ty = Type::getInt8Ty(Ctx);
  IntegerType *Int32Ty = Type::getInt32Ty(Ctx);
  IntegerType *Int64Ty = Type::getInt64Ty(Ctx);
  PointerType *AllocaPtrTy = PointerType::get(Ctx, DL.getAllocaAddrSpace());
  PointerType *ASPtrTy[NumSupportedAddressSpaces] = {
      PointerType::get(Ctx, 0), PointerType::get(Ctx, 1),
      PointerType::get(Ctx, 2), PointerType::get(Ctx, 3),
      PointerType::get(Ctx, 4), PointerType::get(Ctx, 5)};
  Type *AllocationInfoTy = StructType::get(Ctx, {ASPtrTy[1], Int64Ty}, true);
  Type *InfoTy = StructType::get(Ctx, {AllocationInfoTy, Int32Ty}, true);
};

} // end anonymous namespace

bool OffloadSanitizerImpl::shouldInstrumentFunction(Function *Fn) {
  if (!Fn || Fn->isDeclaration())
    return false;
  //  if (Fn->getName().contains("ompx") || Fn->getName().contains("__kmpc") ||
  //      Fn->getName().starts_with("rpc_"))
  //    return false;
  return !Fn->hasFnAttribute(Attribute::DisableSanitizerInstrumentation);
}

void OffloadSanitizerImpl::removeAS(Function &Fn,
                                    SmallVectorImpl<Instruction *> &ASInsts) {

  DenseMap<Value *, Value *> VMap;

  std::function<Value *(Value &)> GetAsGeneric = [&](Value &V) -> Value * {
    if (!isASType(*V.getType()))
      return &V;
    auto *&NewV = VMap[&V];
    if (!NewV) {
      auto *IP = &Fn.getEntryBlock().front();
      if (auto *PHI = dyn_cast<PHINode>(&V))
        IP = PHI->getParent()->getFirstNonPHIOrDbgOrLifetime();
      else if (auto *I = dyn_cast<Instruction>(&V))
        IP = I->getNextNode();
      NewV = new AddrSpaceCastInst(&V, getWithoutAS(*V.getType()),
                                   V.getName() + ".noas", IP);
    }
    return NewV;
  };

  SmallVector<PHINode *> PHIs;
  for (auto *I : ASInsts) {
    errs() << "I: " << *I << "\n";
    switch (I->getOpcode()) {
    case Instruction::Load: {
      auto &LI = cast<LoadInst>(*I);
      auto *GenericOp = GetAsGeneric(*LI.getPointerOperand());
      if (LI.getPointerAddressSpace())
        GenericOp = new AddrSpaceCastInst(GenericOp, LI.getPointerOperandType(),
                                          GenericOp->getName() + ".as", &LI);
      LI.setOperand(LI.getPointerOperandIndex(), GenericOp);
      VMap[I] = GetAsGeneric(LI);
      break;
    }
    case Instruction::Store: {
      auto &SI = cast<StoreInst>(*I);
      auto *GenericOp = GetAsGeneric(*SI.getPointerOperand());
      if (SI.getPointerAddressSpace())
        GenericOp = new AddrSpaceCastInst(GenericOp, SI.getPointerOperandType(),
                                          GenericOp->getName() + ".as", &SI);
      SI.setOperand(SI.getPointerOperandIndex(), GenericOp);
      break;
    }
    case Instruction::AtomicRMW: {
      auto &ARMW = cast<AtomicRMWInst>(*I);
      auto *GenericOp = GetAsGeneric(*ARMW.getPointerOperand());
      if (ARMW.getPointerAddressSpace())
        GenericOp = new AddrSpaceCastInst(GenericOp,
                                          ARMW.getPointerOperand()->getType(),
                                          GenericOp->getName() + ".as", &ARMW);
      ARMW.setOperand(ARMW.getPointerOperandIndex(), GenericOp);
      VMap[I] = GetAsGeneric(ARMW);
      break;
    }
    case Instruction::AtomicCmpXchg: {
      auto &ACX = cast<AtomicCmpXchgInst>(*I);
      auto *GenericOp = GetAsGeneric(*ACX.getPointerOperand());
      if (ACX.getPointerAddressSpace())
        GenericOp =
            new AddrSpaceCastInst(GenericOp, ACX.getPointerOperand()->getType(),
                                  GenericOp->getName() + ".as", &ACX);
      ACX.setOperand(ACX.getPointerOperandIndex(), GenericOp);
      VMap[I] = GetAsGeneric(ACX);
      break;
    }
    case Instruction::GetElementPtr: {
      auto &GEP = cast<GetElementPtrInst>(*I);
      GEP.mutateType(getWithoutAS(*GEP.getType()));
      GEP.setSourceElementType(getWithoutAS(*GEP.getSourceElementType()));
      GEP.setResultElementType(getWithoutAS(*GEP.getResultElementType()));
      GEP.setOperand(GEP.getPointerOperandIndex(),
                     GetAsGeneric(*GEP.getPointerOperand()));
      break;
    }
    case Instruction::AddrSpaceCast: {
      auto &ASC = cast<AddrSpaceCastInst>(*I);
      Value *PlainV;
      VMap[I] = PlainV = GetAsGeneric(*ASC.getPointerOperand());
      while (!ASC.use_empty()) {
        Use &U = *ASC.use_begin();
        U.set(PlainV);
      }
      ASC.eraseFromParent();
      break;
    }
    case Instruction::Select: {
      auto &SI = cast<SelectInst>(*I);
      SI.mutateType(getWithoutAS(*SI.getType()));
      SI.setTrueValue(GetAsGeneric(*SI.getTrueValue()));
      SI.setFalseValue(GetAsGeneric(*SI.getFalseValue()));
      break;
    }
    case Instruction::PHI: {
      auto &PHI = cast<PHINode>(*I);
      PHI.mutateType(getWithoutAS(*PHI.getType()));
      PHIs.push_back(&PHI);
      break;
    }
    case Instruction::ICmp: {
      auto &II = cast<ICmpInst>(*I);
      II.setOperand(0, GetAsGeneric(*II.getOperand(0)));
      II.setOperand(1, GetAsGeneric(*II.getOperand(1)));
      break;
    }
    case Instruction::Call: {
      auto &CI = cast<CallInst>(*I);
      auto *Callee = CI.getCalledFunction();
      if (shouldInstrumentFunction(Callee)) {
        for (unsigned I = 0, E = CI.arg_size(); I < E; ++I)
          CI.setArgOperand(I, GetAsGeneric(*CI.getArgOperand(I)));

        auto *FT = CI.getFunctionType();
        SmallVector<Type *> ArgTypes;
        for (auto *ArgType : FT->params())
          ArgTypes.push_back(getWithoutAS(*ArgType));
        FunctionType *NewFT = FunctionType::get(
            getWithoutAS(*FT->getReturnType()), ArgTypes, FT->isVarArg());
        CI.mutateFunctionType(NewFT);
        CI.mutateType(getWithoutAS(*CI.getType()));
      } else {
        if (isASType(*CI.getType()))
          VMap[I] = GetAsGeneric(CI);
        IRBuilder<> IRB(&CI);
        for (unsigned I = 0, E = CI.arg_size(); I < E; ++I) {
          auto *Op = CI.getArgOperand(I);
          if (!isASType(*Op->getType()))
            continue;
          auto *NewOp = GetAsGeneric(*Op);
          Value *NewArg = IRB.CreateAddrSpaceCast(NewOp, Op->getType());
          CI.setArgOperand(I, NewArg);
        }
      }
      break;
    }
    default:
      I->dump();
      llvm_unreachable("Instruction with AS not handled");
    }
    //  if (VMap.count(I))
    //    errs() << "I: " << *I << " --> " << *VMap[I] << "\n";
  }

  for (auto *PHI : PHIs)
    for (unsigned I = 0, E = PHI->getNumIncomingValues(); I < E; ++I)
      PHI->setIncomingValue(I, GetAsGeneric(*PHI->getIncomingValue(I)));
}

void OffloadSanitizerImpl::instrumentCallInsts(
    SmallVectorImpl<CallInst *> &CallInsts) {
  for (auto *CI : CallInsts) {
    assert(!isa<LifetimeIntrinsic>(CI));
    auto *Fn = CI->getCalledFunction();
    if (shouldInstrumentFunction(Fn))
      continue;
    IRBuilder<> IRB(CI);
    for (int I = 0, E = CI->arg_size(); I != E; ++I) {
      Value *Op = CI->getArgOperand(I);
      if (!Op->getType()->isPointerTy())
        continue;
      auto *PlainOp = Op;
      auto AS = Op->getType()->getPointerAddressSpace();
      if (AS)
        if (auto *AC = dyn_cast<AddrSpaceCastInst>(Op))
          PlainOp = AC->getPointerOperand();
      auto *CB = createCall(IRB, getUnpackFn(AS),
                            {getPC(IRB), getSourceIndex(*CI),
                             IRB.CreateAddrSpaceCast(PlainOp, PtrTy)},
                            Op->getName() + ".unpack");
      CI->setArgOperand(I, CB);
    }
  }
}

void OffloadSanitizerImpl::instrumentLifetimeIntrinsics(
    SmallVectorImpl<LifetimeIntrinsic *> &LifetimeInsts) {
  for (auto *LI : LifetimeInsts)
    LI->eraseFromParent();
}

void OffloadSanitizerImpl::instrumentTrapInstructions(
    SmallVectorImpl<IntrinsicInst *> &TrapCalls) {
  for (auto *II : TrapCalls) {
    IRBuilder<> IRB(II);
    createCall(IRB, getTrapInfoFn(), {getPC(IRB), getSourceIndex(*II)});
  }
}

void OffloadSanitizerImpl::instrumentUnreachableInstructions(
    SmallVectorImpl<UnreachableInst *> &UnreachableInsts) {
  for (auto *II : UnreachableInsts) {
    // Skip unreachables after traps since we instrument those as well.
    if (&II->getParent()->front() != II)
      if (auto *CI = dyn_cast<CallInst>(II->getPrevNode()))
        if (CI->getIntrinsicID() == Intrinsic::trap)
          continue;
    IRBuilder<> IRB(II);
    createCall(IRB, getUnreachableInfoFn(), {getPC(IRB), getSourceIndex(*II)});
  }
}

void OffloadSanitizerImpl::instrumentAccesses(
    SmallVectorImpl<AccessInfoTy> &AccessInfos) {
  for (auto &AI : AccessInfos) {
    auto *FakePtr = AI.I->getOperand(AI.PtrOpIdx);
    auto *Size =
        ConstantInt::get(Int32Ty, DL.getTypeStoreSize(AI.I->getAccessType()));
    if (FakePtr->getType()->getPointerAddressSpace()) {
      auto *ASC = cast<AddrSpaceCastInst>(FakePtr);
      FakePtr = ASC->getPointerOperand();
    }
    assert(FakePtr->getType()->getPointerAddressSpace() == 0);

    IRBuilder<> IRB(AI.I);
    SmallVector<Value *> Args;
    Args.append({getPC(IRB), getSourceIndex(*AI.I), FakePtr, Size});

    auto *Obj = getUnderlyingObject(FakePtr);
    if (AI.AS == 0)
      AI.AS = Obj->getType()->getPointerAddressSpace();
    if (AI.AS == 0 && isa<Argument>(Obj))
      if (cast<Argument>(Obj)->getParent()->getCallingConv() ==
          CallingConv::AMDGPU_KERNEL)
        AI.AS = 1;
    if (AI.AS == 1)
      AI.AS = 0;
    const auto &PtrInfo = getPtrInfoTy(*Obj, AI);
    Args.push_back(PtrInfo.AS);
    Args.push_back(PtrInfo.Start);
    Args.push_back(PtrInfo.Length);
    FunctionCallee FC = getCheckAccessWithInfoFn(AI.AS);

    auto *RealPtr = createCall(IRB, FC, Args);
    AI.I->setOperand(AI.PtrOpIdx, RealPtr);
    assert(RealPtr->getParent());
  }
}

void OffloadSanitizerImpl::instrumentAllocaInstructions(
    SmallVectorImpl<AllocaInst *> &AllocaInsts) {

  auto IsApplicable = [&](AllocaInst &AI, TypeSize &TS) {
    // Check the type and size.
    if (AI.getAllocatedType()->isScalableTy())
      return false;
    auto AllocSize = AI.getAllocationSize(DL);
    assert(AllocSize && "Alloc size not known!");
    if (AllocSize->getKnownMinValue() >= (1UL << 32))
      return false;
    TS = *AllocSize;
    return true;
  };

  for (auto *AI : AllocaInsts) {
    TypeSize TS(0, false);
    if (!IsApplicable(*AI, TS))
      continue;

    IRBuilder<> IRB(AI->getNextNode());
    auto *Size = ConstantInt::get(Int32Ty, TS);
    auto *FakePtr = createCall(IRB, getAllocaRegisterFn(),
                               {getPC(IRB), getSourceIndex(*AI), AI, Size});
    for (auto *U : AI->users()) {
      auto *UI = cast<Instruction>(U);
      if (UI == FakePtr)
        continue;
      if (!isa<AddrSpaceCastInst>(UI)) {
        AI->getFunction()->dump();
        AI->dump();
        UI->dump();
      }
      assert(isa<AddrSpaceCastInst>(UI) &&
             "Expected only address space casts users of allocas");
      assert(UI->getType()->getPointerAddressSpace() == 0 &&
             "Expected only address space casts to AS 0 as users of allocas");
      UI->replaceAllUsesWith(FakePtr);
    }
  }
}

bool OffloadSanitizerImpl::instrumentFunction(Function &Fn) {
  if (!shouldInstrumentFunction(&Fn))
    return false;

  if (Fn.getCallingConv() == CallingConv::AMDGPU_KERNEL)
    Kernels.push_back(&Fn);

  AllocationInfoMap.clear();

  SmallVector<UnreachableInst *> UnreachableInsts;
  SmallVector<IntrinsicInst *> TrapCalls;
  SmallVector<AllocaInst *> AllocaInsts;
  SmallVector<AccessInfoTy> AccessInfos;
  SmallVector<Instruction *> ASInsts;
  SmallVector<LifetimeIntrinsic *> LifetimeInsts;
  SmallVector<CallInst *> CallInsts;
  SmallVector<AddrSpaceCastInst *> ASCInsts;

  ReversePostOrderTraversal<Function *> RPOT(&Fn);
  for (auto &It : RPOT) {
    for (auto &I : *It) {
      if (!I.getType()->isVoidTy())
        I.setName("I");
      switch (I.getOpcode()) {
      case Instruction::Alloca:
        AllocaInsts.push_back(cast<AllocaInst>(&I));
        break;
      case Instruction::Store: {
        auto &SI = cast<StoreInst>(I);
        AccessInfos.push_back(
            {&I, SI.getPointerOperandIndex(), SI.getPointerAddressSpace()});
        if (isASType(*SI.getPointerOperandType()))
          ASInsts.push_back(&I);
        break;
      }
      case Instruction::Load: {
        auto &LI = cast<LoadInst>(I);
        AccessInfos.push_back(
            {&I, LI.getPointerOperandIndex(), LI.getPointerAddressSpace()});
        if (isASType(*LI.getType()) || isASType(*LI.getPointerOperandType()))
          ASInsts.push_back(&I);
        break;
      }
      case Instruction::AtomicRMW: {
        auto &ARMW = cast<AtomicRMWInst>(I);
        AccessInfos.push_back(
            {&I, ARMW.getPointerOperandIndex(), ARMW.getPointerAddressSpace()});
        if (isASType(*ARMW.getType()) ||
            isASType(*ARMW.getPointerOperand()->getType()))
          ASInsts.push_back(&I);
        break;
      }
      case Instruction::AtomicCmpXchg: {
        auto &ACX = cast<AtomicCmpXchgInst>(I);
        AccessInfos.push_back(
            {&I, ACX.getPointerOperandIndex(), ACX.getPointerAddressSpace()});
        if (isASType(*ACX.getType()) ||
            isASType(*ACX.getPointerOperand()->getType()))
          ASInsts.push_back(&I);
        break;
      }
      case Instruction::Unreachable:
        UnreachableInsts.push_back(cast<UnreachableInst>(&I));
        break;
      case Instruction::Call: {
        auto &CI = cast<CallInst>(I);
        if (CI.isIndirectCall())
          AmbiguousCalls.insert(&CI);
        bool Handled = false;
        if (auto *II = dyn_cast<IntrinsicInst>(&CI)) {
          switch (II->getIntrinsicID()) {
          case Intrinsic::trap:
            Handled = true;
            TrapCalls.push_back(II);
            break;
          case Intrinsic::lifetime_start:
          case Intrinsic::lifetime_end:
            Handled = true;
            LifetimeInsts.push_back(cast<LifetimeIntrinsic>(II));
            break;
          }
        }
        if (!Handled)
          CallInsts.push_back(&CI);
        if (isASType(*CI.getType()))
          ASInsts.push_back(&I);
        else if (any_of(CI.args(),
                        [&](Value *Op) { return isASType(*Op->getType()); }))
          ASInsts.push_back(&I);
        break;
      }
      case Instruction::AddrSpaceCast:
        ASCInsts.push_back(cast<AddrSpaceCastInst>(&I));
        ASInsts.push_back(&I);
        break;
      case Instruction::GetElementPtr:
        if (isASType(*I.getType()))
          ASInsts.push_back(&I);
        break;
      default:
        if (isASType(*I.getType()))
          ASInsts.push_back(&I);
        else if (any_of(I.operand_values(),
                        [&](Value *Op) { return isASType(*Op->getType()); }))
          ASInsts.push_back(&I);
        break;
      }
    }
  }

  Fn.dump();
  removeAS(Fn, ASInsts);
  instrumentCallInsts(CallInsts);
  instrumentLifetimeIntrinsics(LifetimeInsts);
  instrumentTrapInstructions(TrapCalls);
  instrumentUnreachableInstructions(UnreachableInsts);
  instrumentAccesses(AccessInfos);
  instrumentAllocaInstructions(AllocaInsts);

  // for (auto *ASC : ASCInsts) {
  //   if (ASC->getPointerOperand()->getType() == ASC->getType())
  //     ASC->replaceAllUsesWith(ASC->getPointerOperand());
  //   if (ASC->use_empty())
  //     ASC->eraseFromParent();
  // }

  //  for (auto *CI : RTCalls) {
  //    InlineFunctionInfo IFI;
  //    InlineFunction(*CI, IFI);
  //  }
  RTCalls.clear();

  auto &BB = Fn.getEntryBlock();
  SmallVector<AllocaInst *> Allocas;
  for (auto &I : BB)
    if (auto *AI = dyn_cast<AllocaInst>(&I))
      Allocas.push_back(AI);
  for (auto *AI : Allocas)
    AI->moveBefore(&*BB.getFirstInsertionPt());

  return true;
}

bool OffloadSanitizerImpl::handleAmbiguousCalls() {
  if (AmbiguousCalls.empty())
    return false;

  SmallVector<CallBase *> AmbiguousCallsOrdered;
  SmallVector<Constant *> AmbiguousCallsMapping;
  for (size_t I = 0; I < AmbiguousCalls.size(); ++I) {
    CallBase &CB = *AmbiguousCalls[I];
    AmbiguousCallsOrdered.push_back(&CB);
    AmbiguousCallsMapping.push_back(getSourceIndex(CB));
  }

  uint64_t AmbiguousCallsBitWidth =
      llvm::Log2_64_Ceil(AmbiguousCalls.size() + 1);

  new GlobalVariable(M, Int64Ty, /*isConstant=*/true,
                     GlobalValue::ExternalLinkage,
                     ConstantInt::get(Int64Ty, AmbiguousCallsBitWidth),
                     "__offload_san_num_ambiguous_calls", nullptr,
                     GlobalValue::ThreadLocalMode::NotThreadLocal, 1);

  size_t NumAmbiguousCalls = AmbiguousCalls.size();
  {
    auto *ArrayTy = ArrayType::get(Int64Ty, NumAmbiguousCalls);
    auto *GV = new GlobalVariable(
        M, ArrayTy, /*isConstant=*/true, GlobalValue::ExternalLinkage,
        ConstantArray::get(ArrayTy, AmbiguousCallsMapping),
        "__offload_san_ambiguous_calls_mapping", nullptr,
        GlobalValue::ThreadLocalMode::NotThreadLocal, 4);
    GV->setVisibility(GlobalValue::ProtectedVisibility);
  }

  auto *ArrayTy = ArrayType::get(Int64Ty, 1024);
  LocationsArray = new GlobalVariable(
      M, ArrayTy, /*isConstant=*/false, GlobalValue::PrivateLinkage,
      UndefValue::get(ArrayTy), "__offload_san_calls", nullptr,
      GlobalValue::ThreadLocalMode::NotThreadLocal, SHARED_ADDRSPACE);

  for (const auto &It : llvm::enumerate(AmbiguousCallsOrdered)) {
    IRBuilder<> IRB(It.value());
    Value *Idx = createCall(IRB, getThreadIdFn(), {}, "san.gtid");
    Value *Ptr = IRB.CreateGEP(Int64Ty, LocationsArray, {Idx});
    Value *OldVal = IRB.CreateLoad(Int64Ty, Ptr);
    Value *OldValShifted = IRB.CreateShl(
        OldVal, ConstantInt::get(Int64Ty, AmbiguousCallsBitWidth));
    Value *NewVal = IRB.CreateBinOp(Instruction::Or, OldValShifted,
                                    ConstantInt::get(Int64Ty, It.index() + 1));
    IRB.CreateStore(NewVal, Ptr);
    IRB.SetInsertPoint(It.value()->getNextNode());
    IRB.CreateStore(OldVal, Ptr);
  }

  return true;
}

bool OffloadSanitizerImpl::handleCallStackSupport() {
  if (LocationMap.empty())
    return false;

  handleAmbiguousCalls();

  auto *NamesTy = ArrayType::get(Int8Ty, ConcatenatedString.size() + 1);
  auto *Names = new GlobalVariable(
      M, NamesTy, /*isConstant=*/true, GlobalValue::ExternalLinkage,
      ConstantDataArray::getString(Ctx, ConcatenatedString),
      "__offload_san_location_names", nullptr,
      GlobalValue::ThreadLocalMode::NotThreadLocal, 4);
  Names->setVisibility(GlobalValue::ProtectedVisibility);

  auto *ArrayTy = ArrayType::get(Int64Ty, LocationEncoding.size());
  auto *GV = new GlobalVariable(
      M, ArrayTy, /*isConstant=*/true, GlobalValue::ExternalLinkage,
      ConstantArray::get(ArrayTy, LocationEncoding), "__offload_san_locations",
      nullptr, GlobalValue::ThreadLocalMode::NotThreadLocal, 4);
  GV->setVisibility(GlobalValue::ProtectedVisibility);

  return true;
}

bool OffloadSanitizerImpl::finalizeKernels() {
  for (auto *Kernel : Kernels) {
    Function *InitKernelFn = createSanitizerInitKernel();
    IRBuilder<> IRB(&*Kernel->getEntryBlock().getFirstNonPHIOrDbgOrAlloca());
    createCall(IRB, InitKernelFn, {});
  }
  return Kernels.size();
}

bool OffloadSanitizerImpl::addCtor() {
  Function *CtorFn =
      Function::Create(FunctionType::get(VoidTy, false),
                       GlobalValue::PrivateLinkage, "__offload_san_ctor", &M);
  CtorFn->addFnAttr(Attribute::DisableSanitizerInstrumentation);

  BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", CtorFn);
  IRBuilder<> IRB(Entry);

  IRB.CreateRetVoid();

  appendToGlobalCtors(M, CtorFn, 0, nullptr);
  return true;
}

bool OffloadSanitizerImpl::addDtor() {
  Function *DtorFn =
      Function::Create(FunctionType::get(VoidTy, false),
                       GlobalValue::PrivateLinkage, "__offload_san_dtor", &M);
  DtorFn->addFnAttr(Attribute::DisableSanitizerInstrumentation);
  BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", DtorFn);
  IRBuilder<> IRB(Entry);

  createCall(IRB, getLeakCheckFn());

  IRB.CreateRetVoid();
  appendToGlobalDtors(M, DtorFn, 0, nullptr);
  return true;
}

Function *OffloadSanitizerImpl::createSanitizerInitKernel() {
  if (auto *Fn = M.getFunction("__offload_san_init_kernel"))
    return Fn;

  Function *InitSharedFn = Function::Create(FunctionType::get(VoidTy, false),
                                            GlobalValue::PrivateLinkage,
                                            "__offload_san_init_kernel", &M);
  InitSharedFn->addFnAttr(Attribute::DisableSanitizerInstrumentation);

  auto *EntryBB = BasicBlock::Create(Ctx, "entry", InitSharedFn);
  IRBuilder<> IRB(EntryBB, EntryBB->getFirstNonPHIOrDbgOrAlloca());
  auto *Barrier = createCall(IRB, getSyncBlockFn());
  IRB.CreateRetVoid();
  IRB.SetInsertPoint(Barrier);

  if (!AmbiguousCalls.empty()) {
    Value *Idx = createCall(IRB, getThreadIdFn(), {}, "san.gtid");
    Value *Ptr = IRB.CreateGEP(Int64Ty, LocationsArray, {Idx});
    IRB.CreateStore(ConstantInt::get(Int64Ty, 42), Ptr);

    auto *CondV = IRB.CreateICmpEQ(Idx, IRB.getInt32(0));

    auto *CondTI = SplitBlockAndInsertIfThen(CondV, Barrier, false);
    IRB.SetInsertPoint(CondTI);
    auto *AmbiguousCallsInfoPtrGV =
        M.getNamedGlobal("__offload_san_ambiguous_calls_info_ptr");
    assert(AmbiguousCallsInfoPtrGV);
    IRB.CreateStore(LocationsArray, AmbiguousCallsInfoPtrGV);
  }

  InitSharedFn->dump();
  return InitSharedFn;
}

constexpr StringRef ShadowGlobalPrefix = "__offload_san_global.";
constexpr StringRef ShadowSharedPrefix = "__offload_san_shared.";
constexpr StringRef GlobalIgnorePrefix[] = {"llvm.", "__offload_san", "__san"};
static bool canInstrumentGlobal(const GlobalVariable &G) {
  auto Name = G.getName();
  if (Name.empty())
    return false;
  for (const auto &S : GlobalIgnorePrefix) {
    if (Name.starts_with(S))
      return false;
  }
  return true;
}

static Twine getShadowGlobalName(const GlobalValue &G, bool IsShared) {
  return (IsShared ? ShadowSharedPrefix : ShadowGlobalPrefix) + G.getName();
}

bool OffloadSanitizerImpl::instrumentGlobals() {
  bool Changed = false;
  for (GlobalVariable &GV : M.globals()) {
    if (!canInstrumentGlobal(GV))
      continue;

    bool IsShared = isSharedGlobal(GV);
    GlobalVariable *ShadowVar;
    if (IsShared) {
      ShadowVar = new GlobalVariable(
          M, ASPtrTy[SHARED_ADDRSPACE], false, GlobalValue::ExternalLinkage,
          UndefValue::get(ASPtrTy[SHARED_ADDRSPACE]),
          getShadowGlobalName(GV, IsShared), &GV,
          GlobalValue::ThreadLocalMode::NotThreadLocal, SHARED_ADDRSPACE);
    } else {
      ShadowVar =
          new GlobalVariable(M, IntptrTy, false, GlobalValue::ExternalLinkage,
                             Constant::getNullValue(IntptrTy),
                             getShadowGlobalName(GV, IsShared), &GV);
    }

    ShadowVar->setVisibility(GlobalValue::ProtectedVisibility);
    //    ShadowGlobalMap[&GV] = ShadowVar;
    Changed = true;
  }

  return Changed;
}

bool OffloadSanitizerImpl::instrument() {
  bool Changed = instrumentGlobals();

  for (Function &Fn : M)
    Changed |= instrumentFunction(Fn);

  Changed |= addCtor();
  Changed |= addDtor();
  Changed |= handleCallStackSupport();
  Changed |= finalizeKernels();

  removeFromUsedLists(M, [&](Constant *C) {
    if (!C->getName().starts_with("__offload_san"))
      return false;
    return Changed = true;
  });

  return Changed;
}

PreservedAnalyses OffloadSanitizerPass::run(Module &M,
                                            ModuleAnalysisManager &AM) {
  FunctionAnalysisManager &FAM =
      AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  OffloadSanitizerImpl Impl(M, FAM);
  if (!Impl.instrument())
    return PreservedAnalyses::all();
  M.dump();
  assert(!verifyModule(M, &errs()));
  return PreservedAnalyses::none();
}
