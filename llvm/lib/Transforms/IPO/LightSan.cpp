//===-- LightSan.cpp - Sanitization instrumentation pass ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/LightSan.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/GEPNoWrapFlags.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/Attributor.h"
#include "llvm/Transforms/IPO/Instrumentor.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/SROA.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#include "llvm/Transforms/Utils/SimplifyCFGOptions.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <cassert>
#include <cstdint>
#include <functional>

using namespace llvm;
using namespace llvm::instrumentor;

#define DEBUG_TYPE "lightsan"

static constexpr char LightSanRuntimePrefix[] = "__objsan_";
static constexpr char LightSanGlobalShadowPrefix[] = "__objsan_shadow.";
static constexpr char AdapterPrefix[] = "__adapter_";
[[maybe_unused]] static constexpr uint8_t SmallObjectEnc = 1;
[[maybe_unused]] static constexpr uint8_t LargeObjectEnc = 2;
[[maybe_unused]] static constexpr uint64_t SmallObjectSize = (1LL << 12);

namespace {

struct LightSanImpl;

/// Information cache collected from attributor.
class AttributorInfoCache {
public:
  AttributorInfoCache(Attributor &A) : A(A) {}

  Value *getSafeAccessObj(Instruction *I) const {
    return SafeAccesses.lookup(I);
  }
  bool isAccessSafe(Instruction *I) const { return SafeAccesses.lookup(I); }
  bool insertSafeAccess(Instruction *I, Value *Obj) {
    return (SafeAccesses[I] = Obj);
  }

  bool insertSafeObject(Value *Obj) { return SafeObjects.insert(Obj); }

  bool isObjectSafe(Value *Obj) const { return SafeObjects.contains(Obj); }

  bool insertKnownObject(Value *Obj, uint64_t Size) {
    return (SanitizedObjects[Obj] = Size);
  }
  bool isKnownObject(Value *Obj) const {
    return SanitizedObjects.contains(Obj);
  }
  Value *getUnderlyingObject(Value *Obj) const {
    if (!Obj)
      return nullptr;
    if (auto *UO = getPreRegister(Obj))
      return UO;
    if (auto *AAUO = getAAUO(Obj)) {
      Value *UO = nullptr;
      if (AAUO->forallUnderlyingObjects([&](Value &V) {
            if (UO && UO != &V)
              return false;
            UO = &V;
            return true;
          }))
        return UO;
    }
    return nullptr;
  }
  uint64_t getObjectSize(Value *Obj, bool *CanEscape = nullptr) const {
    if (auto *AAUO = getAAUO(Obj)) {
      uint64_t Size = ~0U;
      if (AAUO->forallUnderlyingObjects([&](Value &V) {
            uint64_t VSize = SanitizedObjects.lookup(&V);
            if (!isKnownObject(&V))
              return false;
            if (CanEscape)
              *CanEscape = !isNonEscapingObj(&V);
            if (Size != ~0U && VSize != Size)
              return false;
            Size = VSize;
            return true;
          }))
        return Size;
    }
    if (isKnownObject(Obj)) {
      if (CanEscape)
        *CanEscape = !isNonEscapingObj(Obj);
      return SanitizedObjects.lookup(Obj);
    }
    if (CanEscape)
      *CanEscape = true;
    return ~0UL;
  }
  uint64_t getEncodingNo(Value *Obj) const {
    bool CanEscape = false;
    uint64_t Size = getObjectSize(Obj, &CanEscape);
    if (Size == ~0UL)
      return ~0UL;
    if (Size > SmallObjectSize || CanEscape)
      return LargeObjectEnc;
    return SmallObjectEnc;
  }
  bool insertNonEscapingObj(Value *Obj) {
    return NonEscapingObjects.insert(Obj);
  }
  bool isNonEscapingObj(Value *Obj) const {
    return NonEscapingObjects.contains(Obj);
  }

  void insertRegisterCall(Value *Obj, CallInst *CI) {
    RegisterCallsMap[Obj] = CI;
  }
  Value *getPreRegister(Value *Obj) const {
    if (!Obj)
      return nullptr;
    if (auto *UO = RegisterCallsMap.lookup(Obj))
      return UO;
    if (auto *LI = dyn_cast<LoadInst>(Obj)) {
      if (auto *GV = dyn_cast<GlobalVariable>(LI->getPointerOperand()))
        if (GV->getName().starts_with(LightSanGlobalShadowPrefix))
          return GV->getParent()->getGlobalVariable(
              GV->getName().drop_front(strlen(LightSanGlobalShadowPrefix)));
    }
    return nullptr;
  }

  Attributor &getAttributor() const { return A; }

private:
  const AAUnderlyingObjects *getAAUO(Value *Obj) const {
    if (Obj)
      if (auto *AAUO =
              A.lookupAAFor<AAUnderlyingObjects>(IRPosition::value(*Obj)))
        return AAUO;
    return nullptr;
  }
  DenseMap<Value *, uint64_t> SanitizedObjects;
  SetVector<Value *> NonEscapingObjects;
  /// Objects that have been verified that all their accesses are safe.
  SetVector<Value *> SafeObjects;
  /// Accesses that are known safe.
  DenseMap<Instruction *, Value *> SafeAccesses;
  DenseMap<Value *, CallInst *> RegisterCallsMap;

  Attributor &A;
};

struct LightSanInstrumentationConfig : public InstrumentationConfig {

  LightSanInstrumentationConfig(LightSanImpl &LSI, Module &M);
  virtual ~LightSanInstrumentationConfig() {}

  void initializeFunctionCallees(Module &M);

  void populate(InstrumentorIRBuilderTy &IRB) override;

  struct ExtendedBasePointerInfo {
    Value *ObjectSize = nullptr;
    Value *ObjectSizePtr = nullptr;
    Value *EncodingNo = nullptr;
  };

  DenseMap<std::pair<Value *, Function *>, ExtendedBasePointerInfo>
      BasePointerSizeOffsetMap;

  Value *getBasePointerObjectSize(Value &Ptr, InstrumentorIRBuilderTy &IIRB) {
    Function *Fn = IIRB.IRB.GetInsertBlock()->getParent();
    auto Size = AIC->getObjectSize(&Ptr);
    if (Size != ~0UL)
      return IIRB.IRB.getInt64(Size);
    Value *Obj = getUnderlyingObjectRecursive(&Ptr);
    auto &EBPI = BasePointerSizeOffsetMap[{Obj, Fn}];
    if (!EBPI.ObjectSize) {
      getBasePointerInfo(*Obj, IIRB);
      EBPI = BasePointerSizeOffsetMap[{Obj, Fn}];
    }
    if (EBPI.ObjectSizePtr)
      return IIRB.IRB.CreateLoad(IIRB.Int64Ty, EBPI.ObjectSizePtr);
    assert(EBPI.ObjectSize);
    return EBPI.ObjectSize;
  }

  Value *getBasePointerEncodingNo(Value &Ptr, InstrumentorIRBuilderTy &IIRB) {
    Function *Fn = IIRB.IRB.GetInsertBlock()->getParent();
    auto EncodingNo = AIC->getEncodingNo(&Ptr);
    if (EncodingNo != ~0UL)
      return IIRB.IRB.getInt8(EncodingNo);
    Value *Obj = getUnderlyingObjectRecursive(&Ptr);
    auto &EBPI = BasePointerSizeOffsetMap[{Obj, Fn}];
    if (!EBPI.ObjectSize) {
      getBasePointerInfo(*Obj, IIRB);
      EBPI = BasePointerSizeOffsetMap[{Obj, Fn}];
    }
    assert(EBPI.EncodingNo);
    return EBPI.EncodingNo;
  }
  /// Get a size alloca.
  AllocaInst *getSizeAlloca(Function &Fn, InstrumentorIRBuilderTy &IIRB,
                            Value &Obj) {
    const DataLayout &DL = Fn.getDataLayout();
    /// TODO: check if the size of base ptr can change in the function, if not,
    /// use a temporary alloca. Globals and allocas don't change size in the
    /// function.
    auto *TmpAI = IIRB.getAlloca(&Fn, IIRB.Int64Ty, /*MatchType=*/true);
    if (isa<GlobalValue>(Obj) || isa<AllocaInst>(Obj)) {
      return TmpAI;
    }
    auto *SizeAI = new AllocaInst(IIRB.Int64Ty, DL.getAllocaAddrSpace(), "size",
                                  Fn.getEntryBlock().begin());
    SizeAllocas.push_back({&Obj, SizeAI});
    TmpToSizeAllocas[TmpAI] = SizeAI;
    return TmpAI;
  }
  SmallVector<std::pair<Value *, AllocaInst *>> SizeAllocas;
  DenseMap<AllocaInst *, AllocaInst *> TmpToSizeAllocas;

  void startFunction() override {
    EscapedAllocas.clear();
    SizeAllocas.clear();
    PotentiallyFreeCalls.clear();
  }

  // This is a vector of allocas holding call inst that registered user allocas.
  SmallVector<AllocaInst *> EscapedAllocas;
  SmallVector<CallInst *> PotentiallyFreeCalls;

  Value *getBaseMPtr(Value &VPtr, InstrumentorIRBuilderTy &IIRB) {
    return getBasePointerInfo(VPtr, IIRB);
  }

  Value *getMPtr(Value &VPtr, InstrumentorIRBuilderTy &IIRB) {
    auto *Fn = IIRB.IRB.GetInsertBlock()->getParent();

#if 0
    IRBuilderBase::InsertPointGuard IPG(IIRB.IRB);
    auto BestIP = IIRB.getBestHoistPoint(IIRB.IRB.GetInsertPoint(),
                                         HoistKindTy::HOIST_MAXIMALLY);
    DenseMap<std::pair<Instruction *, unsigned>, Value *> ReplaceMap;
#endif
    SmallVector<Value *> Worklist;
    Worklist.push_back(&VPtr);

    while (!Worklist.empty()) {
      Value *Ptr = Worklist.pop_back_val();
      auto *&MPtr = V2M[{Ptr, Fn}];
      if (MPtr)
        continue;
#if 0
      // This does not work since stuff moves and we break dominance.
      if (auto *CE = dyn_cast<ConstantExpr>(Ptr)) {
        if (CE->getOpcode() == Instruction::GetElementPtr) {
          auto *GEP = CE->getAsInstruction();
          GEP->insertBefore(BestIP);
          auto *PtrOp = GEP->getOperand(0);
          ReplaceMap[{GEP, 0}] = PtrOp;
          Worklist.push_back(PtrOp);
          MPtr = GEP;
        }
        continue;
      }
      auto *PtrI = dyn_cast<Instruction>(Ptr);
      errs() << "Ptr " << *Ptr << "\n";
      if (PtrI) {
        IIRB.IRB.SetInsertPoint(*PtrI->getInsertionPointAfterDef());
        ensureDbgLoc(IIRB.IRB);
        switch (PtrI->getOpcode()) {
        case Instruction::GetElementPtr: {
          auto *GEP = cast<GetElementPtrInst>(PtrI)->clone();
          IIRB.IRB.Insert(GEP);
          auto *PtrOp = GEP->getOperand(0);
          ReplaceMap[{GEP, 0}] = PtrOp;
          Worklist.push_back(PtrOp);
          MPtr = GEP;
          errs() << " MPTR -> " << *GEP << "\n";
          continue;
        }
        default:
          // TODO: more inst and const expr
          break;
        }
      } else {
        IIRB.IRB.SetInsertPoint(BestIP);
        ensureDbgLoc(IIRB.IRB);
        // TODO: Globals?
      }
#endif
      // Fallback to rt call.
      auto *BaseMPtr = getBaseMPtr(*Ptr, IIRB);
      auto *EncNo = getBasePointerEncodingNo(*Ptr, IIRB);
      auto *CI = IIRB.IRB.CreateCall(GetMPtrFC, {Ptr, BaseMPtr, EncNo});
      MPtr = CI;
    }

#if 0
    auto &DT = IIRB.DTGetter(*Fn);
    for (auto &It : ReplaceMap) {
      It.first.first->setOperand(It.first.second, V2M.lookup({It.second, Fn}));
      // Other stuff moves, so we need to move this too.
      IIRB.hoistInstructionsAndAdjustIP(*It.first.first, BestIP, DT);
    }
#endif
    return V2M.lookup({&VPtr, Fn});
  }

  DenseMap<std::pair<Value *, Function *>, Value *> V2M;

  LightSanImpl &LSI;

  FunctionCallee GetMPtrFC;
  FunctionCallee LVRFC;
  FunctionCallee LRAFC;

  AttributorInfoCache *AIC = nullptr;
};

struct LightSanImpl {
  LightSanImpl(Module &M, ModuleAnalysisManager &MAM)
      : M(M), MAM(MAM),
        FAM(MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager()),
        IConf(*this, M) {}

  bool instrument();

  bool shouldImplementAdapter(Function &Fn);
  bool shouldInstrumentCall(CallInst &CI, InstrumentorIRBuilderTy &IIRB);
  bool shouldInstrumentLoad(LoadInst &LI, InstrumentorIRBuilderTy &IIRB);
  bool shouldInstrumentStore(StoreInst &SI, InstrumentorIRBuilderTy &IIRB);

private:
  bool updateSizesAfterPotentialFree();
  bool hoistLoopLoads(Loop &L, LoopInfo &LI, DominatorTree &DT);
  void foreachRTCaller(StringRef Name, function_ref<void(CallInst &)> CB);

  SmallVector<Function *> FuncsForWeakAdapters;
  SmallVector<Function *> FuncsForAliasAdapters;
  bool createWeakAdapters();
  bool createAliasAdapters();

  Module &M;
  ModuleAnalysisManager &MAM;
  FunctionAnalysisManager &FAM;
  LightSanInstrumentationConfig IConf;
  const DataLayout &DL = M.getDataLayout();
};

bool LightSanImpl::shouldInstrumentLoad(LoadInst &LI,
                                        InstrumentorIRBuilderTy &IIRB) {
  return true;
}

bool LightSanImpl::shouldInstrumentStore(StoreInst &SI,
                                         InstrumentorIRBuilderTy &IIRB) {
  return true;
}

bool LightSanImpl::shouldImplementAdapter(Function &Fn) {
  return !Fn.isVarArg() && !Fn.isIntrinsic() && !Fn.hasLocalLinkage();
}

bool LightSanImpl::shouldInstrumentCall(CallInst &CI,
                                        InstrumentorIRBuilderTy &IIRB) {
#if 0
  if (!CI.hasFnAttr(Attribute::NoFree)) {
    IConf.PotentiallyFreeCalls.push_back(&CI);
    auto &TLI = IIRB.TLIGetter(*CI.getFunction());
    if (auto *FreedPtr = getFreedOperand(&CI, &TLI)) {
      auto FreeFC = M.getOrInsertFunction(
          IConf.getRTName("", "free_object"),
          FunctionType::get(IIRB.VoidTy, {IIRB.PtrTy}, false));
      IIRB.IRB.CreateCall(FreeFC, {FreedPtr});
    }
  }
#endif

  Function *CalledFn = CI.getCalledFunction();
  if (!CalledFn)
    return true;
  FunctionType *CalledFnTy = CalledFn->getFunctionType();
  if (none_of(CalledFnTy->params(),
              [&](Type *ArgTy) { return ArgTy->isPtrOrPtrVectorTy(); }))
    return false;
  if (auto *II = dyn_cast<IntrinsicInst>(&CI)) {
    //    if (!II->mayHaveSideEffects() && !II->mayReadFromMemory())
    //      return false;
    if (II->isAssumeLikeIntrinsic())
      return false;
    if (auto *MI = dyn_cast<AnyMemIntrinsic>(II)) {
      Value *Src = nullptr;
      Value *Dst = nullptr;
      Value *Length = nullptr;
      Dst = MI->getRawDest();
      Length = MI->getLength();
      if (auto *MT = dyn_cast<AnyMemTransferInst>(II))
        Src = MT->getRawSource();

      for (auto *VPtr : {Src, Dst})
        if (VPtr) {
          IIRB.IRB.CreateCall(IConf.LRAFC,
                              {VPtr, IConf.getMPtr(*VPtr, IIRB), Length,
                               IConf.getBasePointerObjectSize(*VPtr, IIRB),
                               IConf.getBasePointerEncodingNo(*VPtr, IIRB),
                               IIRB.IRB.getInt8(1)});
        }
    }
    return true;
  }
  if (!CalledFn->isDeclaration())
    return false;
  if (CalledFn->getName().starts_with(LightSanRuntimePrefix))
    return false;
  if (CalledFn->isVarArg())
    return true;

  LibFunc TheLibFunc;
  auto &TLI = IIRB.TLIGetter(*CalledFn);
  // Known library functions are not wrapped (for now).
  if (!(TLI.getLibFunc(*CalledFn, TheLibFunc) && TLI.has(TheLibFunc))) {
    // Rest of cases should be instrumented within adapters.
    if (!CI.getFunction()->getName().starts_with(AdapterPrefix))
      return false;
  }
  return true;
#if 0
  switch (TheLibFunc) {
  case LibFunc_ZdaPv:
  case LibFunc_ZdaPvRKSt9nothrow_t:
  case LibFunc_ZdaPvSt11align_val_t:
  case LibFunc_ZdaPvSt11align_val_tRKSt9nothrow_t:
  case LibFunc_ZdaPvj:
  case LibFunc_ZdaPvjSt11align_val_t:
  case LibFunc_ZdaPvm:
  case LibFunc_ZdaPvmSt11align_val_t:
  case LibFunc_ZdlPv:
  case LibFunc_ZdlPvRKSt9nothrow_t:
  case LibFunc_ZdlPvSt11align_val_t:
  case LibFunc_ZdlPvSt11align_val_tRKSt9nothrow_t:
  case LibFunc_ZdlPvj:
  case LibFunc_ZdlPvjSt11align_val_t:
  case LibFunc_ZdlPvm:
  case LibFunc_ZdlPvmSt11align_val_t:
    // TODO: Deletes
    break;
  case LibFunc_Znaj:
  case LibFunc_ZnajRKSt9nothrow_t:
  case LibFunc_ZnajSt11align_val_t:
  case LibFunc_ZnajSt11align_val_tRKSt9nothrow_t:
  case LibFunc_Znam:
  case LibFunc_Znam12__hot_cold_t:
  case LibFunc_ZnamRKSt9nothrow_t:
  case LibFunc_ZnamRKSt9nothrow_t12__hot_cold_t:
  case LibFunc_ZnamSt11align_val_t:
  case LibFunc_ZnamSt11align_val_t12__hot_cold_t:
  case LibFunc_ZnamSt11align_val_tRKSt9nothrow_t:
  case LibFunc_ZnamSt11align_val_tRKSt9nothrow_t12__hot_cold_t:
  case LibFunc_Znwj:
  case LibFunc_ZnwjRKSt9nothrow_t:
  case LibFunc_ZnwjSt11align_val_t:
  case LibFunc_ZnwjSt11align_val_tRKSt9nothrow_t:
  case LibFunc_Znwm:
  case LibFunc_Znwm12__hot_cold_t:
  case LibFunc_ZnwmRKSt9nothrow_t:
  case LibFunc_ZnwmRKSt9nothrow_t12__hot_cold_t:
  case LibFunc_ZnwmSt11align_val_t:
  case LibFunc_ZnwmSt11align_val_t12__hot_cold_t:
  case LibFunc_ZnwmSt11align_val_tRKSt9nothrow_t:
  case LibFunc_ZnwmSt11align_val_tRKSt9nothrow_t12__hot_cold_t:
  case LibFunc_size_returning_new:
  case LibFunc_size_returning_new_hot_cold:
  case LibFunc_size_returning_new_aligned:
  case LibFunc_size_returning_new_aligned_hot_cold:
    // TODO: Allocate
    break;
  case LibFunc_atomic_load:
  case LibFunc_atomic_store:
    // TODO: Mem access
    break;
  case LibFunc_dunder_isoc99_scanf:
  case LibFunc_dunder_isoc99_sscanf:
    // TODO: Mem access
    break;
  case LibFunc___kmpc_alloc_shared:
  case LibFunc___kmpc_free_shared:
    // TODO: allocate
    break;
  case LibFunc_memccpy_chk:
  case LibFunc_memcpy_chk:
  case LibFunc_memmove_chk:
  case LibFunc_mempcpy_chk:
  case LibFunc_memset_chk:
    // TODO: Mem access
    break;
  case LibFunc_sincospi_stret:
  case LibFunc_sincospif_stret:
    // TODO: Mem access
    break;
  case LibFunc_small_fprintf:
  case LibFunc_small_printf:
  case LibFunc_small_sprintf:
  case LibFunc_snprintf_chk:
  case LibFunc_sprintf_chk:
  case LibFunc_stpcpy_chk:
  case LibFunc_stpncpy_chk:
  case LibFunc_strcat_chk:
  case LibFunc_strcpy_chk:
  case LibFunc_dunder_strdup:
  case LibFunc_strlcat_chk:
  case LibFunc_strlcpy_chk:
  case LibFunc_strlen_chk:
  case LibFunc_strncat_chk:
  case LibFunc_strncpy_chk:
  case LibFunc_dunder_strndup:
  case LibFunc_dunder_strtok_r:
  case LibFunc_vsnprintf_chk:
  case LibFunc_vsprintf_chk:
  case LibFunc_access:
    // TODO: Mem access
    break;
  case LibFunc_aligned_alloc:
    // TODO: Allocate
    break;
  case LibFunc_bcmp:
  case LibFunc_bcopy:
  case LibFunc_bzero:
    // TODO: Mem access
    break;
  case LibFunc_calloc:
    // TODO: Allocate
    break;
  case LibFunc_fclose:
    // TODO: Free
    break;
  case LibFunc_fdopen:
  case LibFunc_fopen64:
  case LibFunc_fopen:
    // TODO: Allocate
    break;
  case LibFunc_fgetc:
  case LibFunc_fgetc_unlocked:
  case LibFunc_fgetpos:
  case LibFunc_fgets:
  case LibFunc_fgets_unlocked:
  case LibFunc_fileno:
  case LibFunc_fiprintf:
  case LibFunc_fprintf:
  case LibFunc_fputc:
  case LibFunc_fputc_unlocked:
  case LibFunc_fputs:
  case LibFunc_fputs_unlocked:
  case LibFunc_fread:
  case LibFunc_fread_unlocked:
    // TODO: Mem access
    break;
  case LibFunc_free:
    // TODO: free
  case LibFunc_fscanf:
  case LibFunc_fseek:
  case LibFunc_fseeko:
  case LibFunc_fseeko64:
  case LibFunc_fsetpos:
  case LibFunc_fstat:
  case LibFunc_fstat64:
  case LibFunc_fstatvfs:
  case LibFunc_fstatvfs64:
  case LibFunc_ftell:
  case LibFunc_ftello:
  case LibFunc_ftello64:
  case LibFunc_ftrylockfile:
  case LibFunc_funlockfile:
  case LibFunc_fwrite:
  case LibFunc_fwrite_unlocked:
  case LibFunc_getc:
  case LibFunc_getc_unlocked:
  case LibFunc_getchar:
  case LibFunc_getchar_unlocked:
  case LibFunc_getenv:
  case LibFunc_getitimer:
  case LibFunc_getlogin_r:
  case LibFunc_getpwnam:
  case LibFunc_gets:
  case LibFunc_gettimeofday:
  case LibFunc_iprintf:
  case LibFunc_lstat:
  case LibFunc_lstat64:
    // TODO: Mem access
    break;
  case LibFunc_malloc:
    // TODO: allocate
    break;
  case LibFunc_memalign:
  case LibFunc_posix_memalign:
    break;
  case LibFunc_memccpy:
  case LibFunc_memchr:
  case LibFunc_memcmp:
  case LibFunc_memcpy:
  case LibFunc_memmove:
  case LibFunc_mempcpy:
  case LibFunc_memrchr:
  case LibFunc_memset:
  case LibFunc_memset_pattern16:
  case LibFunc_memset_pattern4:
  case LibFunc_memset_pattern8:
    // TODO: Mem access
    break;
  case LibFunc_open:
  case LibFunc_open64:
  case LibFunc_opendir:
  case LibFunc_popen:
    // TODO: Allocate
    break;
  case LibFunc_pclose:
  case LibFunc_perror:
  case LibFunc_pread:
  case LibFunc_printf:
  case LibFunc_putc:
  case LibFunc_putc_unlocked:
  case LibFunc_putchar:
  case LibFunc_putchar_unlocked:
  case LibFunc_puts:
  case LibFunc_pwrite:
    // TODO:
    break;
  case LibFunc_qsort:
  case LibFunc_read:
  case LibFunc_readlink:
    // TODO: mem access
    break;
  case LibFunc_realloc:
  case LibFunc_reallocf:
  case LibFunc_reallocarray:
    // TODO: free + allocate
    break;
  case LibFunc_realpath:
  case LibFunc_remquo:
  case LibFunc_remquof:
  case LibFunc_remquol:
  case LibFunc_remove:
  case LibFunc_rename:
  case LibFunc_rewind:
  case LibFunc_rmdir:
  case LibFunc_scanf:
  case LibFunc_setbuf:
  case LibFunc_setitimer:
  case LibFunc_setvbuf:
  case LibFunc_sincos:
  case LibFunc_sincosf:
  case LibFunc_sincosl:
  case LibFunc_siprintf:
  case LibFunc_snprintf:
  case LibFunc_sprintf:
  case LibFunc_sscanf:
  case LibFunc_stat:
  case LibFunc_stat64:
  case LibFunc_statvfs:
  case LibFunc_statvfs64:
  case LibFunc_stpcpy:
  case LibFunc_stpncpy:
  case LibFunc_strcasecmp:
  case LibFunc_strcat:
  case LibFunc_strchr:
  case LibFunc_strcmp:
  case LibFunc_strcoll:
  case LibFunc_strcpy:
  case LibFunc_strcspn:
  case LibFunc_strdup:
  case LibFunc_strlcat:
  case LibFunc_strlcpy:
  case LibFunc_strlen:
  case LibFunc_strncasecmp:
  case LibFunc_strncat:
  case LibFunc_strncmp:
  case LibFunc_strncpy:
  case LibFunc_strndup:
  case LibFunc_strnlen:
  case LibFunc_strpbrk:
  case LibFunc_strrchr:
  case LibFunc_strspn:
  case LibFunc_strstr:
  case LibFunc_strtod:
  case LibFunc_strtof:
  case LibFunc_strtok:
  case LibFunc_strtok_r:
  case LibFunc_strtol:
  case LibFunc_strtold:
  case LibFunc_strtoll:
  case LibFunc_strtoul:
  case LibFunc_strtoull:
  case LibFunc_strxfrm:
  case LibFunc_system:
  case LibFunc_times:
  case LibFunc_tmpfile:
  case LibFunc_tmpfile64:
  case LibFunc_uname:
  case LibFunc_ungetc:
  case LibFunc_unlink:
  case LibFunc_unsetenv:
  case LibFunc_utime:
  case LibFunc_utimes:
    // TODO: mem access
    break;
  case LibFunc_valloc:
  case LibFunc_vec_calloc:
  case LibFunc_vec_free:
  case LibFunc_vec_malloc:
  case LibFunc_vec_realloc:
    // TODO: allocate
    break;
  case LibFunc_vfprintf:
  case LibFunc_vfscanf:
  case LibFunc_vprintf:
  case LibFunc_vscanf:
  case LibFunc_vsnprintf:
  case LibFunc_vsprintf:
  case LibFunc_vsscanf:
  case LibFunc_wcslen:
  case LibFunc_write:
    // TODO: mem access
    break;
  default:
    break;
  }
  return true;
#endif
}

bool LightSanImpl::hoistLoopLoads(Loop &L, LoopInfo &LI, DominatorTree &DT) {
  bool Changed = false;

  auto *LatchBB = L.getLoopLatch();
  auto *PreHeaderBB = L.getLoopPreheader();
  if (!LatchBB || !PreHeaderBB)
    return Changed;
  auto *HeaderBB = L.getHeader();

  // auto *Int64Ty = IntegerType::getInt64Ty(M.getContext());
  // auto *PtrTy = PointerType::get(M.getContext(), 0);

  SmallVector<std::tuple<LoadInst *, Value *, APInt>> Loads;
  for (auto *BB : L.blocks())
    for (auto &I : *BB)
      if (auto *LoadI = dyn_cast<LoadInst>(&I)) {
        auto *Ty = LoadI->getType();
        if (!Ty->isPointerTy())
          continue;
        auto *Ptr = LoadI->getPointerOperand();
        APInt Offset(
            DL.getIndexSizeInBits(Ptr->getType()->getPointerAddressSpace()), 0);
        auto *UnderlyingPtr = Ptr->stripAndAccumulateConstantOffsets(
            DL, Offset, /*AllowNonInbounds=*/true,
            /* AllowInvariant */ true);
        if (!isa<Instruction>(UnderlyingPtr)) {
          if (isa<Argument>(UnderlyingPtr))
            Loads.push_back({LoadI, UnderlyingPtr, Offset});
          if (auto *GV = dyn_cast<GlobalVariable>(UnderlyingPtr))
            if (GV->hasInitializer())
              Loads.push_back({LoadI, UnderlyingPtr, Offset});
        } else if (auto *PtrI = dyn_cast<Instruction>(UnderlyingPtr)) {
          if (!L.contains(PtrI))
            Loads.push_back({LoadI, UnderlyingPtr, Offset});
        }
      }

  auto &Ctx = HeaderBB->getContext();
  auto *VoidTy = Type::getVoidTy(Ctx);
  auto *Int1Ty = IntegerType::getInt1Ty(Ctx);
  auto *Int8Ty = IntegerType::getInt8Ty(Ctx);
  auto *Int64Ty = IntegerType::getInt64Ty(Ctx);
  auto *PtrTy = PointerType::get(Ctx, 0);
  auto CheckFC =
      M.getOrInsertFunction(IConf.getRTName("", "check_ptr_load"),
                            FunctionType::get(Int8Ty, {PtrTy, Int64Ty}, false));
  auto CheckNZFC =
      M.getOrInsertFunction(IConf.getRTName("", "check_non_zero"),
                            FunctionType::get(VoidTy, {PtrTy}, false));

  DomTreeUpdater DTU(DT, DomTreeUpdater::UpdateStrategy::Eager);
  auto IP = PreHeaderBB->getTerminator()->getIterator();
  while (!Loads.empty()) {
    auto [LoadI, Ptr, Offset] = Loads.pop_back_val();
    auto *Ty = LoadI->getType();

    Instruction *CheckRetVal =
        CallInst::Create(CheckFC, {Ptr, ConstantInt::get(Int64Ty, Offset)},
                         LoadI->getName() + ".check", IP);
    if (auto &DL = LoadI->getStableDebugLoc())
      CheckRetVal->setDebugLoc(DL);
    else if (auto *SP = LoadI->getFunction()->getSubprogram())
      CheckRetVal->setDebugLoc(DILocation::get(Ctx, 0, 0, SP));
    auto *Cond = new TruncInst(CheckRetVal, Int1Ty, "", IP);
    auto *NewTI = SplitBlockAndInsertIfThen(
        Cond, IP, /*Unreachable*/ false, /*BranchWeights=*/nullptr, &DTU, &LI);
    auto *LoadCloneI = LoadI->clone();
    LoadCloneI->insertBefore(NewTI->getIterator());
    auto *EarlyPHI = PHINode::Create(Ty, 2, ".earlyPHI", IP);
    LoadI->replaceAllUsesWith(EarlyPHI);
    EarlyPHI->addIncoming(LoadCloneI, NewTI->getParent());
    EarlyPHI->addIncoming(ConstantInt::getNullValue(Ty), Cond->getParent());

    // TODO: Verify not zero at LoadI position
    Instruction *CheckNZ =
        CallInst::Create(CheckNZFC, {EarlyPHI}, "", LoadI->getIterator());
    if (auto &DL = LoadI->getStableDebugLoc())
      CheckNZ->setDebugLoc(DL);
    else if (auto *SP = LoadI->getFunction()->getSubprogram())
      CheckNZ->setDebugLoc(DILocation::get(Ctx, 0, 0, SP));
    LoadI->eraseFromParent();

    // auto LoadIP = LoadI->getIterator();
    // auto *NotCond = new ICmpInst(LoadIP, ICmpInst::ICMP_EQ, Cond,
    //                              ConstantInt::getFalse(Ctx));
    // auto *ReloadTI =
    //     SplitBlockAndInsertIfThen(NotCond, LoadIP, /*Unreachable*/ false,
    //                               /*BranchWeights=*/nullptr, &DTU, &LI);
    // auto *LatePHI = PHINode::Create(Ty, 2, ".latePHI", LoadIP);
    // LoadI->replaceAllUsesWith(LatePHI);
    // LatePHI->addIncoming(LoadI, ReloadTI->getParent());
    // LatePHI->addIncoming(EarlyPHI, NotCond->getParent());

    // LoadI->moveBefore(ReloadTI->getIterator());
  }

  for (auto *ChildL : L)
    Changed |= hoistLoopLoads(*ChildL, LI, DT);
  return Changed;
}

static bool collectAttributorInfo(Attributor &A, Module &M,
                                  AttributorInfoCache &Cache) {
  const DataLayout &DL = M.getDataLayout();

  // A list of pairs of objects and their size
  // TODO: Use weak value handles
  SmallVector<std::tuple<Value *, uint64_t, const AAPointerInfo *>> WorkList;

  for (GlobalVariable &GV : M.globals()) {
    if (GV.getValueType()->isSized() && !GV.hasExternalWeakLinkage() &&
        GV.hasInitializer() && !GV.isInterposable()) {
      auto *AAPI = A.getOrCreateAAFor<AAPointerInfo>(IRPosition::value(GV),
                                                     /*QueryingAA=*/nullptr,
                                                     DepClassTy::REQUIRED);
      WorkList.emplace_back(&GV, DL.getTypeAllocSize(GV.getValueType()), AAPI);
    }
  }

  auto HandleAllocaInst = [&](AllocaInst *AI) {
    auto Size = AI->getAllocationSize(DL);
    // TODO: Probably we can handle dynamic alloca?
    if (!Size || !Size->isFixed())
      return;
    auto SizeV = Size->getFixedValue();
    auto *AAPI = A.getOrCreateAAFor<AAPointerInfo>(IRPosition::value(*AI),
                                                   /*QueryingAA=*/nullptr,
                                                   DepClassTy::REQUIRED);
    WorkList.emplace_back(AI, SizeV, AAPI);
  };

  auto HandleMallocLikeFn = [&](CallBase *CB, AllocationCallInfo &ACI) {
    uint64_t Size = 1;
    if (ACI.SizeLHSArgNo >= 0) {
      auto *SizeCI = dyn_cast<ConstantInt>(CB->getArgOperand(ACI.SizeLHSArgNo));
      if (!SizeCI)
        return;
      Size *= SizeCI->getSExtValue();
    }
    if (ACI.SizeRHSArgNo >= 0) {
      auto *SizeCI = dyn_cast<ConstantInt>(CB->getArgOperand(ACI.SizeRHSArgNo));
      if (!SizeCI)
        return;
      Size *= SizeCI->getSExtValue();
    }
    auto *AAPI = A.getOrCreateAAFor<AAPointerInfo>(IRPosition::value(*CB),
                                                   /*QueryingAA=*/nullptr,
                                                   DepClassTy::REQUIRED);
    WorkList.emplace_back(CB, Size, AAPI);
  };
  auto HandleCallBase = [&](CallBase *CI) {
    for (auto &U : CI->args()) {
      if (U->getType()->isPtrOrPtrVectorTy()) {
        A.getOrCreateAAFor<AAUnderlyingObjects>(IRPosition::value(*U));
      }
    }
  };

  for (auto &GV : M.globals()) {
    uint64_t Size = 0;
    if (!getObjectSize(&GV, Size, DL, /*TLI=*/nullptr))
      continue;
    auto *AAPI = A.getOrCreateAAFor<AAPointerInfo>(IRPosition::value(GV),
                                                   /*QueryingAA=*/nullptr,
                                                   DepClassTy::REQUIRED);
    WorkList.emplace_back(&GV, Size, AAPI);
  }
  for (Function &F : M) {
    if (F.isIntrinsic())
      continue;

    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (auto *Ptr = AA::getPointerOperand(&I, /*AllowVolatile*/ true)) {
          A.getOrCreateAAFor<AAUnderlyingObjects>(IRPosition::value(*Ptr));
        } else if (auto *AI = dyn_cast<AllocaInst>(&I)) {
          HandleAllocaInst(AI);
        } else if (auto *CB = dyn_cast<CallBase>(&I)) {
          if (CB->getCalledFunction()) {
            const auto *TLI = A.getInfoCache().getTargetLibraryInfoForFunction(
                *CB->getCalledFunction());
            assert(TLI);
            if (auto ACI = getAllocationCallInfo(CB, TLI)) {
              HandleMallocLikeFn(CB, *ACI);
              continue;
            }
          }
          HandleCallBase(CB);
        }
      }
    }
  }

  ChangeStatus Changed = A.run();

  AA::AccessRangeTy Range(AA::RangeTy::getUnknown(),
                          AA::RangeTy::getUnknownSize());
  AA::AccessRangeListTy RangeList(Range);

  for (auto [Obj, Size, AAPI] : WorkList) {
    Cache.insertKnownObject(Obj, Size);

    if (!AAPI || !AAPI->getState().isValidState())
      continue;

    int64_t ObjSize = Size;
    auto *ObjPtr = Obj;
    bool AnyAccessIsProblematic = false;
    auto CheckAccessCB = [&](const AAPointerInfo::Access &Acc, bool) -> bool {
      for (auto &R : Acc) {
        if (R.isUnknown() || !R.isSizeKnown() ||
            Acc.getAccessSize() == AA::RangeTy::getUnknownSize())
          return AnyAccessIsProblematic = true;
        int64_t AccOffset = R.getOffset();
        int64_t AccSize = R.getSize();
        if (AccOffset < 0)
          return AnyAccessIsProblematic = true;
        // TODO: Consider wrap here?
        if (AccOffset + AccSize + Acc.getAccessSize() > ObjSize)
          return AnyAccessIsProblematic = true;
      }
      if (!Cache.getUnderlyingObject(AA::getPointerOperand(
              Acc.getRemoteInst(), /*AllowVolatile*/ true)))
        return AnyAccessIsProblematic = true;

      Cache.insertSafeAccess(Acc.getRemoteInst(), ObjPtr);
      return true;
    };

    AAPI->forallInterferingAccesses(RangeList, CheckAccessCB,
                                    AAPointerInfo::AK_NONE,
                                    AAPointerInfo::AK_ANY);
    if (!AAPI->hasPotentiallyAliasingPointers())
      Cache.insertNonEscapingObj(Obj);

    if (AAPI->hasPotentiallyAliasingPointers() || AnyAccessIsProblematic)
      continue;

    Cache.insertSafeObject(Obj);
  }

  return Changed == ChangeStatus::CHANGED;
}

void LightSanImpl::foreachRTCaller(StringRef Name,
                                   function_ref<void(CallInst &)> CB) {
  auto *FC = M.getFunction(Name);
  if (!FC)
    return;
  for (auto *U : make_early_inc_range(FC->users())) {
    auto *CI = cast<CallInst>(U);
    CB(*CI);
  }
}

bool LightSanImpl::createWeakAdapters() {
  for (Function &Fn : M) {
    if (!shouldImplementAdapter(Fn))
      continue;

    if (Fn.isDeclaration()) {
      auto &TLI = FAM.getResult<TargetLibraryAnalysis>(Fn);
      LibFunc TheLibFunc;
      if (TLI.getLibFunc(Fn, TheLibFunc) && TLI.has(TheLibFunc))
        continue;
      if (Fn.getName().starts_with(LightSanRuntimePrefix))
        continue;
      FuncsForWeakAdapters.push_back(&Fn);
    } else if (!Fn.hasLocalLinkage()) {
      FuncsForAliasAdapters.push_back(&Fn);
    }
  }

  if (FuncsForWeakAdapters.empty())
    return false;

  for (Function *Fn : FuncsForWeakAdapters) {
    std::string FnName = Fn->getName().str();
    std::string AdapterFnName = AdapterPrefix + FnName;

    auto *AdapterFn = Function::Create(
        Fn->getFunctionType(), Function::WeakAnyLinkage, AdapterFnName, M);
    AdapterFn->copyAttributesFrom(Fn);
    AdapterFn->setCallingConv(Fn->getCallingConv());

    Fn->replaceAllUsesWith(AdapterFn);

    SmallVector<Value *> AdapterArgs;
    for (auto &Arg : AdapterFn->args())
      AdapterArgs.push_back(&Arg);

    auto FC = M.getOrInsertFunction(FnName, Fn->getFunctionType());

    auto *EntryBB = BasicBlock::Create(M.getContext(), "entry", AdapterFn);
    IRBuilder<> IRB(EntryBB, EntryBB->getFirstNonPHIOrDbgOrAlloca());
    ensureDbgLoc(IRB);
    auto *CI = IRB.CreateCall(FC, AdapterArgs);
    if (CI->getType()->isVoidTy())
      IRB.CreateRetVoid();
    else
      IRB.CreateRet(CI);
  }
  return true;
}

bool LightSanImpl::createAliasAdapters() {
  if (FuncsForAliasAdapters.empty())
    return false;

  for (Function *Fn : FuncsForAliasAdapters)
    GlobalAlias::create(Fn->getLinkage(), AdapterPrefix + Fn->getName(), Fn);

  return true;
}

bool LightSanImpl::instrument() {
  bool Changed = false;

  // Create weak function adapters for external functions.
  Changed |= createWeakAdapters();

#if 0
  for (auto &Fn : M) {
    if (Fn.isDeclaration())
      continue;
    auto &LI = FAM.getResult<LoopAnalysis>(Fn);
    auto &DT = FAM.getResult<DominatorTreeAnalysis>(Fn);
    for (auto *L : LI)
      Changed |= hoistLoopLoads(*L, LI, DT);
  }
#endif

  // Set up attributor
  FunctionAnalysisManager &FAM =
      MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

  CallGraphUpdater CGUpdater;
  BumpPtrAllocator Allocator;
  AnalysisGetter AG(FAM);

  InformationCache InfoCache(M, AG, Allocator, /*CGSCC=*/nullptr);

  // FIXME: Do we need more stuff here or just allow everything?
  DenseSet<const char *> Allowed(
      {&AAPointerInfo::ID, &AAUnderlyingObjects::ID, &AAPotentialValues::ID,
       &AAPotentialConstantValues::ID, &AAValueConstantRange::ID,
       &AAInstanceInfo::ID});

  AttributorConfig AC(CGUpdater);
  AC.Allowed = &Allowed;
  AC.DefaultInitializeLiveInternals = false;
  AC.IsModulePass = true;
  AC.RewriteSignatures = false;
  AC.MaxFixpointIterations = 32;
  AC.PassName = DEBUG_TYPE;

  SetVector<Function *> Functions;
  for (Function &F : M) {
    if (!F.isIntrinsic())
      Functions.insert(&F);
  }

  Attributor A(Functions, InfoCache, AC);
  AttributorInfoCache Cache(A);
  IConf.AIC = &Cache;
  IConf.InlineRuntimeEagerly->setBool(false);

  Changed |= collectAttributorInfo(A, M, Cache);

  InstrumentorPass IP(&IConf);
  auto PA = IP.run(M, MAM);
  if (!PA.areAllPreserved())
    Changed = true;

  // Run this again as the calles might have changed.
  IConf.initializeFunctionCallees(M);

  auto &Ctx = M.getContext();
  auto *Int8Ty = IntegerType::getInt8Ty(Ctx);
  auto *Int64Ty = IntegerType::getInt64Ty(Ctx);

  DenseMap<Function *,
           DenseMap<std::pair<BasicBlock *, Value *>, SmallVector<CallInst *>>>
      MergeMap;

#if 1
  auto CheckForRequiredRanged = [&](StringRef Name) {
    auto *FC = M.getFunction(IConf.getRTName("pre_", Name));
    if (!FC)
      return;
    for (auto *U : FC->users()) {
      auto *CI = cast<CallInst>(U);
      auto *BB = CI->getParent();
      auto *Fn = BB->getParent();

      auto *BPI = CI->getArgOperand(1);
      auto &DT = FAM.getResult<DominatorTreeAnalysis>(*Fn);

      auto *LVRI = CI->getArgOperand(2);
      bool HasRangeInfo = !isa<ConstantPointerNull>(LVRI);
      if (!HasRangeInfo) {
        MergeMap[Fn][{BB, BPI}].push_back(CI);
        continue;
      }
      auto &LI = FAM.getResult<LoopAnalysis>(*Fn);
      auto *L = LI.getLoopFor(BB);
      if (!L)
        continue;
      auto *PreHeaderBB = L->getLoopPreheader();
      if (!PreHeaderBB || !PreHeaderBB->getSingleSuccessor())
        continue;
      auto &PDT = FAM.getResult<PostDominatorTreeAnalysis>(*Fn);
      auto *LVRICI = cast<CallInst>(LVRI);
      // Try to sink the range check to make sure it can be enforced instead of
      // the access check.
      if (!PDT.dominates(CI, LVRICI)) {
        BasicBlock *BestBB = BB;
        do {
          auto *DBB = DT.getNode(BestBB)->getIDom()->getBlock();
          if (!PDT.dominates(BB, DBB))
            break;
          BestBB = DBB;
        } while (1);
        if (BestBB == BB)
          continue;
        auto *CloneLVRICI = cast<CallInst>(LVRICI->clone());
        CloneLVRICI->insertBefore(BestBB->getTerminator()->getIterator());
        LVRICI->replaceUsesWithIf(
            CloneLVRICI, [&](Use &U) { return DT.dominates(BestBB, U); });
        if (LVRICI->use_empty())
          LVRICI->eraseFromParent();
        LVRICI = CloneLVRICI;
      }
      // TODO: we need to use the must-be-executed stuff in
      // llvm/include/llvm/Analysis/MustExecute.h to avoid weird exits
      auto *ExitingBB = L->getExitingBlock();
      if (!ExitingBB || !(BB == ExitingBB || DT.dominates(BB, ExitingBB)))
        continue;
      auto MaxOffset =
          cast<ConstantInt>(LVRICI->getArgOperand(2))->getSExtValue();
      auto IsExecuted =
          cast<ConstantInt>(LVRICI->getArgOperand(5))->getSExtValue();
      if (!IsExecuted) {
        auto Offset = cast<ConstantInt>(CI->getArgOperand(3))->getSExtValue();
        if (MaxOffset > Offset)
          continue;
        LVRICI->setArgOperand(
            5, ConstantInt::get(Type::getInt8Ty(M.getContext()), 1));
      }
      CI->setArgOperand(7,
                        ConstantInt::get(Type::getInt8Ty(M.getContext()), 1));
    }
  };
  CheckForRequiredRanged("load");
  CheckForRequiredRanged("store");
#endif

#if 1

  auto *TrueInt8 = ConstantInt::get(Int8Ty, 1, /*IsSigned*/ false);
  for (const auto &It : MergeMap) {
    auto *Fn = It.first;
    auto &DT = FAM.getResult<DominatorTreeAnalysis>(*Fn);

    for (auto ItIt : It.second) {
      auto &Calls = ItIt.second;
      if (Calls.size() < 2)
        continue;
      SmallVector<Value *> MinPtrs, MaxPtrs;
      SmallVector<CallInst *> CheckedCalls;
      CallInst *FirstCI = nullptr;
      for (auto *CI : Calls)
        if (!FirstCI || DT.dominates(CI, FirstCI))
          FirstCI = CI;

      IRBuilder<> IRB(FirstCI);
      ensureDbgLoc(IRB);

      for (auto *CI : Calls) {
        auto *Ptr = CI->getArgOperand(0);
        if (auto *PtrI = dyn_cast<Instruction>(Ptr)) {
          InstrumentorIRBuilderTy::hoistInstructionsAndAdjustIP(
              *PtrI, FirstCI->getIterator(), DT);
          if (!DT.dominates(PtrI, FirstCI))
            continue;
        }
        CheckedCalls.push_back(CI);
        MinPtrs.push_back(Ptr);
        MaxPtrs.push_back(IRB.CreatePtrAdd(Ptr, CI->getArgOperand(3), "",
                                           GEPNoWrapFlags::inBounds()));
      }
      if (MinPtrs.size() < 2)
        continue;

      Value *MinV = MinPtrs.pop_back_val();
      Value *MaxV = MaxPtrs.pop_back_val();
      auto *CheckedCI = CheckedCalls.pop_back_val();
      CheckedCI->setArgOperand(7, TrueInt8);
      while (!MinPtrs.empty()) {
        auto *MinVal = MinPtrs.pop_back_val();
        auto *MaxVal = MaxPtrs.pop_back_val();
        CheckedCI = CheckedCalls.pop_back_val();
        CheckedCI->setArgOperand(7, TrueInt8);
        auto *MinC = IRB.CreateICmpULT(MinVal, MinV);
        MinV = IRB.CreateSelect(MinC, MinVal, MinV);
        auto *MaxC = IRB.CreateICmpUGT(MaxVal, MaxV);
        MaxV = IRB.CreateSelect(MaxC, MaxVal, MaxV);
      }
      auto *ObjSize = FirstCI->getArgOperand(5);
      auto *EncNo = FirstCI->getArgOperand(6);

      IRB.CreateCall(
          IConf.LVRFC,
          {MinV, MaxV, ConstantInt::getNullValue(Int64Ty), ObjSize, EncNo});
      LLVM_DEBUG(errs() << "Use range access in BB for " << Calls.size()
                        << " checks\n");
    }
  }
#endif

#if 1
  auto UseMPtr = [&](StringRef Name) {
    auto *FC = M.getFunction(IConf.getRTName("pre_", Name));
    if (!FC)
      return;
    for (auto *U : make_early_inc_range(FC->users())) {
      auto *CI = cast<CallInst>(U);
      auto *VPtr = CI->getArgOperand(0);
      if (auto *PReg = IConf.AIC->getPreRegister(VPtr))
        VPtr = PReg;
      auto *WasCheckedCI = dyn_cast<ConstantInt>(CI->getArgOperand(7));
      if (!WasCheckedCI || !WasCheckedCI->isOne())
        continue;
      CI->replaceAllUsesWith(CI->getArgOperand(4));
      CI->eraseFromParent();
    }
  };
  UseMPtr("load");
  UseMPtr("store");
#endif

  SmallVector<std::tuple<Instruction *, uint32_t, Value *, Value *>> ReplVec;
  foreachRTCaller(IConf.getRTName("", "get_mptr"), [&](CallInst &CI) {
    auto &Fn = *CI.getFunction();
    auto *VPtr = CI.getArgOperand(0);
    auto *BaseMPtr = CI.getArgOperand(1);
    auto *EncNo = CI.getArgOperand(2);

    SmallVector<Value *> Worklist;
    Worklist.push_back(VPtr);

    while (!Worklist.empty()) {
      Value *Ptr = Worklist.pop_back_val();
      auto *&MPtr = IConf.V2M[{Ptr, &Fn}];
      if (MPtr && Ptr != VPtr)
        continue;
      if (auto *PtrI = dyn_cast<Instruction>(Ptr)) {
        switch (PtrI->getOpcode()) {
        case Instruction::GetElementPtr: {
          auto *GEP = cast<GetElementPtrInst>(PtrI)->clone();
          GEP->insertBefore(PtrI->getIterator());
          ReplVec.push_back({GEP, 0, BaseMPtr, EncNo});
          Worklist.push_back(GEP->getOperand(0));
          MPtr = GEP;
          if (Ptr == VPtr)
            CI.replaceAllUsesWith(GEP);
        }
        };
      }
    }
  });
  for (auto [I, ArgNo, BaseMPtr, EncNo] : ReplVec) {
    auto *Fn = I->getFunction();
    auto *VPtr = I->getOperand(ArgNo);
    auto *&MPtr = IConf.V2M[{VPtr, Fn}];
    auto &DT = FAM.getResult<DominatorTreeAnalysis>(*Fn);
    if (!MPtr) {
      BasicBlock::iterator IP =
          Fn->getEntryBlock().getFirstNonPHIOrDbgOrAlloca();
      for (auto *V : {VPtr, BaseMPtr, EncNo})
        if (auto *I = dyn_cast<Instruction>(V))
          if (!DT.dominates(I, IP))
            IP = *I->getInsertionPointAfterDef();
      IRBuilder<> IRB(IP->getParent(), IP);
      ensureDbgLoc(IRB);
      MPtr = IRB.CreateCall(IConf.GetMPtrFC, {VPtr, BaseMPtr, EncNo});
    }
    I->setOperand(ArgNo, MPtr);
  }
  foreachRTCaller(IConf.getRTName("", "get_mptr"), [&](CallInst &CI) {
    if (CI.use_empty()) {
      CI.eraseFromParent();
      return;
    }
    auto &Fn = *CI.getFunction();
    BasicBlock::iterator IP = Fn.getEntryBlock().getFirstNonPHIOrDbgOrAlloca();

    auto &DT = FAM.getResult<DominatorTreeAnalysis>(Fn);
    InstrumentorIRBuilderTy::hoistInstructionsAndAdjustIP(
        CI, IP, DT, /*ForceInitial=*/true);
  });

#if 0
  auto CheckForBasePtrInLoop = [&]() {
    auto *FC = M.getFunction(IConf.getRTName("post_", "base_pointer_info"));
    for (auto *U : FC->users()) {
      auto *CI = cast<CallInst>(U);
      auto *PHI = dyn_cast<PHINode>(CI->getArgOperand(0));
      if (!PHI)
        continue;
      auto *BB = CI->getParent();
      auto *Fn = BB->getParent();
      auto &LI = FAM.getResult<LoopAnalysis>(*Fn);
      auto *L = LI.getLoopFor(BB);
      if (!L || L->getHeader() != BB)
        continue;
      // TODO:check no free loop
      auto *PreHeaderBB = L->getLoopPreheader();
      auto *LatchBB = L->getLoopLatch();
      if (!PreHeaderBB | !LatchBB)
        continue;
      auto *InitialVal = PHI->getIncomingValueForBlock(PreHeaderBB);
      auto *LatchValI =
          dyn_cast<Instruction>(PHI->getIncomingValueForBlock(LatchBB));
      if (!LatchValI || !L->contains(LatchValI))
        continue;
      auto *InitCI = cast<CallInst>(CI->clone());
      InitCI->setArgOperand(0, InitialVal);
      auto PreheaderTI = PreHeaderBB->getTerminator()->getIterator();
      InitCI->insertBefore(PreheaderTI);
      auto *SizeLI = cast<LoadInst>(CI->getNextNode());
      auto *EncLI = cast<LoadInst>(SizeLI->getNextNode());
      auto *OffLI = cast<LoadInst>(EncLI->getNextNode());
      SizeLI->moveBefore(PreheaderTI);
      EncLI->moveBefore(PreheaderTI);
      OffLI->moveBefore(PreheaderTI);
      auto *MPtrPHI = PHINode::Create(CI->getType(), 2, "", PHI->getIterator());
      auto *EncPHI =
          PHINode::Create(EncLI->getType(), 2, "", PHI->getIterator());
      auto *SizePHI =
          PHINode::Create(SizeLI->getType(), 2, "", PHI->getIterator());
      auto *OffPHI =
          PHINode::Create(OffLI->getType(), 2, "", PHI->getIterator());
      CI->replaceAllUsesWith(MPtrPHI);
      SizeLI->replaceAllUsesWith(SizePHI);
      EncLI->replaceAllUsesWith(EncPHI);
      OffLI->replaceAllUsesWith(OffPHI);
      MPtrPHI->addIncoming(InitCI, PreHeaderBB);
      EncPHI->addIncoming(EncLI, PreHeaderBB);
      SizePHI->addIncoming(SizeLI, PreHeaderBB);
      OffPHI->addIncoming(OffLI, PreHeaderBB);
      CI->moveAfter(LatchValI);
      CI->setArgOperand(0, LatchValI);
      auto *NewSizeLI = SizeLI->clone();
      auto *NewEncLI = EncLI->clone();
      auto *NewOffLI = OffLI->clone();
      NewOffLI->insertAfter(CI);
      NewEncLI->insertAfter(CI);
      NewSizeLI->insertAfter(CI);
      MPtrPHI->addIncoming(CI, LatchBB);
      SizePHI->addIncoming(NewSizeLI, LatchBB);
      EncPHI->addIncoming(NewEncLI, LatchBB);
      OffPHI->addIncoming(NewOffLI, LatchBB);
    }
  };
  CheckForBasePtrInLoop();
#endif

  // Create strong function aliases for exported functions.
  Changed |= createAliasAdapters();

  return Changed;
}

LightSanInstrumentationConfig::LightSanInstrumentationConfig(LightSanImpl &Impl,
                                                             Module &M)
    : InstrumentationConfig(), LSI(Impl) {
  ReadConfig = false;
  RuntimePrefix->setString(LightSanRuntimePrefix);
  RuntimeStubsFile->setString("");
  initializeFunctionCallees(M);
}

void LightSanInstrumentationConfig::initializeFunctionCallees(Module &M) {
  LLVMContext &Ctx = M.getContext();
  Type *VoidTy = Type::getVoidTy(Ctx);
  Type *PtrTy = PointerType::get(Ctx, 0);
  Type *Int8Ty = IntegerType::getInt8Ty(Ctx);
  Type *Int64Ty = IntegerType::getInt64Ty(Ctx);

  LRAFC = M.getOrInsertFunction(
      getRTName("pre_", "ranged_access"),
      FunctionType::get(
          VoidTy, {PtrTy, PtrTy, Int64Ty, Int64Ty, Int8Ty, Int8Ty}, false));

  LVRFC = M.getOrInsertFunction(
      getRTName("post_", "loop_value_ptr_range"),
      FunctionType::get(VoidTy, {PtrTy, PtrTy, Int64Ty, Int64Ty, Int8Ty},
                        false));

  GetMPtrFC = M.getOrInsertFunction(
      getRTName("", "get_mptr"),
      FunctionType::get(PtrTy, {PtrTy, PtrTy, Int8Ty}, false));
}

struct ExtendedAllocaIO : public AllocaIO {
  ExtendedAllocaIO(bool IsPRE) : AllocaIO(IsPRE) {}
  virtual ~ExtendedAllocaIO() {};

  void init(InstrumentationConfig &IConf, InstrumentorIRBuilderTy &IIRB) {
    AllocaIO::ConfigTy AICConfig(/*Enable=*/false);
    AICConfig.set(AllocaIO::PassAddress);
    AICConfig.set(AllocaIO::ReplaceAddress);
    AICConfig.set(AllocaIO::PassSize);
    AllocaIO::init(IConf, IIRB.Ctx, &AICConfig);

    IRTArgs.push_back(IRTArg(IIRB.Int8Ty, "requires_temporal_check",
                             "Flag to indicate that the alloca might be "
                             "accessed after it was freed.",
                             IRTArg::NONE, getRequiresTemporalCheck));
  }

  static Value *getRequiresTemporalCheck(Value &V, Type &Ty,
                                         InstrumentationConfig &IConf,
                                         InstrumentorIRBuilderTy &IIRB) {
    auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);
    bool MayEscape = !LSIConf.AIC->isNonEscapingObj(&V);
    return ConstantInt::get(&Ty, MayEscape);
  }

  Value *instrument(Value *&V, InstrumentationConfig &IConf,
                    InstrumentorIRBuilderTy &IIRB,
                    InstrumentationCaches &ICaches) override {
    if (auto *CI = cast_if_present<CallInst>(
            AllocaIO::instrument(V, IConf, IIRB, ICaches))) {
      auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);
      if (!cast<ConstantInt>(CI->getArgOperand(CI->arg_size() - 1))->isZero()) {
        AllocaInst *AI;
        {
          IRBuilderBase::InsertPointGuard IPG(IIRB.IRB);
          IIRB.IRB.SetInsertPointPastAllocas(CI->getFunction());
          AI = IIRB.IRB.CreateAlloca(IIRB.PtrTy);
        }
        IIRB.IRB.CreateStore(CI, AI);
        LSIConf.EscapedAllocas.push_back(AI);
      }
      LSIConf.AIC->insertRegisterCall(V, CI);
      return CI;
    }
    return nullptr;
  }

  static void populate(InstrumentationConfig &IConf,
                       InstrumentorIRBuilderTy &IIRB) {
    auto *EAIO = IConf.allocate<ExtendedAllocaIO>(/*IsPRE*/ false);
    auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);
    EAIO->CB = [&](Value &V) { return !LSIConf.AIC->isObjectSafe(&V); };
    EAIO->init(IConf, IIRB);
  }
};

struct ExtendedGlobalIO : public GlobalIO {
  ExtendedGlobalIO() : GlobalIO() {}
  virtual ~ExtendedGlobalIO() {};

  void init(InstrumentationConfig &IConf, InstrumentorIRBuilderTy &IIRB) {
    GlobalIO::ConfigTy PreGlobalConfig(/*Enable=*/false);
    PreGlobalConfig.set(GlobalIO::PassAddress);
    PreGlobalConfig.set(GlobalIO::PassInitialValueSize);
    PreGlobalConfig.set(GlobalIO::PassIsDefinition);
    PreGlobalConfig.set(GlobalIO::ReplaceAddress);
    GlobalIO::init(IConf, IIRB.Ctx, &PreGlobalConfig);

    IRTArgs.push_back(IRTArg(IIRB.Int8Ty, "requires_temporal_check",
                             "Flag to indicate that the global might be "
                             "accessed after it was freed.",
                             IRTArg::NONE, getRequiresTemporalCheck));
  }

  static Value *getRequiresTemporalCheck(Value &V, Type &Ty,
                                         InstrumentationConfig &IConf,
                                         InstrumentorIRBuilderTy &IIRB) {
    auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);
    bool MayEscape = !LSIConf.AIC->isNonEscapingObj(&V);
    return ConstantInt::get(&Ty, MayEscape);
  }

  Value *instrument(Value *&V, InstrumentationConfig &IConf,
                    InstrumentorIRBuilderTy &IIRB,
                    InstrumentationCaches &ICaches) override {
    if (auto *CI = cast_if_present<CallInst>(
            GlobalIO::instrument(V, IConf, IIRB, ICaches))) {
      auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);
      LSIConf.AIC->insertRegisterCall(V, CI);
      return CI;
    }
    return nullptr;
  }

  static void populate(InstrumentationConfig &IConf,
                       InstrumentorIRBuilderTy &IIRB) {
    auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);
    auto *EAIO = IConf.allocate<ExtendedGlobalIO>();
    EAIO->CB = [&](Value &V) {
      if (LSIConf.AIC->isObjectSafe(&V))
        return false;
      auto &GV = cast<GlobalVariable>(V);
      return GV.getValueType()->isSized() && !GV.hasWeakLinkage() &&
             !GV.isInterposable();
    };
    EAIO->init(IConf, IIRB);
  }
};

struct AllocatorCallIO : public CallIO {
  AllocatorCallIO() : CallIO(/*IsPRE*/ false) {}
  virtual ~AllocatorCallIO() {};

  void init(InstrumentationConfig &IConf, InstrumentorIRBuilderTy &IIRB) {
    CallIO::ConfigTy PostCICConfig(/*Enable=*/false);
    PostCICConfig.set(CallIO::PassReturnedValue);
    CallIO::init(IConf, IIRB.Ctx, &PostCICConfig);

    IRTArgs.push_back(IRTArg(IIRB.Int64Ty, "object_size",
                             "The allocatdd object size.", IRTArg::NONE,
                             getObjSize));
    IRTArgs.push_back(IRTArg(IIRB.Int8Ty, "requires_temporal_check",
                             "Flag to indicate that the global might be "
                             "accessed after it was freed.",
                             IRTArg::NONE, getRequiresTemporalCheck));
  }

  static Value *getObjSize(Value &V, Type &Ty, InstrumentationConfig &IConf,
                           InstrumentorIRBuilderTy &IIRB) {
    auto &CI = cast<CallInst>(V);
    auto &TLI = IIRB.TLIGetter(*CI.getFunction());
    auto ACI = getAllocationCallInfo(&CI, &TLI);
    Value *Size = nullptr;
    for (auto Idx : {ACI->SizeLHSArgNo, ACI->SizeRHSArgNo}) {
      if (Idx >= 0) {
        auto *V = CI.getArgOperand(ACI->SizeLHSArgNo);
        Size = Size ? IIRB.IRB.CreateMul(Size, V) : V;
      }
    }
    return Size;
  }
  static Value *getRequiresTemporalCheck(Value &V, Type &Ty,
                                         InstrumentationConfig &IConf,
                                         InstrumentorIRBuilderTy &IIRB) {
    auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);
    bool MayEscape = !LSIConf.AIC->isNonEscapingObj(&V);
    return ConstantInt::get(&Ty, MayEscape);
  }

  static void populate(InstrumentationConfig &IConf,
                       InstrumentorIRBuilderTy &IIRB) {
    auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);
    auto *EAIO = IConf.allocate<AllocatorCallIO>();
    EAIO->CB = [&](Value &V) {
      if (LSIConf.AIC->isObjectSafe(&V))
        return false;
      auto &CI = cast<CallInst>(V);
      if (CI.getCalledFunction() && !CI.getCalledFunction()->isDeclaration())
        return false;
      auto &TLI = IIRB.TLIGetter(*CI.getFunction());
      auto ACI = getAllocationCallInfo(&CI, &TLI);
      // TODO: check for escaping -> temporal checks
      return !!ACI;
    };
    EAIO->init(IConf, IIRB);
  }

  virtual Type *getRetTy(LLVMContext &Ctx) const override {
    return PointerType::getUnqual(Ctx);
  }
};

struct ExtendedBasePointerIO : public BasePointerIO {
  ExtendedBasePointerIO() : BasePointerIO() {}
  virtual ~ExtendedBasePointerIO() {};

  void init(InstrumentationConfig &IConf, LLVMContext &Ctx) {
    BasePointerIO::ConfigTy BaseConfig(/*Enable=*/false);
    BaseConfig.set(BasePointerIO::PassPointer, true);
    BasePointerIO::init(IConf, Ctx, &BaseConfig);
    IRTArgs.push_back(
        IRTArg(PointerType::getUnqual(Ctx), "object_size_ptr",
               "Return the size of the object in question as uint64_t.",
               IRTArg::NONE, getObjectSizePtr));
    IRTArgs.push_back(IRTArg(
        PointerType::getUnqual(Ctx), "encoding_no_ptr",
        "Return the encoding number of the object in question as uint8_t.",
        IRTArg::NONE, getEncodingNoPtr));
  }

  static Value *getObjectSizePtr(Value &V, Type &Ty,
                                 InstrumentationConfig &IConf,
                                 InstrumentorIRBuilderTy &IIRB) {
    Function *Fn = IIRB.IRB.GetInsertBlock()->getParent();
    return IIRB.getAlloca(Fn, IIRB.Int64Ty);
  }
  static Value *getEncodingNoPtr(Value &V, Type &Ty,
                                 InstrumentationConfig &IConf,
                                 InstrumentorIRBuilderTy &IIRB) {
    Function *Fn = IIRB.IRB.GetInsertBlock()->getParent();
    return IIRB.getAlloca(Fn, IIRB.Int8Ty);
  }

  Value *instrument(Value *&V, InstrumentationConfig &IConf,
                    InstrumentorIRBuilderTy &IIRB,
                    InstrumentationCaches &ICaches) override {
    if (auto *I = dyn_cast<Instruction>(V)) {
      auto &LI = IIRB.LIGetter(*I->getFunction());
      if (auto *L = LI.getLoopFor(I->getParent())) {
        errs() << "Base pointer " << *I << " in " << I->getFunction()->getName()
               << "\n"
               << *L << "\n";
      }
    }
    auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);
    auto *NewV = BasePointerIO::instrument(V, IConf, IIRB, ICaches);
    auto *CI = cast<CallInst>(NewV);
    auto *VPtr = CI->getArgOperand(0);
    Value *MPtr = nullptr, *ObjSize = nullptr, *Obj = nullptr;
    if (auto *VPtrCI = dyn_cast<CallInst>(VPtr)) {
      auto *Callee = VPtrCI->getCalledFunction();
      if (Callee && (Callee->getName() == IConf.getRTName("post_", "call"))) {
        MPtr = VPtrCI->getArgOperand(0);
        Obj = MPtr;
        ObjSize = VPtrCI->getArgOperand(1);
      }
      if (Callee && (Callee->getName() == IConf.getRTName("post_", "alloca"))) {
        MPtr = VPtrCI->getArgOperand(0);
        Obj = MPtr;
        ObjSize = VPtrCI->getArgOperand(1);
      }
      // TODO: load of shadow global
    }

    Function *Fn = CI->getFunction();
    Value *BaseMPtr = MPtr;
    if (!BaseMPtr)
      BaseMPtr = CI;

    assert(IIRB.IRB.GetInsertPoint() == CI->getNextNode()->getIterator());
    if (!ObjSize)
      ObjSize = IIRB.IRB.CreateLoad(IIRB.Int64Ty, CI->getArgOperand(1));
    auto &EBPI = LSIConf.BasePointerSizeOffsetMap[{VPtr, Fn}];
    EBPI.ObjectSize = ObjSize;
    if (Obj && (LSIConf.AIC->isKnownObject(Obj) &&
                (!LSIConf.AIC->isNonEscapingObj(Obj) ||
                 LSIConf.AIC->getObjectSize(Obj) >= (1LL << 12)))) {
      EBPI.EncodingNo = IIRB.IRB.getInt8(LargeObjectEnc);
    } else if (Obj && (LSIConf.AIC->isKnownObject(Obj) &&
                       (LSIConf.AIC->isNonEscapingObj(Obj) &&
                        LSIConf.AIC->getObjectSize(Obj) < (1LL << 12)))) {
      EBPI.EncodingNo = IIRB.IRB.getInt8(SmallObjectEnc);
    } else {
      EBPI.EncodingNo = IIRB.IRB.CreateLoad(IIRB.Int8Ty, CI->getArgOperand(2));
    }

    if (!MPtr) {
      auto MPtrFC = Fn->getParent()->getOrInsertFunction(
          IConf.getRTName("", "get_mptr"),
          FunctionType::get(IIRB.PtrTy, {IIRB.PtrTy, IIRB.PtrTy, IIRB.Int8Ty},
                            false));
      MPtr = IIRB.IRB.CreateCall(MPtrFC, {VPtr, BaseMPtr, EBPI.EncodingNo});
    }

    auto *&MappedMPtr = LSIConf.V2M[{VPtr, Fn}];
    if (MappedMPtr) {
      // This can happen if someone called getMPtr before someone requested the
      // base pointer info. Unsure if we should prevent it.
      MappedMPtr->replaceAllUsesWith(MPtr);
    }
    MappedMPtr = MPtr;

    return BaseMPtr;
  }

  static void populate(InstrumentationConfig &IConf,
                       InstrumentorIRBuilderTy &IIRB) {
    auto *EVPIO = IConf.allocate<ExtendedBasePointerIO>();
    EVPIO->init(IConf, IIRB.Ctx);
  }

  virtual Type *getRetTy(LLVMContext &Ctx) const override {
    return PointerType::get(Ctx, 0);
  }
};

struct ExtendedLoopValueRangeIO : public LoopValueRangeIO {
  ExtendedLoopValueRangeIO() : LoopValueRangeIO() {}
  virtual ~ExtendedLoopValueRangeIO() {};

  StringRef getName() const override { return "loop_value_range"; }

  void init(InstrumentationConfig &IConf, InstrumentorIRBuilderTy &IIRB) {
    LoopValueRangeIO::ConfigTy Config(/*Enable=*/true);
    Config.set(LoopValueRangeIO::PassId, false);
    LoopValueRangeIO::init(IConf, IIRB, &Config);

    IRTArgs.push_back(IRTArg(IIRB.Int64Ty, "object_size",
                             "The size of the underlying object.", IRTArg::NONE,
                             getObjectSize));
    IRTArgs.push_back(IRTArg(IIRB.Int8Ty, "encoding_no",
                             "The encoding number used for the pointer.",
                             IRTArg::NONE, getEncodingNo));
    IRTArgs.push_back(
        IRTArg(IIRB.Int8Ty, "is_definitively_executed",
               "Flag to indicate the range is definitively executed.",
               IRTArg::NONE, getIsDefinitivelyExecuted));
  }

  static Value *getObjectSize(Value &V, Type &Ty, InstrumentationConfig &IConf,
                              InstrumentorIRBuilderTy &IIRB) {
    auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);
    return LSIConf.getBasePointerObjectSize(V, IIRB);
  }
  static Value *getEncodingNo(Value &V, Type &Ty, InstrumentationConfig &IConf,
                              InstrumentorIRBuilderTy &IIRB) {
    auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);
    return LSIConf.getBasePointerEncodingNo(V, IIRB);
  }
  static Value *getIsDefinitivelyExecuted(Value &V, Type &Ty,
                                          InstrumentationConfig &IConf,
                                          InstrumentorIRBuilderTy &IIRB) {
    return ConstantInt::get(&Ty, 0);
  }

  static void populate(InstrumentationConfig &IConf,
                       InstrumentorIRBuilderTy &IIRB) {
    auto *LVRIO = IConf.allocate<ExtendedLoopValueRangeIO>();
    LVRIO->HoistKind = HOIST_IN_BLOCK;
    LVRIO->init(IConf, IIRB);
  }
};

struct ExtendedLoadIO : public LoadIO {
  ExtendedLoadIO(bool IsPRE) : LoadIO(IsPRE) {}
  virtual ~ExtendedLoadIO() {};

  void init(InstrumentationConfig &IConf, InstrumentorIRBuilderTy &IIRB) {
    LoadIO::ConfigTy LICConfig(/*Enable=*/false);
    LICConfig.set(LoadIO::PassPointer);
    LICConfig.set(LoadIO::ReplacePointer);
    LICConfig.set(LoadIO::PassBasePointerInfo);
    LICConfig.set(LoadIO::PassLoopValueRangeInfo);
    LICConfig.set(LoadIO::PassValueSize);
    LoadIO::init(IConf, IIRB, &LICConfig);

    IRTArgs.push_back(
        IRTArg(IIRB.PtrTy, "mptr", "The real pointer.", IRTArg::NONE, getMPtr));
    IRTArgs.push_back(IRTArg(IIRB.Int64Ty, "object_size",
                             "The size of the underlying object.", IRTArg::NONE,
                             getObjectSize));
    IRTArgs.push_back(IRTArg(IIRB.Int8Ty, "encoding_no",
                             "The encoding number used for the pointer.",
                             IRTArg::NONE, getEncodingNo));
    IRTArgs.push_back(IRTArg(IIRB.Int8Ty, "was_checked",
                             "Flag to indicate the access range was checked.",
                             IRTArg::NONE, getWasChecked));
  }

  static Value *getMPtr(Value &V, Type &Ty, InstrumentationConfig &IConf,
                        InstrumentorIRBuilderTy &IIRB) {
    auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);
    auto &LI = cast<LoadInst>(V);
    return LSIConf.getMPtr(*LI.getPointerOperand(), IIRB);
  }
  static Value *getObjectSize(Value &V, Type &Ty, InstrumentationConfig &IConf,
                              InstrumentorIRBuilderTy &IIRB) {
    auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);
    auto &LI = cast<LoadInst>(V);
    return LSIConf.getBasePointerObjectSize(*LI.getPointerOperand(), IIRB);
  }
  static Value *getEncodingNo(Value &V, Type &Ty, InstrumentationConfig &IConf,
                              InstrumentorIRBuilderTy &IIRB) {
    auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);
    auto &LI = cast<LoadInst>(V);
    return LSIConf.getBasePointerEncodingNo(*LI.getPointerOperand(), IIRB);
  }
  static Value *getWasChecked(Value &V, Type &Ty, InstrumentationConfig &IConf,
                              InstrumentorIRBuilderTy &IIRB) {
    auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);
    return ConstantInt::get(&Ty,
                            LSIConf.AIC->isAccessSafe(cast<Instruction>(&V)));
  }

  static void populate(InstrumentationConfig &IConf,
                       InstrumentorIRBuilderTy &IIRB) {
    auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);
    auto *ESIO = IConf.allocate<ExtendedLoadIO>(/*IsPRE*/ true);
    ESIO->HoistKind = HOIST_IN_BLOCK;
    ESIO->CB = [&](Value &V) {
      if (auto *Obj = LSIConf.AIC->getSafeAccessObj(cast<Instruction>(&V)))
        return !LSIConf.AIC->isObjectSafe(Obj);
      return true;
    };
    ESIO->init(IConf, IIRB);
  }
};

struct ExtendedStoreIO : public StoreIO {
  ExtendedStoreIO(bool IsPRE) : StoreIO(IsPRE) {}
  virtual ~ExtendedStoreIO() {};

  void init(InstrumentationConfig &IConf, InstrumentorIRBuilderTy &IIRB) {
    StoreIO::ConfigTy SICConfig(/*Enable=*/false);
    SICConfig.set(StoreIO::PassPointer);
    SICConfig.set(StoreIO::ReplacePointer);
    SICConfig.set(StoreIO::PassBasePointerInfo);
    SICConfig.set(StoreIO::PassLoopValueRangeInfo);
    SICConfig.set(StoreIO::PassStoredValueSize);
    StoreIO::init(IConf, IIRB, &SICConfig);

    IRTArgs.push_back(
        IRTArg(IIRB.PtrTy, "mptr", "The real pointer.", IRTArg::NONE, getMPtr));
    IRTArgs.push_back(IRTArg(IIRB.Int64Ty, "object_size",
                             "The size of the underlying object.", IRTArg::NONE,
                             getObjectSize));
    IRTArgs.push_back(IRTArg(IIRB.Int8Ty, "encoding_no",
                             "The encoding number used for the pointer.",
                             IRTArg::NONE, getEncodingNo));
    IRTArgs.push_back(IRTArg(IIRB.Int8Ty, "was_checked",
                             "Flag to indicate the access range was checked.",
                             IRTArg::NONE, getWasChecked));
  }

  static Value *getMPtr(Value &V, Type &Ty, InstrumentationConfig &IConf,
                        InstrumentorIRBuilderTy &IIRB) {
    auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);
    auto &SI = cast<StoreInst>(V);
    return LSIConf.getMPtr(*SI.getPointerOperand(), IIRB);
  }
  static Value *getObjectSize(Value &V, Type &Ty, InstrumentationConfig &IConf,
                              InstrumentorIRBuilderTy &IIRB) {
    auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);
    auto &SI = cast<StoreInst>(V);
    return LSIConf.getBasePointerObjectSize(*SI.getPointerOperand(), IIRB);
  }
  static Value *getEncodingNo(Value &V, Type &Ty, InstrumentationConfig &IConf,
                              InstrumentorIRBuilderTy &IIRB) {
    auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);
    auto &SI = cast<StoreInst>(V);
    return LSIConf.getBasePointerEncodingNo(*SI.getPointerOperand(), IIRB);
  }
  static Value *getWasChecked(Value &V, Type &Ty, InstrumentationConfig &IConf,
                              InstrumentorIRBuilderTy &IIRB) {
    auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);
    return ConstantInt::get(&Ty,
                            LSIConf.AIC->isAccessSafe(cast<Instruction>(&V)));
  }

  static void populate(InstrumentationConfig &IConf,
                       InstrumentorIRBuilderTy &IIRB) {
    auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);
    auto *ESIO = IConf.allocate<ExtendedStoreIO>(/*IsPRE*/ true);
    ESIO->HoistKind = HOIST_IN_BLOCK;
    ESIO->CB = [&](Value &V) {
      if (auto *Obj = LSIConf.AIC->getSafeAccessObj(cast<Instruction>(&V)))
        return !LSIConf.AIC->isObjectSafe(Obj);
      return true;
    };
    ESIO->init(IConf, IIRB);
  }
};

struct ExtendedFunctionIO : public FunctionIO {
  ExtendedFunctionIO(bool IsPRE) : FunctionIO(IsPRE) {}
  virtual ~ExtendedFunctionIO() {};

  void init(InstrumentationConfig &IConf, InstrumentorIRBuilderTy &IIRB) {
    FunctionIO::ConfigTy FCConfig(/*Enable=*/false);
    bool IsPRE = getLocationKind() == InstrumentationLocation::FUNCTION_PRE;
    if (IsPRE) {
      FCConfig.set(FunctionIO::PassArguments);
      FCConfig.set(FunctionIO::PassNumArguments);
      FCConfig.set(FunctionIO::ReplaceArguments);
    } else {
      IRTArgs.push_back(IRTArg(IIRB.Int32Ty, "num_allocas",
                               "The number of allocas that are deallocated.",
                               IRTArg::NONE, getNumAllocas));
      IRTArgs.push_back(
          IRTArg(IIRB.PtrTy, "allocas_ptr",
                 "Pointer to the array of allocas that are deallocated.",
                 IRTArg::NONE, getAllocaPtr));
    }
    FunctionIO::init(IConf, IIRB.Ctx, &FCConfig);
  }

  static Value *getNumAllocas(Value &V, Type &Ty, InstrumentationConfig &IConf,
                              InstrumentorIRBuilderTy &IIRB) {
    auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);
    return IIRB.IRB.getInt32(LSIConf.EscapedAllocas.size());
  }

  static Value *getAllocaPtr(Value &V, Type &Ty, InstrumentationConfig &IConf,
                             InstrumentorIRBuilderTy &IIRB) {
    auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);

    Function *Fn = IIRB.IRB.GetInsertBlock()->getParent();
    auto *ATy = ArrayType::get(IIRB.PtrTy, LSIConf.EscapedAllocas.size());
    auto *AllocasAI = IIRB.getAlloca(Fn, ATy);
    for (auto [Idx, AI] : enumerate(LSIConf.EscapedAllocas)) {
      auto *Ptr = IIRB.IRB.CreateConstGEP1_32(IIRB.PtrTy, AllocasAI, Idx);
      IIRB.IRB.CreateStore(IIRB.IRB.CreateLoad(IIRB.PtrTy, AI), Ptr);
    }
    return AllocasAI;
  }

  Value *instrument(Value *&V, InstrumentationConfig &IConf,
                    InstrumentorIRBuilderTy &IIRB,
                    InstrumentationCaches &ICaches) override {

    // auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);
    // Function *Fn = cast<Function>(V);
    // auto GetSizeFC = Fn->getParent()->getOrInsertFunction(
    //     IConf.getRTName("", "get_object_size"),
    //     FunctionType::get(IIRB.Int64Ty, {IIRB.PtrTy}, false));
    //{
    //   IRBuilderBase::InsertPointGuard IPG(IIRB.IRB);
    //   for (auto *CI : LSIConf.PotentiallyFreeCalls) {
    //     for (auto [Obj, SizeAI] : LSIConf.SizeAllocas) {
    //       IIRB.IRB.SetInsertPoint(CI->getNextNode());
    //       ensureDbgLoc(IIRB.IRB);
    //       CallInst *NewSizeVal = IIRB.IRB.CreateCall(GetSizeFC, {Obj},
    //       "size"); IIRB.IRB.CreateStore(NewSizeVal, SizeAI);
    //     }
    //   }
    // }
    return FunctionIO::instrument(V, IConf, IIRB, ICaches);
  }

  static void populate(InstrumentationConfig &IConf,
                       InstrumentorIRBuilderTy &IIRB) {
    auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);
    for (auto IsPRE : {true, false}) {
      auto *EAIO = IConf.allocate<ExtendedFunctionIO>(IsPRE);
      if (IsPRE)
        EAIO->CB = [](Value &V) { return V.getName() == "main"; };
      else
        EAIO->CB = [&](Value &V) { return !LSIConf.EscapedAllocas.empty(); };
      EAIO->init(IConf, IIRB);
    }
  }
};

void LightSanInstrumentationConfig::populate(InstrumentorIRBuilderTy &IIRB) {
  UnreachableIO::ConfigTy UIOConfig(/*Enable=*/false);
  UnreachableIO::populate(*this, IIRB.Ctx, &UIOConfig);
  ExtendedBasePointerIO::populate(*this, IIRB);
  ExtendedStoreIO::populate(*this, IIRB);
  ExtendedLoadIO::populate(*this, IIRB);
  ExtendedLoopValueRangeIO::populate(*this, IIRB);
  ExtendedAllocaIO::populate(*this, IIRB);
  ExtendedFunctionIO::populate(*this, IIRB);
  ExtendedGlobalIO::populate(*this, IIRB);
  AllocatorCallIO::populate(*this, IIRB);
  // ModuleIO::populate(*this, IIRB.Ctx);

  CallIO::ConfigTy PreCICConfig(/*Enable=*/false);
  PreCICConfig.set(CallIO::PassIntrinsicId);
  PreCICConfig.set(CallIO::PassNumParameters);
  PreCICConfig.set(CallIO::PassParameters);
  PreCICConfig.set(CallIO::PassIsDefinition);
  PreCICConfig.ArgFilter = [&](Use &Op) {
    return Op->getType()->isPointerTy();
  };
  auto *PreCIC = InstrumentationConfig::allocate<CallIO>(/*IsPRE=*/true);
  PreCIC->CB = [&](Value &V) {
    return LSI.shouldInstrumentCall(cast<CallInst>(V), IIRB);
  };
  PreCIC->init(*this, IIRB.Ctx, &PreCICConfig);
}

PreservedAnalyses run(Module &M, AnalysisManager<Module> &MAM) {
  LightSanImpl Impl(M, MAM);
  LLVM_DEBUG(dbgs() << "Running objsan\n");

  bool Changed = Impl.instrument();
  if (!Changed)
    return PreservedAnalyses::all();

  SmallVector<Function *> DeadFns;
  for (Function &Fn : M)
    if (Fn.use_empty() && Fn.getName().starts_with(LightSanRuntimePrefix))
      DeadFns.push_back(&Fn);
  for (auto *Fn : DeadFns)
    Fn->eraseFromParent();

  if (verifyModule(M))
    M.dump();

  assert(!verifyModule(M, &errs()));

  ModulePassManager MPM;
  MPM.addPass(AlwaysInlinerPass(/*InsertLifetimeIntrinsics=*/true));
  auto GetFPM = [&]() -> FunctionPassManager {
    FunctionPassManager FPM;
    FPM.addPass(SROAPass(SROAOptions::ModifyCFG));
    FPM.addPass(
        SimplifyCFGPass(SimplifyCFGOptions().convertSwitchRangeToICmp(true)));
    FPM.addPass(InstCombinePass());
    return FPM;
  };
  MPM.addPass(createModuleToFunctionPassAdaptor(GetFPM()));
  MPM.run(M, MAM);

  return PreservedAnalyses::none();
}

} // namespace

PreservedAnalyses LightSanPass::run(Module &M, AnalysisManager<Module> &MAM) {
  static constexpr char ModuleFlag[] = "sanitize_obj";
  switch (Phase) {
  case ThinOrFullLTOPhase::None:
    return ::run(M, MAM);
  case ThinOrFullLTOPhase::ThinLTOPreLink:
  case ThinOrFullLTOPhase::FullLTOPreLink:
    M.addModuleFlag(llvm::Module::Max, ModuleFlag, 1);
    return PreservedAnalyses::all();
  case ThinOrFullLTOPhase::ThinLTOPostLink:
  case ThinOrFullLTOPhase::FullLTOPostLink:
    if (M.getModuleFlag(ModuleFlag))
      return ::run(M, MAM);
    return PreservedAnalyses::all();
  }
  llvm_unreachable("Unknown LTO phase.");
}
