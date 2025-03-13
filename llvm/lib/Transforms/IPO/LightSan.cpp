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

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Transforms/IPO/Attributor.h"
#include "llvm/Transforms/IPO/Instrumentor.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <cstdint>
#include <functional>

using namespace llvm;
using namespace llvm::instrumentor;

#define DEBUG_TYPE "lightsan"

static constexpr char LightSanRuntimePrefix[] = "__objsan_";
[[maybe_unused]] static constexpr int64_t SmallObjectEnc = 1;
static constexpr int64_t LargeObjectEnc = 2;

namespace {

static Value *stripRegisterCall(Value *V, InstrumentationConfig &IConf) {
  V = V->stripPointerCasts();
  if (auto *CI = dyn_cast<CallInst>(V)) {
    assert(CI->getCalledFunction());
    auto CalleeName = CI->getCalledFunction()->getName();
    if (CalleeName == IConf.getRTName("post_", "alloca")) {
      return CI->getArgOperand(0);
    }
    if (CalleeName == IConf.getRTName("post_", "call")) {
      auto *P2I = cast<PtrToIntInst>(CI->getArgOperand(1));
      return P2I->getOperand(0);
    }
    return CI;
  }
  if (auto *I2P = dyn_cast<IntToPtrInst>(V))
    return stripRegisterCall(I2P->getOperand(0), IConf);
  return V;
}

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
    return SafeAccesses[I] = Obj;
  }

  bool insertSafeObject(Value *Obj) { return SafeObjects.insert(Obj); }

  bool isObjectSafe(Value *Obj) const { return SafeObjects.contains(Obj); }

  bool insertSanitizedObject(Value *Obj, int64_t Size) {
    return SanitizedObjects[Obj] = Size;
  }
  bool isSanitizedObject(Value *Obj) const {
    return SanitizedObjects.contains(Obj);
  }
  int64_t getSanitizedObjectSize(Value *Obj) const {
    return SanitizedObjects.lookup(Obj);
  }

  bool insertNonEscapingObj(Value *Obj) {
    return NonEscapingObjects.insert(Obj);
  }
  bool isNonEscapingObj(Value *Obj) const {
    return NonEscapingObjects.contains(Obj);
  }

  Attributor &getAttributor() const { return A; }

private:
  DenseMap<Value *, int64_t> SanitizedObjects;
  SetVector<Value *> NonEscapingObjects;
  /// Objects that have been verified that all their accesses are safe.
  SetVector<Value *> SafeObjects;
  /// Accesses that are known safe.
  DenseMap<Instruction *, Value *> SafeAccesses;

  Attributor &A;
};

struct LightSanInstrumentationConfig : public InstrumentationConfig {

  LightSanInstrumentationConfig(LightSanImpl &LSI);
  virtual ~LightSanInstrumentationConfig() {}

  void populate(InstrumentorIRBuilderTy &IRB) override;

  struct ExtendedBasePointerInfo {
    Value *ObjectSize = nullptr;
    Value *ObjectSizePtr = nullptr;
    Value *EncodingNo = nullptr;
  };

  DenseMap<std::pair<Value *, Function *>, ExtendedBasePointerInfo>
      BasePointerSizeOffsetMap;
  Value *getBasePointerObjectSize(Value &V, InstrumentorIRBuilderTy &IIRB) {
    Function *Fn = IIRB.IRB.GetInsertBlock()->getParent();
    Value *Ptr = instrumentor::getUnderlyingObjectRecursive(&V);
    auto &EBPI = BasePointerSizeOffsetMap[{Ptr, Fn}];
    if (EBPI.ObjectSizePtr)
      return IIRB.IRB.CreateLoad(IIRB.Int64Ty, EBPI.ObjectSizePtr);
    return EBPI.ObjectSize;
  }

  Value *getStaticBasePointerEncodingNo(Value *Ptr) {
    Ptr = stripRegisterCall(Ptr, *this);
    if ((AIC->isSanitizedObject(Ptr) &&
         (!AIC->isNonEscapingObj(Ptr) ||
          AIC->getSanitizedObjectSize(Ptr) >= (1LL << 12)))) {
      return ConstantInt::get(IntegerType::getInt8Ty(Ptr->getContext()),
                              LargeObjectEnc);
    }
    return nullptr;
  }
  Value *getStaticBasePointerEncodingNo(Instruction &I) {
    if (auto *Obj = AIC->getSafeAccessObj(&I))
      return getStaticBasePointerEncodingNo(Obj);
    return nullptr;
  }
  Value *getBasePointerEncodingNo(Value &V, Function &Fn) {
    Value *Ptr = instrumentor::getUnderlyingObjectRecursive(&V);
    if (auto *EncNo = getStaticBasePointerEncodingNo(Ptr))
      return EncNo;
    return BasePointerSizeOffsetMap[{Ptr, &Fn}].EncodingNo;
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

  LightSanImpl &LSI;

  AttributorInfoCache *AIC = nullptr;
};

struct LightSanImpl {
  LightSanImpl(Module &M, ModuleAnalysisManager &MAM)
      : M(M), MAM(MAM),
        FAM(MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager()),
        IConf(*this) {}

  bool instrument();

  bool shouldInstrumentFunction(Function &Fn);
  bool shouldInstrumentCall(CallInst &CI, InstrumentorIRBuilderTy &IIRB);
  bool shouldInstrumentLoad(LoadInst &LI, InstrumentorIRBuilderTy &IIRB);
  bool shouldInstrumentStore(StoreInst &SI, InstrumentorIRBuilderTy &IIRB);

private:
  bool updateSizesAfterPotentialFree();
  bool hoistLoopLoads(Loop &L);
  void foreachRTCaller(StringRef Name, function_ref<void(CallInst &)> CB);

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

bool LightSanImpl::shouldInstrumentFunction(Function &Fn) {
  return Fn.hasFnAttribute(Attribute::SanitizeObj);
}

bool LightSanImpl::shouldInstrumentCall(CallInst &CI,
                                        InstrumentorIRBuilderTy &IIRB) {
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

  Function *CalledFn = CI.getCalledFunction();
  if (!CalledFn)
    return true;
  if (!CalledFn->isDeclaration())
    return false;
  if (CalledFn->getName().starts_with(LightSanRuntimePrefix))
    return false;
  FunctionType *CalledFnTy = CalledFn->getFunctionType();
  if (none_of(CalledFnTy->params(),
              [&](Type *ArgTy) { return ArgTy->isPtrOrPtrVectorTy(); }))
    return false;
  LibFunc TheLibFunc;
  auto &TLI = IIRB.TLIGetter(*CalledFn);
  if (!(TLI.getLibFunc(*CalledFn, TheLibFunc) && TLI.has(TheLibFunc)))
    return true;
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
}

bool LightSanImpl::hoistLoopLoads(Loop &L) {
  bool Changed = false;
  for (auto *ChildL : L)
    Changed |= hoistLoopLoads(*ChildL);

  auto *LatchBB = L.getLoopLatch();
  auto *PreHeaderBB = L.getLoopPreheader();
  if (!LatchBB || !PreHeaderBB)
    return Changed;
  auto *HeaderBB = L.getHeader();

  // auto *Int64Ty = IntegerType::getInt64Ty(M.getContext());
  // auto *PtrTy = PointerType::get(M.getContext(), 0);

  SmallVector<std::tuple<LoadInst *, PHINode *, APInt>> Loads;
  DenseMap<LoadInst *, std::pair<LoadInst *, APInt>> LoadMap;
  for (auto *BB : L.blocks())
    for (auto &I : *BB)
      if (auto *LI = dyn_cast<LoadInst>(&I)) {
        auto *Ty = LI->getType();
        auto AccessSize = DL.getTypeStoreSize(Ty);
        if (AccessSize != 1 && AccessSize != 2 && AccessSize != 4 &&
            AccessSize != 8)
          continue;
        auto *Ptr = LI->getPointerOperand();
        APInt Offset(
            DL.getIndexSizeInBits(Ptr->getType()->getPointerAddressSpace()), 0);
        auto *UnderlyingPtr = Ptr->stripAndAccumulateConstantOffsets(
            DL, Offset, /*AllowNonInbounds=*/true,
            /* AllowInvariant */ true);
        if (auto *ULI = dyn_cast<LoadInst>(UnderlyingPtr)) {
          LoadMap[ULI] = {LI, Offset};
          continue;
        }
        auto *UPtrPHI = dyn_cast<PHINode>(UnderlyingPtr);
        if (!UPtrPHI || UPtrPHI->getParent() != HeaderBB)
          continue;
        Loads.push_back({LI, UPtrPHI, Offset});
      }

  while (!Loads.empty()) {
    auto [LI, UPtrPHI, Offset] = Loads.pop_back_val();
    auto It = LoadMap.find(LI);
    if (It == LoadMap.end())
      continue;
    auto *Ty = LI->getType();
    //    auto AccessSize = DL.getTypeStoreSize(Ty);

    auto *InitialPtrVal = UPtrPHI->getIncomingValueForBlock(PreHeaderBB);
    auto *LatchPtrValI =
        dyn_cast<Instruction>(UPtrPHI->getIncomingValueForBlock(LatchBB));
    if (!LatchPtrValI)
      continue;
    auto *ValPHI = PHINode::Create(Ty, 2, LI->getName() + ".spec_val",
                                   UPtrPHI->getIterator());
    LI->replaceAllUsesWith(ValPHI);
    if (LatchPtrValI == LI)
      LatchPtrValI = ValPHI;

    //    auto SpecLoadFC = M.getOrInsertFunction(
    //        IConf.getRTName("pre_spec_", "load_", std::to_string(AccessSize)),
    //        FunctionType::get(Int64Ty, {PtrTy, Int64Ty}, false));

    auto *LatchLI = LI->clone();
    LI->removeFromParent();

    IRBuilder<> IRB(LatchBB->getTerminator());
    ensureDbgLoc(IRB);
    LI->addAnnotationMetadata("speculated");
    LatchLI->addAnnotationMetadata("speculated");
    auto *OffsetVal = IRB.getInt64(Offset.getSExtValue());
    //    Value *InitialVal = IRB.CreateCall(SpecLoadFC, {InitialPtrVal,
    //    OffsetVal},
    //                                       LI->getName() + ".spec_pre");
    //    InitialVal = tryToCast(IRB, InitialVal, Ty, DL);
    ValPHI->addIncoming(LI, PreHeaderBB);

    //    Value *LatchVal = IRB.CreateCall(SpecLoadFC, {LatchPtrValI,
    //    OffsetVal},
    //                                     LI->getName() + ".spec_latch");
    //    LatchVal = tryToCast(IRB, LatchVal, Ty, DL);
    ValPHI->addIncoming(LatchLI, LatchBB);

    auto *LatchPtrWithOffset =
        IRB.CreateGEP(IRB.getInt8Ty(), LatchPtrValI, {OffsetVal});
    IRB.Insert(LatchLI);
    LatchLI->setOperand(LI->getPointerOperandIndex(), LatchPtrWithOffset);

    IRB.SetInsertPoint(PreHeaderBB->getTerminator());
    ensureDbgLoc(IRB);
    auto *InitialPtrWithOffset =
        IRB.CreateGEP(IRB.getInt8Ty(), InitialPtrVal, {OffsetVal});
    IRB.Insert(LI);
    LI->setOperand(LI->getPointerOperandIndex(), InitialPtrWithOffset);

    //      LI->replaceUsesWithIf(LatchVal, [&](Use &U) {
    //        return isa<PHINode>(U.getUser()) &&
    //               cast<PHINode>(U.getUser())->getParent() == HeaderBB;
    //      });
    //    LI->eraseFromParent();
    Changed = true;
    Loads.push_back({It->second.first, ValPHI, It->second.second});
  }
  return Changed;
}

static bool collectAttributorInfo(Attributor &A, Module &M,
                                  AttributorInfoCache &Cache) {
  const DataLayout &DL = M.getDataLayout();

  // A list of pairs of objects and their size
  SmallVector<std::tuple<Value *, int64_t, const AAPointerInfo *>> WorkList;

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

  auto HandleMallocLikeFn = [&](CallInst *CI, AllocationCallInfo &ACI) {
    uint64_t Size = 1;
    if (ACI.SizeLHSArgNo >= 0) {
      auto *SizeCI = dyn_cast<ConstantInt>(CI->getArgOperand(ACI.SizeLHSArgNo));
      if (!SizeCI)
        return;
      Size *= SizeCI->getSExtValue();
    }
    if (ACI.SizeRHSArgNo >= 0) {
      auto *SizeCI = dyn_cast<ConstantInt>(CI->getArgOperand(ACI.SizeRHSArgNo));
      if (!SizeCI)
        return;
      Size *= SizeCI->getSExtValue();
    }
    auto *AAPI = A.getOrCreateAAFor<AAPointerInfo>(IRPosition::value(*CI),
                                                   /*QueryingAA=*/nullptr,
                                                   DepClassTy::REQUIRED);
    WorkList.emplace_back(CI, Size, AAPI);
  };

  DenseMap<Instruction *, const AAUnderlyingObjects *> InstToAAUOMap;
  for (Function &F : M) {
    if (F.isIntrinsic())
      continue;

    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (auto *SI = dyn_cast<StoreInst>(&I)) {
          auto *AAUO = A.getOrCreateAAFor<AAUnderlyingObjects>(
              IRPosition::value(*SI->getPointerOperand()));
          InstToAAUOMap[SI] = AAUO;
        } else if (auto *LI = dyn_cast<LoadInst>(&I)) {
          auto *AAUO = A.getOrCreateAAFor<AAUnderlyingObjects>(
              IRPosition::value(*LI->getPointerOperand()));
          InstToAAUOMap[LI] = AAUO;
        } else if (auto *AI = dyn_cast<AllocaInst>(&I)) {
          HandleAllocaInst(AI);
        } else if (auto *CI = dyn_cast<CallInst>(&I)) {
          if (CI->getCalledFunction()) {
            const auto *TLI = A.getInfoCache().getTargetLibraryInfoForFunction(
                *CI->getCalledFunction());
            assert(TLI);
            if (auto ACI = getAllocationCallInfo(CI, TLI))
              HandleMallocLikeFn(CI, *ACI);
          }
        }
      }
    }
  }

  ChangeStatus Changed = A.run();

  AA::AccessRangeTy Range(AA::RangeTy::getUnknown(),
                          AA::RangeTy::getUnknownSize());
  AA::AccessRangeListTy RangeList(Range);

  for (auto [Obj, Size, AAPI] : WorkList) {
    Cache.insertSanitizedObject(Obj, Size);

    if (!AAPI->getState().isValidState())
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
      auto *AAUO = InstToAAUOMap.lookup(Acc.getRemoteInst());
      if (!AAUO || !AAUO->getState().isValidState() ||
          !AAUO->forallUnderlyingObjects([ObjPtr](Value &V) {
            return &V == ObjPtr || isa<UndefValue>(V);
          }))
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
  for (auto *U : FC->users()) {
    auto *CI = cast<CallInst>(U);
    CB(*CI);
  }
}

bool LightSanImpl::instrument() {
  bool Changed = false;

#if 0
  for (auto &Fn : M) {
    if (Fn.isDeclaration())
      continue;
    auto &LI = FAM.getResult<LoopAnalysis>(Fn);
    for (auto *L : LI)
      Changed |= hoistLoopLoads(*L);
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

  Changed |= collectAttributorInfo(A, M, Cache);

  InstrumentorPass IP(&IConf);
  auto PA = IP.run(M, MAM);
  if (!PA.areAllPreserved())
    Changed = true;

  auto CheckForRequiredRanged = [&](StringRef Name) {
    auto *FC = M.getFunction(IConf.getRTName("pre_", Name));
    if (!FC)
      return;
    for (auto *U : FC->users()) {
      auto *CI = cast<CallInst>(U);
      auto *BB = CI->getParent();
      auto *Fn = BB->getParent();
      auto &LI = FAM.getResult<LoopAnalysis>(*Fn);
      auto *L = LI.getLoopFor(BB);
      if (!L)
        continue;
      auto &DT = FAM.getResult<DominatorTreeAnalysis>(*Fn);
      // TODO: we need to use the must-be-executed stuff in
      // llvm/include/llvm/Analysis/MustExecute.h to avoid weird exits
      auto *ExitBB = L->getExitBlock();
      if (!ExitBB || !(BB == ExitBB || DT.dominates(BB, ExitBB)))
        continue;
      auto *LVRI = CI->getArgOperand(2);
      if (isa<ConstantPointerNull>(LVRI))
        continue;
      auto *LVRICI = cast<CallInst>(LVRI);
      auto MaxOffset =
          cast<ConstantInt>(LVRICI->getArgOperand(2))->getSExtValue();
      auto IsExecuted =
          cast<ConstantInt>(LVRICI->getArgOperand(6))->getSExtValue();
      if (!IsExecuted) {
        auto Offset = cast<ConstantInt>(CI->getArgOperand(3))->getSExtValue();
        if (MaxOffset > Offset)
          continue;
        LVRICI->setArgOperand(
            6, ConstantInt::get(Type::getInt8Ty(M.getContext()), 1));
      }
      CI->setArgOperand(6,
                        ConstantInt::get(Type::getInt8Ty(M.getContext()), 1));
    }
  };
  CheckForRequiredRanged("load");
  CheckForRequiredRanged("store");

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

  return Changed;
}

LightSanInstrumentationConfig::LightSanInstrumentationConfig(LightSanImpl &Impl)
    : InstrumentationConfig(), LSI(Impl) {
  ReadConfig = false;
  RuntimePrefix->setString(LightSanRuntimePrefix);
  RuntimeStubsFile->setString("");
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
      if (!cast<ConstantInt>(CI->getArgOperand(CI->arg_size() - 1))->isZero()) {
        auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);
        AllocaInst *AI;
        {
          IRBuilderBase::InsertPointGuard IPG(IIRB.IRB);
          IIRB.IRB.SetInsertPointPastAllocas(CI->getFunction());
          AI = IIRB.IRB.CreateAlloca(IIRB.PtrTy);
        }
        IIRB.IRB.CreateStore(CI, AI);
        LSIConf.EscapedAllocas.push_back(AI);
      }
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

  static void populate(InstrumentationConfig &IConf,
                       InstrumentorIRBuilderTy &IIRB) {
    auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);
    auto *EAIO = IConf.allocate<ExtendedGlobalIO>();
    EAIO->CB = [&](Value &V) {
      if (LSIConf.AIC->isObjectSafe(&V))
        return false;
      auto &GV = cast<GlobalVariable>(V);
      return GV.getValueType()->isSized() && !GV.hasExternalWeakLinkage() &&
             GV.hasInitializer() && !GV.isInterposable();
    };
    EAIO->init(IConf, IIRB);
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
               IRTArg::REPLACABLE_CUSTOM, getObjectSizePtr, setObjectSize));
    IRTArgs.push_back(IRTArg(
        PointerType::getUnqual(Ctx), "encoding_no_ptr",
        "Return the encoding number of the object in question as uint8_t.",
        IRTArg::REPLACABLE_CUSTOM, getEncodingNoPtr, setEncodingNo));
  }

  static Value *getObjectSizePtr(Value &V, Type &Ty,
                                 InstrumentationConfig &IConf,
                                 InstrumentorIRBuilderTy &IIRB) {
    Function *Fn = IIRB.IRB.GetInsertBlock()->getParent();
    auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);
    return LSIConf.getSizeAlloca(*Fn, IIRB, *stripRegisterCall(&V, IConf));
  }
  static Value *setObjectSize(Value &V, Value &NewV,
                              InstrumentationConfig &IConf,
                              InstrumentorIRBuilderTy &IIRB) {
    Function *Fn = IIRB.IRB.GetInsertBlock()->getParent();
    auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);
    auto *ObjSize = IIRB.IRB.CreateLoad(IIRB.Int64Ty, &NewV);
    assert(isa<CallInst>(V) &&
           cast<CallInst>(V).getCalledFunction()->getName() ==
               IConf.getRTName("post_", "base_pointer_info"));
    auto *BasePtr = cast<CallInst>(V).getArgOperand(0);
    auto &EBPI = LSIConf.BasePointerSizeOffsetMap[{BasePtr, Fn}];
    EBPI.ObjectSize = ObjSize;
    auto *SizeAI = LSIConf.TmpToSizeAllocas.lookup(cast<AllocaInst>(&NewV));
    if (SizeAI) {
      IIRB.IRB.CreateStore(ObjSize, SizeAI);
      EBPI.ObjectSizePtr = SizeAI;
    }
    return &V;
  }
  static Value *getEncodingNoPtr(Value &V, Type &Ty,
                                 InstrumentationConfig &IConf,
                                 InstrumentorIRBuilderTy &IIRB) {
    Function *Fn = IIRB.IRB.GetInsertBlock()->getParent();
    return IIRB.getAlloca(Fn, IIRB.Int8Ty);
  }
  static Value *setEncodingNo(Value &V, Value &NewV,
                              InstrumentationConfig &IConf,
                              InstrumentorIRBuilderTy &IIRB) {
    Function *Fn = IIRB.IRB.GetInsertBlock()->getParent();
    auto *EncodingNo = IIRB.IRB.CreateLoad(IIRB.Int8Ty, &NewV);
    auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);
    auto *BasePtr = cast<CallInst>(V).getArgOperand(0);
    LSIConf.BasePointerSizeOffsetMap[{BasePtr, Fn}].EncodingNo = EncodingNo;
    return &V;
  }
  static void populate(InstrumentationConfig &IConf,
                       InstrumentorIRBuilderTy &IIRB) {
    auto *EVPIO = IConf.allocate<ExtendedBasePointerIO>();
    EVPIO->init(IConf, IIRB.Ctx);
  }
};

struct ExtendedLoopValueRangeIO : public LoopValueRangeIO {
  ExtendedLoopValueRangeIO() : LoopValueRangeIO() {}
  virtual ~ExtendedLoopValueRangeIO() {};

  StringRef getName() const override { return "loop_value_range"; }

  void init(InstrumentationConfig &IConf, InstrumentorIRBuilderTy &IIRB) {
    LoopValueRangeIO::init(IConf, IIRB);

    IRTArgs.push_back(IRTArg(IIRB.PtrTy, "base_pointer_info",
                             "The runtime provided base pointer info.",
                             IRTArg::NONE, getBasePointerInfo));
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

  static Value *getBasePointerInfo(Value &V, Type &Ty,
                                   InstrumentationConfig &IConf,
                                   InstrumentorIRBuilderTy &IIRB) {
    return IConf.getBasePointerInfo(V, IIRB);
  }
  static Value *getObjectSize(Value &V, Type &Ty, InstrumentationConfig &IConf,
                              InstrumentorIRBuilderTy &IIRB) {
    auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);
    return LSIConf.getBasePointerObjectSize(V, IIRB);
  }
  static Value *getEncodingNo(Value &V, Type &Ty, InstrumentationConfig &IConf,
                              InstrumentorIRBuilderTy &IIRB) {
    auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);
    Function *Fn = IIRB.IRB.GetInsertBlock()->getParent();
    return LSIConf.getBasePointerEncodingNo(V, *Fn);
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
    LICConfig.set(LoadIO::PassValueSize);
    LICConfig.set(LoadIO::PassLoopValueRangeInfo);
    LoadIO::init(IConf, IIRB, &LICConfig);

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

  static Value *getObjectSize(Value &V, Type &Ty, InstrumentationConfig &IConf,
                              InstrumentorIRBuilderTy &IIRB) {
    auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);
    return LSIConf.getBasePointerObjectSize(
        *cast<LoadInst>(V).getPointerOperand(), IIRB);
  }
  static Value *getEncodingNo(Value &V, Type &Ty, InstrumentationConfig &IConf,
                              InstrumentorIRBuilderTy &IIRB) {
    auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);
    auto &LI = cast<LoadInst>(V);
    if (auto *EncNo = LSIConf.getStaticBasePointerEncodingNo(LI))
      return EncNo;
    Function *Fn = IIRB.IRB.GetInsertBlock()->getParent();
    return LSIConf.getBasePointerEncodingNo(*LI.getPointerOperand(), *Fn);
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
    SICConfig.set(StoreIO::PassStoredValueSize);
    SICConfig.set(StoreIO::PassLoopValueRangeInfo);
    StoreIO::init(IConf, IIRB, &SICConfig);

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

  static Value *getObjectSize(Value &V, Type &Ty, InstrumentationConfig &IConf,
                              InstrumentorIRBuilderTy &IIRB) {
    auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);
    return LSIConf.getBasePointerObjectSize(
        *cast<StoreInst>(V).getPointerOperand(), IIRB);
  }
  static Value *getEncodingNo(Value &V, Type &Ty, InstrumentationConfig &IConf,
                              InstrumentorIRBuilderTy &IIRB) {
    auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);
    auto &SI = cast<StoreInst>(V);
    if (auto *EncNo = LSIConf.getStaticBasePointerEncodingNo(SI))
      return EncNo;
    Function *Fn = IIRB.IRB.GetInsertBlock()->getParent();
    return LSIConf.getBasePointerEncodingNo(*SI.getPointerOperand(), *Fn);
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
  UnreachableIO::populate(*this, IIRB.Ctx);
  ExtendedBasePointerIO::populate(*this, IIRB);
  ExtendedStoreIO::populate(*this, IIRB);
  ExtendedLoadIO::populate(*this, IIRB);
  ExtendedLoopValueRangeIO::populate(*this, IIRB);
  ExtendedAllocaIO::populate(*this, IIRB);
  ExtendedFunctionIO::populate(*this, IIRB);
  ExtendedGlobalIO::populate(*this, IIRB);
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

  CallIO::ConfigTy PostCICConfig(/*Enable=*/false);
  PostCICConfig.set(CallIO::PassAllocationInfo);
  PostCICConfig.set(CallIO::PassNumParameters);
  PostCICConfig.set(CallIO::PassParameters);
  PostCICConfig.set(CallIO::PassReturnedValue);
  auto *PostCIC = InstrumentationConfig::allocate<CallIO>(/*IsPRE=*/false);
  PostCIC->CB = [&](Value &V) {
    if (AIC->isObjectSafe(&V))
      return false;
    auto &CI = cast<CallInst>(V);
    auto &TLI = IIRB.TLIGetter(*CI.getFunction());
    auto ACI = getAllocationCallInfo(&CI, &TLI);
    // TODO: check for escaping -> temporal checks
    return !!ACI;
  };
  PostCIC->init(*this, IIRB.Ctx, &PostCICConfig);
}

} // namespace

PreservedAnalyses LightSanPass::run(Module &M, AnalysisManager<Module> &MAM) {
  if (!M.getModuleFlag("sanitize_obj"))
    return PreservedAnalyses::all();

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

  return PreservedAnalyses::none();
}
