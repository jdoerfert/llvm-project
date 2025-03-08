//===-- LightSan.cpp - Sanitization instrumentation pass ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/LightSan.h"

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/Instrumentation/Instrumentor.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <functional>

using namespace llvm;
using namespace llvm::instrumentor;

#define DEBUG_TYPE "lightsan"

static constexpr char LightSanRuntimePrefix[] = "__objsan_";

namespace {

struct LightSanImpl;

struct LightSanInstrumentationConfig : public InstrumentationConfig {

  LightSanInstrumentationConfig(LightSanImpl &LSI);
  virtual ~LightSanInstrumentationConfig() {}

  void populate(InstrumentorIRBuilderTy &IRB) override;

  struct ExtendedBasePointerInfo {
    Value *ObjectSize = nullptr;
    Value *EncodingNo = nullptr;
    Value *Offset = nullptr;
  };

  Value *getUnderlyingObject(Value *Ptr) {
    auto *NewVPtr = const_cast<Value *>(getUnderlyingObjectAggressive(Ptr));
    while (NewVPtr != Ptr) {
      Ptr = NewVPtr;
      NewVPtr = const_cast<Value *>(getUnderlyingObjectAggressive(Ptr));
    }
    return Ptr;
  }

  DenseMap<std::pair<Value *, Function *>, ExtendedBasePointerInfo>
      BasePointerSizeOffsetMap;
  Value *getBasePointerObjectSize(Value &V, InstrumentorIRBuilderTy &IIRB) {
    Function *Fn = IIRB.IRB.GetInsertBlock()->getParent();
    Value *Ptr = getUnderlyingObject(&V);
    return BasePointerSizeOffsetMap[{Ptr, Fn}].ObjectSize;
  }
  Value *getBasePointerEncodingNo(Value &V, InstrumentorIRBuilderTy &IIRB) {
    Function *Fn = IIRB.IRB.GetInsertBlock()->getParent();
    Value *Ptr = getUnderlyingObject(&V);
    return BasePointerSizeOffsetMap[{Ptr, Fn}].EncodingNo;
  }
  Value *getBasePointerOffset(Value &V, InstrumentorIRBuilderTy &IIRB) {
    Function *Fn = IIRB.IRB.GetInsertBlock()->getParent();
    Value *Ptr = getUnderlyingObject(&V);
    return BasePointerSizeOffsetMap[{Ptr, Fn}].Offset;
  }

  LightSanImpl &LSI;
};

struct LightSanImpl {
  LightSanImpl(Module &M, ModuleAnalysisManager &MAM)
      : M(M), MAM(MAM), IConf(*this) {}

  bool instrument();

  bool shouldInstrumentFunction(Function &Fn);
  bool shouldInstrumentCall(CallInst &CI, InstrumentorIRBuilderTy &IIRB);
  bool shouldInstrumentLoad(LoadInst &LI, InstrumentorIRBuilderTy &IIRB);
  bool shouldInstrumentStore(StoreInst &SI, InstrumentorIRBuilderTy &IIRB);
  bool shouldInstrumentAlloca(AllocaInst &AI, InstrumentorIRBuilderTy &IIRB);

private:
  Module &M;
  ModuleAnalysisManager &MAM;
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

bool LightSanImpl::shouldInstrumentAlloca(AllocaInst &AI,
                                          InstrumentorIRBuilderTy &IIRB) {
  return true;
}

bool LightSanImpl::shouldInstrumentFunction(Function &Fn) {
  return Fn.getName() == "main";
}

bool LightSanImpl::shouldInstrumentCall(CallInst &CI,
                                        InstrumentorIRBuilderTy &IIRB) {
  Function *CalledFn = CI.getCalledFunction();
  if (!CalledFn)
    return true;
  if (!CalledFn->isDeclaration())
    return false;
  FunctionType *CalledFnTy = CalledFn->getFunctionType();
  if (!CalledFnTy->getReturnType()->isPtrOrPtrVectorTy() &&
      none_of(CalledFnTy->params(),
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

bool LightSanImpl::instrument() {
  bool Changed = false;

  InstrumentorPass IP(&IConf);

  auto PA = IP.run(M, MAM);
  if (!PA.areAllPreserved())
    Changed = true;

  auto &FAM = MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

  auto CheckForRequiredRanged = [&](StringRef Name) {
    auto *FC = M.getFunction(IConf.getRTName("pre_", Name));
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
          cast<ConstantInt>(LVRICI->getArgOperand(7))->getSExtValue();
      if (!IsExecuted) {
        auto Offset = cast<ConstantInt>(CI->getArgOperand(3))->getSExtValue();
        if (MaxOffset > Offset)
          continue;
        LVRICI->setArgOperand(
            7, ConstantInt::get(Type::getInt8Ty(M.getContext()), 1));
      }
      CI->setArgOperand(7,
                        ConstantInt::get(Type::getInt8Ty(M.getContext()), 1));
    }
  };
  CheckForRequiredRanged("load");
  CheckForRequiredRanged("store");

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
    // TODO: Be smarter about this, e.g., AAPointerInfo.
    bool MayEscape = !all_of(V.uses(), [](const Use &U) {
      return isa<LoadInst>(U.getUser()) ||
             (isa<StoreInst>(U.getUser()) &&
              U.getOperandNo() != StoreInst::getPointerOperandIndex());
    });
    return ConstantInt::get(&Ty, MayEscape);
  }

  static void populate(InstrumentationConfig &IConf,
                       InstrumentorIRBuilderTy &IIRB) {
    auto *EAIO = IConf.allocate<ExtendedAllocaIO>(/*IsPRE*/ false);
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
    IRTArgs.push_back(IRTArg(PointerType::getUnqual(Ctx), "offset_ptr",
                             "Return the offset of the pointer as int64_t.",
                             IRTArg::REPLACABLE_CUSTOM, getOffsetPtr,
                             setOffset));
  }

  static Value *getObjectSizePtr(Value &V, Type &Ty,
                                 InstrumentationConfig &IConf,
                                 InstrumentorIRBuilderTy &IIRB) {
    Function *Fn = IIRB.IRB.GetInsertBlock()->getParent();
    return IIRB.getAlloca(Fn, IIRB.Int64Ty);
  }
  static Value *setObjectSize(Value &V, Value &NewV,
                              InstrumentationConfig &IConf,
                              InstrumentorIRBuilderTy &IIRB) {
    Function *Fn = IIRB.IRB.GetInsertBlock()->getParent();
    auto *ObjectSize = IIRB.IRB.CreateLoad(IIRB.Int64Ty, &NewV);
    auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);
    auto *BasePtr = cast<CallInst>(V).getArgOperand(0);
    LSIConf.BasePointerSizeOffsetMap[{BasePtr, Fn}].ObjectSize = ObjectSize;
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
  static Value *getOffsetPtr(Value &V, Type &Ty, InstrumentationConfig &IConf,
                             InstrumentorIRBuilderTy &IIRB) {
    Function *Fn = IIRB.IRB.GetInsertBlock()->getParent();
    return IIRB.getAlloca(Fn, IIRB.Int64Ty);
  }
  static Value *setOffset(Value &V, Value &NewV, InstrumentationConfig &IConf,
                          InstrumentorIRBuilderTy &IIRB) {
    Function *Fn = IIRB.IRB.GetInsertBlock()->getParent();
    auto *Offset = IIRB.IRB.CreateLoad(IIRB.Int64Ty, &NewV);
    auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);
    auto *BasePtr = cast<CallInst>(V).getArgOperand(0);
    LSIConf.BasePointerSizeOffsetMap[{BasePtr, Fn}].Offset = Offset;
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
    IRTArgs.push_back(
        IRTArg(IIRB.Int64Ty, "object_offset",
               "The offset of the underlying base pointer in the object.",
               IRTArg::NONE, getBaseOffset));
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
  static Value *getBaseOffset(Value &V, Type &Ty, InstrumentationConfig &IConf,
                              InstrumentorIRBuilderTy &IIRB) {
    auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);
    return LSIConf.getBasePointerOffset(V, IIRB);
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
    LICConfig.set(LoadIO::PassValueSize);
    LICConfig.set(LoadIO::PassLoopValueRangeInfo);
    LoadIO::init(IConf, IIRB, &LICConfig);

    IRTArgs.push_back(IRTArg(IIRB.Int64Ty, "object_size",
                             "The size of the underlying object.", IRTArg::NONE,
                             getObjectSize));
    IRTArgs.push_back(
        IRTArg(IIRB.Int64Ty, "object_offset",
               "The offset of the underlying base pointer in the object.",
               IRTArg::NONE, getBaseOffset));
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
  static Value *getBaseOffset(Value &V, Type &Ty, InstrumentationConfig &IConf,
                              InstrumentorIRBuilderTy &IIRB) {
    auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);
    return LSIConf.getBasePointerOffset(*cast<LoadInst>(V).getPointerOperand(),
                                        IIRB);
  }
  static Value *getEncodingNo(Value &V, Type &Ty, InstrumentationConfig &IConf,
                              InstrumentorIRBuilderTy &IIRB) {
    auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);
    return LSIConf.getBasePointerEncodingNo(
        *cast<LoadInst>(V).getPointerOperand(), IIRB);
  }
  static Value *getWasChecked(Value &V, Type &Ty, InstrumentationConfig &IConf,
                              InstrumentorIRBuilderTy &IIRB) {
    return ConstantInt::get(&Ty, 0);
  }

  static void populate(InstrumentationConfig &IConf,
                       InstrumentorIRBuilderTy &IIRB) {
    auto *ESIO = IConf.allocate<ExtendedLoadIO>(/*IsPRE*/ true);
    ESIO->HoistKind = HOIST_IN_BLOCK;
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
    IRTArgs.push_back(
        IRTArg(IIRB.Int64Ty, "object_offset",
               "The offset of the underlying base pointer in the object.",
               IRTArg::NONE, getBaseOffset));
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
  static Value *getBaseOffset(Value &V, Type &Ty, InstrumentationConfig &IConf,
                              InstrumentorIRBuilderTy &IIRB) {
    auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);
    return LSIConf.getBasePointerOffset(*cast<StoreInst>(V).getPointerOperand(),
                                        IIRB);
  }
  static Value *getEncodingNo(Value &V, Type &Ty, InstrumentationConfig &IConf,
                              InstrumentorIRBuilderTy &IIRB) {
    auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);
    return LSIConf.getBasePointerEncodingNo(
        *cast<StoreInst>(V).getPointerOperand(), IIRB);
  }
  static Value *getWasChecked(Value &V, Type &Ty, InstrumentationConfig &IConf,
                              InstrumentorIRBuilderTy &IIRB) {
    return ConstantInt::get(&Ty, 0);
  }

  static void populate(InstrumentationConfig &IConf,
                       InstrumentorIRBuilderTy &IIRB) {
    auto *ESIO = IConf.allocate<ExtendedStoreIO>(/*IsPRE*/ true);
    ESIO->HoistKind = HOIST_IN_BLOCK;
    ESIO->init(IConf, IIRB);
  }
};

void LightSanInstrumentationConfig::populate(InstrumentorIRBuilderTy &IIRB) {
  UnreachableIO::populate(*this, IIRB.Ctx);
  ExtendedBasePointerIO::populate(*this, IIRB);
  ExtendedStoreIO::populate(*this, IIRB);
  ExtendedLoadIO::populate(*this, IIRB);
  ExtendedLoopValueRangeIO::populate(*this, IIRB);
  //  ModuleIO::populate(*this, IIRB.Ctx);
  //  GlobalIO::populate(*this, IIRB.Ctx);

  AllocaIO::ConfigTy AICConfig(/*Enable=*/false);
  AICConfig.set(AllocaIO::PassAddress);
  AICConfig.set(AllocaIO::ReplaceAddress);
  AICConfig.set(AllocaIO::PassSize);
  auto *AIC = InstrumentationConfig::allocate<AllocaIO>(/*IsPRE=*/false);
  AIC->CB = [&](Value &V) {
    return LSI.shouldInstrumentAlloca(cast<AllocaInst>(V), IIRB);
  };
  AIC->init(*this, IIRB.Ctx, &AICConfig);

  CallIO::ConfigTy CICConfig(/*Enable=*/false);
  CICConfig.set(CallIO::PassIntrinsicId);
  CICConfig.set(CallIO::PassAllocationInfo);
  CICConfig.set(CallIO::PassNumParameters);
  CICConfig.set(CallIO::PassParameters);
  CICConfig.set(CallIO::PassIsDefinition);
  CICConfig.ArgFilter = [&](Use &Op) {
    auto *CI = cast<CallInst>(Op.getUser());
    auto &TLI = IIRB.TLIGetter(*CI->getFunction());
    auto ACI = getAllocationCallInfo(CI, &TLI);
    return Op->getType()->isPointerTy() || ACI;
  };
  auto *CIC = InstrumentationConfig::allocate<CallIO>(/*IsPRE=*/true);
  CIC->CB = [&](Value &V) {
    return LSI.shouldInstrumentCall(cast<CallInst>(V), IIRB);
  };
  CIC->init(*this, IIRB.Ctx, &CICConfig);

  //  FunctionIO::ConfigTy FICConfig(/*Enable=*/false);
  //  FICConfig.set(FunctionIO::PassName);
  //  FICConfig.set(FunctionIO::PassNumArguments);
  //  FICConfig.set(FunctionIO::PassArguments);
  //  FICConfig.set(FunctionIO::ReplaceArguments);
  //  auto *FIC = InstrumentationConfig::allocate<FunctionIO>();
  //  FIC->CB = [&](Value &V) {
  //    return LSI.shouldInstrumentFunction(cast<Function>(V));
  //  };
  //  FIC->init(*this, IIRB.Ctx, &FICConfig);
}

} // namespace

PreservedAnalyses LightSanPass::run(Module &M, AnalysisManager<Module> &MAM) {
  LightSanImpl Impl(M, MAM);

  bool Changed = Impl.instrument();
  if (!Changed)
    return PreservedAnalyses::all();

  if (verifyModule(M))
    M.dump();
  assert(!verifyModule(M, &errs()));

  return PreservedAnalyses::none();
}
