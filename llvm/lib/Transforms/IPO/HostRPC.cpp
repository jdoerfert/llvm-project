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

#include "llvm/Transforms/IPO/HostRPC.h"

#include "llvm/ADT/EnumeratedArray.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/Frontend/OpenMP/OMPDeviceConstants.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/IPO/Attributor.h"

#include <cstdint>

#define DEBUG_TYPE "host-rpc"

using namespace llvm;

using ArgType = llvm::omp::OMPTgtHostRPCArgType;

static cl::opt<bool>
    UseDummyHostModule("host-rpc-use-dummy-host-module", cl::init(false),
                       cl::Hidden,
                       cl::desc("Use dummy host module if there no host module "
                                "attached to the device module"));

namespace {

enum class HostRPCRuntimeFunction {
#define __OMPRTL_HOST_RPC(_ENUM) OMPRTL_##_ENUM
  __OMPRTL_HOST_RPC(__kmpc_host_rpc_get_desc),
  __OMPRTL_HOST_RPC(__kmpc_host_rpc_add_arg),
  __OMPRTL_HOST_RPC(__kmpc_host_rpc_get_arg),
  __OMPRTL_HOST_RPC(__kmpc_host_rpc_send_and_wait),
  __OMPRTL_HOST_RPC(__kmpc_host_rpc_set_ret_val),
  __OMPRTL_HOST_RPC(__kmpc_host_rpc_invoke_host_wrapper),
  __OMPRTL_HOST_RPC(__last),
#undef __OMPRTL_HOST_RPC
};

#define __OMPRTL_HOST_RPC(_ENUM)                                               \
  auto OMPRTL_##_ENUM = HostRPCRuntimeFunction::OMPRTL_##_ENUM;
__OMPRTL_HOST_RPC(__kmpc_host_rpc_get_desc)
__OMPRTL_HOST_RPC(__kmpc_host_rpc_add_arg)
__OMPRTL_HOST_RPC(__kmpc_host_rpc_get_arg)
__OMPRTL_HOST_RPC(__kmpc_host_rpc_send_and_wait)
__OMPRTL_HOST_RPC(__kmpc_host_rpc_set_ret_val)
__OMPRTL_HOST_RPC(__kmpc_host_rpc_invoke_host_wrapper)
#undef __OMPRTL_HOST_RPC

// TODO: Remove those functions implemented in device runtime.
static constexpr const char *InternalPrefix[] = {
    "__kmp", "llvm.",        "nvm.",
    "omp_",  "vprintf",      "malloc",
    "free",  "__keep_alive", "__llvm_omp_vprintf"};

bool isInternalFunction(Function &F) {
  auto Name = F.getName();

  for (auto *P : InternalPrefix)
    if (Name.startswith(P))
      return true;

  return false;
}

std::string typeToString(Type *T) {
  if (T->is16bitFPTy())
    return "f16";
  if (T->isFloatTy())
    return "f32";
  if (T->isDoubleTy())
    return "f64";
  if (T->isPointerTy())
    return "ptr";
  if (T->isStructTy())
    return std::string(T->getStructName());
  if (T->isIntegerTy())
    return "i" + std::to_string(T->getIntegerBitWidth());

  LLVM_DEBUG(dbgs() << "[HostRPC] unknown type " << *T
                    << "  for typeToString.\n";);

  llvm_unreachable("unknown type");
}

class HostRPC {
  /// LLVM context instance
  LLVMContext &Context;

  /// Device module.
  Module &M;
  /// Host module
  Module &HM;

  /// Data layout of the device module.
  DataLayout DL;

  IRBuilder<> Builder;

  /// External functions we are operating on.
  SmallSetVector<Function *, 8> FunctionWorkList;

  /// Attributor instance.
  Attributor &A;

  // Types
  Type *Int8PtrTy;
  Type *VoidTy;
  Type *Int32Ty;
  Type *Int64Ty;
  StructType *ArgInfoTy;
  // Values
  Constant *NullPtr;
  Constant *NullInt64;

  struct CallSiteInfo {
    CallInst *CI = nullptr;
    SmallVector<Type *> Params;
  };

  struct HostRPCArgInfo {
    Value *BasePtr = nullptr;
    Constant *Type = nullptr;
    Value *Size = nullptr;
  };

  ///
  SmallVector<Function *> HostEntryTable;

  EnumeratedArray<Function *, HostRPCRuntimeFunction,
                  HostRPCRuntimeFunction::OMPRTL___last>
      RFIs;

  SmallVector<std::pair<CallInst *, CallInst *>> CallInstMap;

  Constant *getConstantInt64(uint64_t Val) {
    return ConstantInt::get(Int64Ty, Val);
  }

  static std::string getWrapperFunctionName(Function *F, CallSiteInfo &CSI) {
    std::string Name = "__kmpc_host_rpc_wrapper_" + std::string(F->getName());
    if (!F->isVarArg())
      return Name;

    for (unsigned I = F->getFunctionType()->getNumParams();
         I < CSI.Params.size(); ++I) {
      Name.push_back('_');
      Name.append(typeToString(CSI.Params[I]));
    }

    return Name;
  }

  void registerAAs();

  Value *convertToInt64Ty(Value *V);

  Value *convertFromInt64TyTo(Value *V, Type *TargetTy);

  Constant *convertToInt64Ty(Constant *C);

  Constant *convertFromInt64TyTo(Constant *C, Type *T);

  // int device_wrapper(call_no, arg_info, ...) {
  //   void *desc = __kmpc_host_rpc_get_desc(call_no, num_args, arg_info);
  //   __kmpc_host_rpc_add_arg(desc, arg1, sizeof(arg1));
  //   __kmpc_host_rpc_add_arg(desc, arg2, sizeof(arg2));
  //   ...
  //   int r = (int)__kmpc_host_rpc_send_and_wait(desc);
  //   return r;
  // }
  Function *getDeviceWrapperFunction(StringRef WrapperName, Function *F,
                                     CallSiteInfo &CSI);

  // void host_wrapper(desc) {
  //   int arg1 = (int)__kmpc_host_rpc_get_arg(desc, 0);
  //   float arg2 = (float)__kmpc_host_rpc_get_arg(desc, 1);
  //   char *arg3 = (char *)__kmpc_host_rpc_get_arg(desc, 2);
  //   ...
  //   int r = actual_call(arg1, arg2, arg3, ...);
  //   __kmpc_host_rpc_set_ret_val(ptr(desc, (int64_t)r);
  // }
  Function *getHostWrapperFunction(StringRef WrapperName, Function *F,
                                   CallSiteInfo &CSI);

  bool rewriteWithHostRPC(Function *F);

  void emitHostWrapperInvoker();

  bool recollectInformation();

public:
  HostRPC(Module &DeviceModule, Module &HostModule, Attributor &A)
      : Context(DeviceModule.getContext()), M(DeviceModule), HM(HostModule),
        DL(M.getDataLayout()), Builder(Context), A(A) {
    assert(&M.getContext() == &HM.getContext() &&
           "device and host modules have different context");

#define __OMP_TYPE(TYPE) TYPE = Type::get##TYPE(Context)
    __OMP_TYPE(Int8PtrTy);
    __OMP_TYPE(VoidTy);
    __OMP_TYPE(Int32Ty);
    __OMP_TYPE(Int64Ty);
#undef __OMP_TYPE

    NullPtr = ConstantInt::getNullValue(Int8PtrTy);
    NullInt64 = ConstantInt::getNullValue(Int64Ty);

#define __OMP_RTL(_ENUM, MOD, VARARG, RETTY, ...)                              \
  {                                                                            \
    SmallVector<Type *> Params{__VA_ARGS__};                                   \
    FunctionType *FT = FunctionType::get(RETTY, Params, VARARG);               \
    Function *F = (MOD).getFunction(#_ENUM);                                   \
    if (!F)                                                                    \
      F = Function::Create(FT, GlobalValue::LinkageTypes::ExternalLinkage,     \
                           #_ENUM, (MOD));                                     \
    RFIs[OMPRTL_##_ENUM] = F;                                                  \
  }
    __OMP_RTL(__kmpc_host_rpc_get_desc, M, false, Int8PtrTy, Int32Ty, Int32Ty,
              Int8PtrTy)
    __OMP_RTL(__kmpc_host_rpc_add_arg, M, false, VoidTy, Int8PtrTy, Int64Ty,
              Int32Ty)
    __OMP_RTL(__kmpc_host_rpc_send_and_wait, M, false, Int64Ty, Int8PtrTy)
    __OMP_RTL(__kmpc_host_rpc_get_arg, HM, false, Int64Ty, Int8PtrTy, Int32Ty)
    __OMP_RTL(__kmpc_host_rpc_set_ret_val, HM, false, VoidTy, Int8PtrTy,
              Int64Ty)
    __OMP_RTL(__kmpc_host_rpc_invoke_host_wrapper, HM, false, VoidTy, Int32Ty,
              Int8PtrTy)
#undef __OMP_RTL

    ArgInfoTy = StructType::create({Int64Ty, Int64Ty, Int64Ty, Int8PtrTy},
                                   "struct.arg_info_t");
  }

  bool run();
};

Value *HostRPC::convertToInt64Ty(Value *V) {
  if (auto *C = dyn_cast<Constant>(V))
    return convertToInt64Ty(C);

  Type *T = V->getType();

  if (T == Int64Ty)
    return V;

  if (T->isPointerTy())
    return Builder.CreatePtrToInt(V, Int64Ty);

  if (T->isIntegerTy())
    return Builder.CreateIntCast(V, Int64Ty, /* isSigned */ true);

  if (T->isFloatingPointTy()) {
    V = Builder.CreateBitCast(
        V, Type::getIntNTy(V->getContext(), T->getScalarSizeInBits()));
    return Builder.CreateIntCast(V, Int64Ty, /* isSigned */ true);
  }

  llvm_unreachable("unknown cast to int64_t");
}

Value *HostRPC::convertFromInt64TyTo(Value *V, Type *T) {
  if (auto *C = dyn_cast<Constant>(V))
    return convertFromInt64TyTo(C, T);

  if (T == Int64Ty)
    return V;

  if (T->isPointerTy())
    return Builder.CreateIntToPtr(V, T);

  if (T->isIntegerTy())
    return Builder.CreateIntCast(V, T, /* isSigned */ true);

  if (T->isFloatingPointTy()) {
    V = Builder.CreateIntCast(
        V, Type::getIntNTy(V->getContext(), T->getScalarSizeInBits()),
        /* isSigned */ true);
    return Builder.CreateBitCast(V, T);
  }

  llvm_unreachable("unknown cast from int64_t");
}

Constant *HostRPC::convertToInt64Ty(Constant *C) {
  Type *T = C->getType();

  if (T == Int64Ty)
    return C;

  if (T->isPointerTy())
    return ConstantExpr::getPtrToInt(C, Int64Ty);

  if (T->isIntegerTy())
    return ConstantExpr::getIntegerCast(C, Int64Ty, /* isSigned */ true);

  if (T->isFloatingPointTy()) {
    C = ConstantExpr::getBitCast(
        C, Type::getIntNTy(C->getContext(), T->getScalarSizeInBits()));
    return ConstantExpr::getIntegerCast(C, Int64Ty, /* isSigned */ true);
  }

  llvm_unreachable("unknown cast to int64_t");
}

Constant *HostRPC::convertFromInt64TyTo(Constant *C, Type *T) {
  if (T == Int64Ty)
    return C;

  if (T->isPointerTy())
    return ConstantExpr::getIntToPtr(C, T);

  if (T->isIntegerTy())
    return ConstantExpr::getIntegerCast(C, T, /* isSigned */ true);

  if (T->isFloatingPointTy()) {
    C = ConstantExpr::getIntegerCast(
        C, Type::getIntNTy(C->getContext(), T->getScalarSizeInBits()),
        /* isSigned */ true);
    return ConstantExpr::getBitCast(C, T);
  }

  llvm_unreachable("unknown cast from int64_t");
}

void HostRPC::registerAAs() {
  for (auto *F : FunctionWorkList)
    for (User *U : F->users()) {
      auto *CI = dyn_cast<CallInst>(U);
      if (!CI)
        continue;

      for (unsigned I = 0; I < CI->arg_size(); ++I) {
        Value *Operand = CI->getArgOperand(I);
        if (!Operand->getType()->isPointerTy())
          continue;
        auto &AA = A.getOrCreateAAFor<AAUnderlyingObjects>(
            IRPosition::callsite_argument(*CI, I),
            /* QueryingAA */ nullptr, DepClassTy::NONE);
        (void)AA;
        assert(AA.getState().isValidState() && "AA is invalid when created");
      }
    }
}

bool HostRPC::recollectInformation() {
  FunctionWorkList.clear();

  for (Function &F : M) {
    // If the function is already defined, it definitely does not require RPC.
    if (!F.isDeclaration())
      continue;

    // If it is an internal function, skip it as well.
    if (isInternalFunction(F))
      continue;

    // If there is no use of the function, skip it.
    if (F.use_empty())
      continue;

    FunctionWorkList.insert(&F);
  }

  return !FunctionWorkList.empty();
}

bool HostRPC::run() {
  bool Changed = false;

  if (!recollectInformation())
    return Changed;

  Changed = true;

  // We add a couple of assumptions to those RPC functions such that AAs will
  // not error out because of unknown implementation of those functions.
  for (Function &F : M) {
    if (!F.isDeclaration())
      continue;

    F.addFnAttr(Attribute::NoRecurse);

    for (auto &Arg : F.args())
      if (Arg.getType()->isPointerTy())
        Arg.addAttr(Attribute::NoCapture);

    if (!F.isVarArg())
      continue;

    for (User *U : F.users()) {
      auto *CB = dyn_cast<CallBase>(U);
      if (!CB)
        continue;
      for (unsigned I = F.getFunctionType()->getNumParams(); I < CB->arg_size();
           ++I) {
        Value *Arg = CB->getArgOperand(I);
        if (Arg->getType()->isPointerTy())
          CB->addParamAttr(I, Attribute::NoCapture);
      }
    }
  }

  LLVM_DEBUG(M.dump());

  registerAAs();

  ChangeStatus Status = A.run();
  if (!recollectInformation())
    return Status == ChangeStatus::CHANGED;

  for (Function *F : FunctionWorkList)
    Changed |= rewriteWithHostRPC(F);

  if (!Changed)
    return Changed;

  for (auto Itr = CallInstMap.rbegin(); Itr != CallInstMap.rend(); ++Itr) {
    auto *CI = Itr->first;
    auto *NewCI = Itr->second;
    CI->replaceAllUsesWith(NewCI);
    CI->eraseFromParent();
  }

  for (Function *F : FunctionWorkList)
    if (F->user_empty())
      F->eraseFromParent();

  emitHostWrapperInvoker();

  return Changed;
}

bool HostRPC::rewriteWithHostRPC(Function *F) {
  SmallVector<CallInst *> WorkList;

  for (User *U : F->users()) {
    auto *CI = dyn_cast<CallInst>(U);
    if (!CI)
      continue;
    WorkList.push_back(CI);
  }

  if (WorkList.empty())
    return false;

  for (CallInst *CI : WorkList) {
    CallSiteInfo CSI;
    CSI.CI = CI;
    unsigned NumArgs = CI->arg_size();
    for (unsigned I = 0; I < NumArgs; ++I)
      CSI.Params.push_back(CI->getArgOperand(I)->getType());

    std::string WrapperName = getWrapperFunctionName(F, CSI);
    Function *DeviceWrapperFn = getDeviceWrapperFunction(WrapperName, F, CSI);
    Function *HostWrapperFn = getHostWrapperFunction(WrapperName, F, CSI);

    int32_t WrapperNumber = -1;
    for (unsigned I = 0; I < HostEntryTable.size(); ++I) {
      if (HostEntryTable[I] == HostWrapperFn) {
        WrapperNumber = I;
        break;
      }
    }
    if (WrapperNumber == -1) {
      WrapperNumber = HostEntryTable.size();
      HostEntryTable.push_back(HostWrapperFn);
    }

    auto CheckIfIdentifierPtr = [this](const Value *V) {
      auto *CI = dyn_cast<CallInst>(V);
      if (!CI)
        return false;
      Function *Callee = CI->getCalledFunction();
      if (this->FunctionWorkList.count(Callee))
        return true;
      return Callee->getName().startswith("__kmpc_host_rpc_wrapper_");
    };

    auto CheckIfDynAlloc = [](Value *V) -> CallInst * {
      auto *CI = dyn_cast<CallInst>(V);
      if (!CI)
        return nullptr;
      Function *Callee = CI->getCalledFunction();
      auto Name = Callee->getName();
      if (Name == "malloc" || Name == "__kmpc_alloc_shared")
        return CI;
      return nullptr;
    };

    auto CheckIfStdIO = [](Value *V) -> GlobalVariable * {
      auto *LI = dyn_cast<LoadInst>(V);
      if (!LI)
        return nullptr;
      auto *GV = dyn_cast<GlobalVariable>(LI->getPointerOperand());
      if (!GV)
        return nullptr;
      auto Name = GV->getName();
      if (Name == "stdout" || Name == "stderr" || Name == "stdin")
        return GV;
      return nullptr;
    };

    auto CheckIfGlobalVariable = [](Value *V) {
      if (auto *GV = dyn_cast<GlobalVariable>(V))
        return GV;
      if (auto *LI = dyn_cast<LoadInst>(V))
        if (auto *GV = dyn_cast<GlobalVariable>(LI->getPointerOperand()))
          return GV;
      return static_cast<GlobalVariable *>(nullptr);
    };

    auto CheckIfNullPtr = [](Value *V) {
      if (!V->getType()->isPointerTy())
        return false;
      return V == ConstantInt::getNullValue(V->getType());
    };

    auto HandleDirectUse = [&](Value *Ptr, HostRPCArgInfo &AI,
                               bool IsPointer = false) {
      AI.BasePtr = Ptr;
      AI.Type = getConstantInt64(IsPointer ? ArgType::OMP_HOST_RPC_ARG_PTR
                                           : ArgType::OMP_HOST_RPC_ARG_SCALAR);
      AI.Size = NullInt64;
    };

    SmallVector<SmallVector<HostRPCArgInfo>> ArgInfo;
    bool IsConstantArgInfo = true;

    for (unsigned I = 0; I < CI->arg_size(); ++I) {
      ArgInfo.emplace_back();
      auto &AII = ArgInfo.back();

      Value *Operand = CI->getArgOperand(I);

      // Check if scalar type.
      if (!Operand->getType()->isPointerTy()) {
        AII.emplace_back();
        HandleDirectUse(Operand, AII.back());
        IsConstantArgInfo = IsConstantArgInfo && isa<Constant>(Operand);
        continue;
      }

      if (CheckIfNullPtr(Operand))
        continue;

      auto Pred = [&](Value &Obj) {
        if (CheckIfNullPtr(&Obj))
          return true;

        bool IsConstantArgument = false;
        if (!F->isVarArg() &&
            F->hasParamAttribute(I, Attribute::AttrKind::ReadOnly))
          IsConstantArgument = true;

        HostRPCArgInfo AI;

        if (auto *IO = CheckIfStdIO(&Obj)) {
          HandleDirectUse(IO, AI, /* IsPointer */ true);
        } else if (CheckIfIdentifierPtr(&Obj)) {
          IsConstantArgInfo = IsConstantArgInfo && isa<Constant>(Operand);
          HandleDirectUse(Operand, AI, /* IsPointer */ true);
        } else if (auto *GV = CheckIfGlobalVariable(&Obj)) {
          AI.BasePtr = GV;
          AI.Size = getConstantInt64(DL.getTypeStoreSize(GV->getValueType()));
          AI.Type =
              getConstantInt64(GV->isConstant() || IsConstantArgument
                                   ? ArgType::OMP_HOST_RPC_ARG_COPY_TO
                                   : ArgType::OMP_HOST_RPC_ARG_COPY_TOFROM);
        } else if (CheckIfDynAlloc(&Obj)) {
          // We will handle this case at runtime so here we don't do anything.
          return true;
        } else if (isa<AllocaInst>(&Obj)) {
          llvm_unreachable("alloca instruction needs to be handled!");
        } else {
          LLVM_DEBUG({
            dbgs() << "[HostRPC] warning: call site " << *CI << ", operand "
                   << *Operand << ", underlying object " << Obj
                   << " cannot be handled.\n";
          });
          return true;
        }
        AII.push_back(std::move(AI));
        return true;
      };

      auto &AAUO = A.getOrCreateAAFor<AAUnderlyingObjects>(
          IRPosition::callsite_argument(*CI, I), nullptr, DepClassTy::NONE);
      if (!AAUO.forallUnderlyingObjects(Pred))
        llvm_unreachable("internal error");
    }

    // Reset the insert point to the call site.
    Builder.SetInsertPoint(CI);

    Value *ArgInfoVal = nullptr;
    if (!IsConstantArgInfo) {
      ArgInfoVal = Builder.CreateAlloca(Int8PtrTy, getConstantInt64(NumArgs),
                                        "arg_info");
      for (unsigned I = 0; I < NumArgs; ++I) {
        auto &AII = ArgInfo[I];
        Value *Next = NullPtr;
        for (auto &AI : AII) {
          Value *AIV = Builder.CreateAlloca(ArgInfoTy);
          Value *AIIArg =
              GetElementPtrInst::Create(Int64Ty, AIV, {getConstantInt64(0)});
          Builder.Insert(AIIArg);
          Builder.CreateStore(convertToInt64Ty(AI.BasePtr), AIIArg);
          Value *AIIType =
              GetElementPtrInst::Create(Int64Ty, AIV, {getConstantInt64(1)});
          Builder.Insert(AIIType);
          Builder.CreateStore(AI.Type, AIIType);
          Value *AIISize =
              GetElementPtrInst::Create(Int64Ty, AIV, {getConstantInt64(2)});
          Builder.Insert(AIISize);
          Builder.CreateStore(AI.Size, AIISize);
          Value *AIINext =
              GetElementPtrInst::Create(Int8PtrTy, AIV, {getConstantInt64(3)});
          Builder.Insert(AIINext);
          Builder.CreateStore(Next, AIINext);
          Next = AIV;
        }
        Value *AIIV = GetElementPtrInst::Create(Int8PtrTy, ArgInfoVal,
                                                {getConstantInt64(I)});
        Builder.Insert(AIIV);
        Builder.CreateStore(Next, AIIV);
      }
    } else {
      SmallVector<Constant *> ArgInfoInitVar;
      for (auto &AII : ArgInfo) {
        Constant *Last = NullPtr;
        for (auto &AI : AII) {
          auto *Arg = cast<Constant>(AI.BasePtr);
          auto *CS =
              ConstantStruct::get(ArgInfoTy, {convertToInt64Ty(Arg), AI.Type,
                                              cast<Constant>(AI.Size), Last});
          auto *GV = new GlobalVariable(
              M, ArgInfoTy, /* isConstant */ true,
              GlobalValue::LinkageTypes::InternalLinkage, CS);
          Last = GV;
        }
        ArgInfoInitVar.push_back(Last);
      }
      Constant *ArgInfoInit = ConstantArray::get(
          ArrayType::get(Int8PtrTy, NumArgs), ArgInfoInitVar);
      ArgInfoVal = new GlobalVariable(
          M, ArrayType::get(Int8PtrTy, NumArgs), /* isConstant */ true,
          GlobalValue::LinkageTypes::InternalLinkage, ArgInfoInit, "arg_info");
    }

    SmallVector<Value *> Args{ConstantInt::get(Int32Ty, WrapperNumber),
                              ArgInfoVal};
    for (Value *Operand : CI->args())
      Args.push_back(Operand);

    CallInst *NewCall = Builder.CreateCall(DeviceWrapperFn, Args);

    CallInstMap.emplace_back(CI, NewCall);
  }

  return true;
}

Function *HostRPC::getDeviceWrapperFunction(StringRef WrapperName, Function *F,
                                            CallSiteInfo &CSI) {
  Function *WrapperFn = M.getFunction(WrapperName);
  if (WrapperFn)
    return WrapperFn;

  // return_type device_wrapper(int32_t call_no, void *arg_info, ...)
  SmallVector<Type *> Params{Int32Ty, Int8PtrTy};
  Params.append(CSI.Params);

  Type *RetTy = F->getReturnType();

  FunctionType *FT = FunctionType::get(RetTy, Params, /*isVarArg*/ false);
  WrapperFn = Function::Create(FT, GlobalValue::LinkageTypes::InternalLinkage,
                               WrapperName, M);

  // Emit the body of the device wrapper
  BasicBlock *EntryBB = BasicBlock::Create(Context, "entry", WrapperFn);
  Builder.SetInsertPoint(EntryBB);

  // skip call_no and arg_info.
  constexpr const unsigned NumArgSkipped = 2;

  Value *Desc = nullptr;
  {
    Function *Fn = RFIs[OMPRTL___kmpc_host_rpc_get_desc];
    Desc = Builder.CreateCall(
        Fn,
        {WrapperFn->getArg(0),
         ConstantInt::get(Int32Ty, WrapperFn->arg_size() - NumArgSkipped),
         WrapperFn->getArg(1)},
        "desc");
  }

  {
    Function *Fn = RFIs[OMPRTL___kmpc_host_rpc_add_arg];
    for (unsigned I = NumArgSkipped; I < WrapperFn->arg_size(); ++I) {
      Value *V = convertToInt64Ty(WrapperFn->getArg(I));
      Builder.CreateCall(
          Fn, {Desc, V, ConstantInt::get(Int32Ty, I - NumArgSkipped)});
    }
  }

  Value *RetVal =
      Builder.CreateCall(RFIs[OMPRTL___kmpc_host_rpc_send_and_wait], {Desc});

  if (RetTy->isVoidTy()) {
    Builder.CreateRetVoid();
    return WrapperFn;
  }

  if (RetTy != RetVal->getType())
    RetVal = convertFromInt64TyTo(RetVal, RetTy);

  Builder.CreateRet(RetVal);

  return WrapperFn;
}

Function *HostRPC::getHostWrapperFunction(StringRef WrapperName, Function *F,
                                          CallSiteInfo &CSI) {
  Function *WrapperFn = HM.getFunction(WrapperName);
  if (WrapperFn)
    return WrapperFn;

  SmallVector<Type *> Params{Int8PtrTy};
  FunctionType *FT = FunctionType::get(VoidTy, Params, /* isVarArg */ false);
  WrapperFn = Function::Create(FT, GlobalValue::LinkageTypes::ExternalLinkage,
                               WrapperName, HM);

  Value *Desc = WrapperFn->getArg(0);

  // Emit the body of the host wrapper
  BasicBlock *EntryBB = BasicBlock::Create(Context, "entry", WrapperFn);
  Builder.SetInsertPoint(EntryBB);

  SmallVector<Value *> Args;
  for (unsigned I = 0; I < CSI.CI->arg_size(); ++I) {
    Value *V = Builder.CreateCall(RFIs[OMPRTL___kmpc_host_rpc_get_arg],
                                  {Desc, ConstantInt::get(Int32Ty, I)});
    Args.push_back(convertFromInt64TyTo(V, CSI.Params[I]));
  }

  // The host callee that will be called eventually by the host wrapper.
  Function *HostCallee = HM.getFunction(F->getName());
  if (!HostCallee)
    HostCallee = Function::Create(F->getFunctionType(), F->getLinkage(),
                                  F->getName(), HM);

  Value *RetVal = Builder.CreateCall(HostCallee, Args);
  if (!RetVal->getType()->isVoidTy()) {
    RetVal = convertToInt64Ty(RetVal);
    Builder.CreateCall(RFIs[OMPRTL___kmpc_host_rpc_set_ret_val],
                       {Desc, RetVal});
  }
  Builder.CreateRetVoid();

  return WrapperFn;
}

void HostRPC::emitHostWrapperInvoker() {
  IRBuilder<> Builder(Context);
  unsigned NumEntries = HostEntryTable.size();
  Function *F = RFIs[OMPRTL___kmpc_host_rpc_invoke_host_wrapper];
  F->setDLLStorageClass(
      GlobalValue::DLLStorageClassTypes::DLLExportStorageClass);
  Value *CallNo = F->getArg(0);
  Value *Desc = F->getArg(1);

  SmallVector<BasicBlock *> SwitchBBs;
  BasicBlock *EntryBB = BasicBlock::Create(Context, "entry", F);
  BasicBlock *ReturnBB = BasicBlock::Create(Context, "return", F);

  // Emit code for the return bb.
  Builder.SetInsertPoint(ReturnBB);
  Builder.CreateRetVoid();

  // Create BB for each host entry and emit function call.
  for (unsigned I = 0; I < NumEntries; ++I) {
    BasicBlock *BB = BasicBlock::Create(Context, "invoke.bb", F, ReturnBB);
    SwitchBBs.push_back(BB);
    Builder.SetInsertPoint(BB);
    Builder.CreateCall(HostEntryTable[I], {Desc});
    Builder.CreateBr(ReturnBB);
  }

  // Emit code for the entry BB.
  Builder.SetInsertPoint(EntryBB);
  SwitchInst *Switch = Builder.CreateSwitch(CallNo, ReturnBB, NumEntries);
  for (unsigned I = 0; I < NumEntries; ++I)
    Switch->addCase(ConstantInt::get(cast<IntegerType>(Int32Ty), I),
                    SwitchBBs[I]);
}

Module *getHostModule(Module &M) {
  auto *MD = M.getNamedMetadata("llvm.hostrpc.hostmodule");
  if (!MD || MD->getNumOperands() == 0)
    return nullptr;
  auto *Node = MD->getOperand(0);
  assert(Node->getNumOperands() == 1 && "invliad named metadata");
  auto *CAM = dyn_cast<ConstantAsMetadata>(Node->getOperand(0));
  if (!CAM)
    return nullptr;
  auto *CI = cast<ConstantInt>(CAM->getValue());
  Module *Mod = reinterpret_cast<Module *>(CI->getZExtValue());
  M.eraseNamedMetadata(MD);
  return Mod;
}
} // namespace

PreservedAnalyses HostRPCPass::run(Module &M, ModuleAnalysisManager &AM) {
  std::unique_ptr<Module> DummyHostModule;

  Module *HostModule = nullptr;

  if (UseDummyHostModule) {
    DummyHostModule =
        std::make_unique<Module>("dummy-host-rpc.bc", M.getContext());
    HostModule = DummyHostModule.get();
  } else {
    HostModule = getHostModule(M);
  }

  if (!HostModule)
    return PreservedAnalyses::all();

  bool PostLink = LTOPhase == ThinOrFullLTOPhase::FullLTOPostLink ||
                  LTOPhase == ThinOrFullLTOPhase::ThinLTOPreLink;

  // The pass will not run if it is not invoked directly or not invoked at link
  // time.
  if (!UseDummyHostModule && !PostLink)
    return PreservedAnalyses::all();

  FunctionAnalysisManager &FAM =
      AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  AnalysisGetter AG(FAM);

  CallGraphUpdater CGUpdater;
  BumpPtrAllocator Allocator;

  AttributorConfig AC(CGUpdater);
  AC.DefaultInitializeLiveInternals = false;
  AC.RewriteSignatures = false;
  AC.PassName = DEBUG_TYPE;
  AC.MaxFixpointIterations = 1024;

  InformationCache InfoCache(M, AG, Allocator, /* CGSCC */ nullptr);

  SetVector<Function *> Functions;
  Attributor A(Functions, InfoCache, AC);

  HostRPC RPC(M, *HostModule, A);
  bool Changed = RPC.run();

  LLVM_DEBUG({
    if (Changed && UseDummyHostModule) {
      M.dump();
      HostModule->dump();
    }
  });

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
