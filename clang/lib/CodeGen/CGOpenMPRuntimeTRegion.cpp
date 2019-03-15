//===-- CGOpenMPRuntimeTRegion.cpp - OpenMP RT TRegion interface codegen --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of the code generation interface for OpenMP target offloading
// though the Target Region (TRegion) interface.
//
// See the file comment in CGOpenMPRuntimeTRegion.h for more information.
//
//===----------------------------------------------------------------------===//

#include "CGOpenMPRuntimeTRegion.h"
#include "CodeGenFunction.h"
#include "clang/AST/StmtVisitor.h"

using namespace clang;
using namespace CodeGen;

void CGOpenMPRuntimeTRegion::emitKernel(const OMPExecutableDirective &D,
                                        StringRef ParentName,
                                        llvm::Function *&OutlinedFn,
                                        llvm::Constant *&OutlinedFnID,
                                        const RegionCodeGenTy &CodeGen) {
  WrapperInfoMap.clear();

  // Emit target region as a standalone region.
  class KernelPrePostActionTy : public PrePostActionTy {
    CGOpenMPRuntimeTRegion &RT;
    llvm::BasicBlock *ExitBB;

  public:
    KernelPrePostActionTy(CGOpenMPRuntimeTRegion &RT)
        : RT(RT), ExitBB(nullptr) {}

    void Enter(CodeGenFunction &CGF) override {
      RT.emitKernelHeader(CGF, ExitBB);
      // Skip target region initialization.
      RT.setLocThreadIdInsertPt(CGF, /* AtCurrentPoint */ true);
    }

    void Exit(CodeGenFunction &CGF) override {
      RT.clearLocThreadIdInsertPt(CGF);
      RT.emitKernelFooter(CGF, ExitBB);
    }

  } Action(*this);
  CodeGen.setAction(Action);

  emitTargetOutlinedFunctionHelper(D, ParentName, OutlinedFn, OutlinedFnID,
                                   /* IsOffloadEntry */ true, CodeGen);
}

void CGOpenMPRuntimeTRegion::emitKernelHeader(CodeGenFunction &CGF,
                                              llvm::BasicBlock *&ExitBB) {
  CGBuilderTy &Bld = CGF.Builder;

  // Setup BBs in entry function.
  llvm::BasicBlock *ExecuteBB = CGF.createBasicBlock(".execute");
  ExitBB = CGF.createBasicBlock(".exit");

  llvm::Value *Args[] = {
      /* Ident */ llvm::Constant::getNullValue(getIdentTyPointerTy()),
      /* UseSPMDMode */ Bld.getInt1(isKnownSPMDMode()),
      /* UseStateMachine */ Bld.getInt1(1),
      /* RequiresOMPRuntime */
      Bld.getInt1(mayNeedRuntimeSupport()),
      /* RequiresDataSharing */ Bld.getInt1(mayPerformDataSharing())};
  llvm::CallInst *InitCI = CGF.EmitRuntimeCall(
      createTargetRuntimeFunction(OMPRTL__kmpc_target_region_kernel_init),
      Args);

  llvm::Value *ExecuteCnd = Bld.CreateICmpEQ(InitCI, Bld.getInt8(1));

  Bld.CreateCondBr(ExecuteCnd, ExecuteBB, ExitBB);
  CGF.EmitBlock(ExecuteBB);
}

void CGOpenMPRuntimeTRegion::emitKernelFooter(CodeGenFunction &CGF,
                                              llvm::BasicBlock *ExitBB) {
  if (!CGF.HaveInsertPoint())
    return;

  llvm::BasicBlock *OMPDeInitBB = CGF.createBasicBlock(".omp.deinit");
  CGF.EmitBranch(OMPDeInitBB);

  CGF.EmitBlock(OMPDeInitBB);

  CGBuilderTy &Bld = CGF.Builder;
  // DeInitialize the OMP state in the runtime; called by all active threads.
  llvm::Value *Args[] = {
      /* Ident */ llvm::Constant::getNullValue(getIdentTyPointerTy()),
      /* UseSPMDMode */ Bld.getInt1(isKnownSPMDMode()),
      /* RequiredOMPRuntime */
      Bld.getInt1(mayNeedRuntimeSupport())};

  CGF.EmitRuntimeCall(
      createTargetRuntimeFunction(OMPRTL__kmpc_target_region_kernel_deinit),
      Args);

  CGF.EmitBranch(ExitBB);
  CGF.EmitBlock(ExitBB);
}

void CGOpenMPRuntimeTRegion::emitTargetOutlinedFunction(
    const OMPExecutableDirective &D, StringRef ParentName,
    llvm::Function *&OutlinedFn, llvm::Constant *&OutlinedFnID,
    bool IsOffloadEntry, const RegionCodeGenTy &CodeGen) {
  if (!IsOffloadEntry) // Nothing to do.
    return;

  assert(!ParentName.empty() && "Invalid target region parent name!");

  emitKernel(D, ParentName, OutlinedFn, OutlinedFnID, CodeGen);

  // Create a unique global variable to indicate the execution mode of this
  // target region. The execution mode is either 'non-SPMD' or 'SPMD'. Initially
  // all regions are executed in non-SPMD mode. This variable is picked up by
  // the offload library to setup the device appropriately before kernel launch.
  auto *GVMode = new llvm::GlobalVariable(
      CGM.getModule(), CGM.Int8Ty, /* isConstant */ true,
      llvm::GlobalValue::WeakAnyLinkage, llvm::ConstantInt::get(CGM.Int8Ty, 1),
      Twine(OutlinedFn->getName(), "_exec_mode"));
  CGM.addCompilerUsedGlobal(GVMode);
}

CGOpenMPRuntimeTRegion::CGOpenMPRuntimeTRegion(CodeGenModule &CGM)
    : CGOpenMPRuntimeTarget(CGM) {
  if (!CGM.getLangOpts().OpenMPIsDevice)
    llvm_unreachable("TRegion code generation does only handle device code!");
}

llvm::Function *CGOpenMPRuntimeTRegion::emitParallelOutlinedFunction(
    const OMPExecutableDirective &D, const VarDecl *ThreadIDVar,
    OpenMPDirectiveKind InnermostKind, const RegionCodeGenTy &CodeGen) {

  // Emit target region as a standalone region.
  llvm::Function *OutlinedFun =
      cast<llvm::Function>(CGOpenMPRuntime::emitParallelOutlinedFunction(
          D, ThreadIDVar, InnermostKind, CodeGen));

  createParallelDataSharingWrapper(OutlinedFun, D);

  return OutlinedFun;
}

void CGOpenMPRuntimeTRegion::createParallelDataSharingWrapper(
    llvm::Function *OutlinedParallelFn, const OMPExecutableDirective &D) {
  ASTContext &Ctx = CGM.getContext();
  const auto &CS = *D.getCapturedStmt(OMPD_parallel);

  // Create a function that takes as argument the source thread.
  FunctionArgList WrapperArgs;
  ImplicitParamDecl SharedVarsArgDecl(Ctx, /* DC */ nullptr, D.getBeginLoc(),
                                      /* Id */ nullptr, Ctx.VoidPtrTy,
                                      ImplicitParamDecl::Other);
  ImplicitParamDecl PrivateVarsArgDecl(Ctx, /* DC */ nullptr, D.getBeginLoc(),
                                       /* Id */ nullptr, Ctx.VoidPtrTy,
                                       ImplicitParamDecl::Other);
  WrapperArgs.emplace_back(&SharedVarsArgDecl);
  WrapperArgs.emplace_back(&PrivateVarsArgDecl);

  const CGFunctionInfo &CGFI =
      CGM.getTypes().arrangeBuiltinFunctionDeclaration(Ctx.VoidTy, WrapperArgs);

  auto *WrapperFn = llvm::Function::Create(
      CGM.getTypes().GetFunctionType(CGFI), llvm::GlobalValue::InternalLinkage,
      Twine(OutlinedParallelFn->getName(), "_wrapper"), &CGM.getModule());
  CGM.SetInternalFunctionAttributes(GlobalDecl(), WrapperFn, CGFI);

  OutlinedParallelFn->setLinkage(llvm::GlobalValue::InternalLinkage);
  OutlinedParallelFn->setDoesNotRecurse();
  WrapperFn->setLinkage(llvm::GlobalValue::InternalLinkage);
  WrapperFn->setDoesNotRecurse();

  CodeGenFunction CGF(CGM, /* suppressNewContext */ true);
  CGF.StartFunction(GlobalDecl(), Ctx.VoidTy, WrapperFn, CGFI, WrapperArgs,
                    D.getBeginLoc(), D.getBeginLoc());

  auto AI = WrapperFn->arg_begin();
  llvm::Argument &SharedVarsArg = *(AI++);
  llvm::Argument &PrivateVarsArg = *(AI);
  SharedVarsArg.setName("shared_vars");
  PrivateVarsArg.setName("private_vars");

  setLocThreadIdInsertPt(CGF, /* AtCurrentPoint */ true);

  Address ThreadIDAddr = emitThreadIDAddress(CGF, D.getBeginLoc());
  Address ZeroAddr = CGF.CreateMemTemp(CGF.getContext().getIntTypeForBitwidth(
                                           /* DestWidth */ 32, /* Signed */ 1),
                                       /* Name */ ".zero.addr");
  CGF.InitTempAlloca(ZeroAddr, CGF.Builder.getInt32(/* C */ 0));

  // Create the array of arguments and fill it with boilerplate values.
  SmallVector<llvm::Value *, 8> Args;
  Args.emplace_back(ThreadIDAddr.getPointer());
  Args.emplace_back(ZeroAddr.getPointer());

  CGBuilderTy &Bld = CGF.Builder;

  // Collect all variables marked as shared.
  llvm::SmallPtrSet<const ValueDecl *, 16> SharedVars;
  for (const auto *C : D.getClausesOfKind<OMPSharedClause>())
    for (const Expr *E : C->getVarRefs())
      SharedVars.insert(CGOpenMPRuntimeTarget::getUnderlyingVar(E));

  // Retrieve the shared and private variables from argument pointers and pass
  // them to the outlined function.
  llvm::SmallVector<llvm::Type *, 8> SharedStructMemberTypes;
  llvm::SmallVector<llvm::Type *, 8> PrivateStructMemberTypes;

  WrapperInfo &WI = WrapperInfoMap[OutlinedParallelFn];
  WI.WrapperFn = WrapperFn;

  llvm::LLVMContext &LLVMCtx = OutlinedParallelFn->getContext();
  auto ArgIt = OutlinedParallelFn->arg_begin();
  ArgIt->addAttr(llvm::Attribute::get(
      LLVMCtx, llvm::Attribute::Dereferenceable,
      ThreadIDAddr.getElementType()->getScalarSizeInBits() / 8));
  ArgIt->addAttr(
      llvm::Attribute::get(LLVMCtx, llvm::Attribute::Alignment,
                           ThreadIDAddr.getAlignment().getQuantity()));
  ArgIt++;

  ArgIt->addAttr(llvm::Attribute::get(
      LLVMCtx, llvm::Attribute::Dereferenceable,
      ZeroAddr.getElementType()->getScalarSizeInBits() / 8));
  ArgIt->addAttr(llvm::Attribute::get(LLVMCtx, llvm::Attribute::Alignment,
                                      ZeroAddr.getAlignment().getQuantity()));
  ArgIt++;

  // If we require loop bounds they are already part of the outlined function
  // encoding, just after global_tid and bound_tid.
  bool RequiresLoopBounds =
      isOpenMPLoopBoundSharingDirective(D.getDirectiveKind());
  if (RequiresLoopBounds) {
    // Register the lower bound in the wrapper info.
    WI.CaptureIsPrivate.push_back(true);
    PrivateStructMemberTypes.push_back((ArgIt++)->getType());
    // Register the upper bound in the wrapper info.
    WI.CaptureIsPrivate.push_back(true);
    PrivateStructMemberTypes.push_back((ArgIt++)->getType());
  }

  auto CIt = CS.capture_begin();
  for (unsigned I = 0, E = CS.capture_size(); I < E; ++I, ++CIt) {
    bool IsPrivate = CIt->capturesVariableArrayType() ||
                     CIt->capturesVariableByCopy() ||
                     !SharedVars.count(CIt->getCapturedVar());
    WI.CaptureIsPrivate.push_back(IsPrivate);

    auto &StructMemberTypes =
        IsPrivate ? PrivateStructMemberTypes : SharedStructMemberTypes;
    llvm::Type *ArgTy = (ArgIt++)->getType();
    if (!IsPrivate) {
      assert(ArgTy->isPointerTy());
      ArgTy = ArgTy->getPointerElementType();
    }
    StructMemberTypes.push_back(ArgTy);
  }

  // Verify the position of the outlined function argument iterator as a sanity
  // check.
  assert(ArgIt == OutlinedParallelFn->arg_end() &&
         "Not all arguments have been processed!");

  llvm::Value *SharedVarsStructPtr = nullptr;
  llvm::Value *PrivateVarsStructPtr = nullptr;
  if (!PrivateStructMemberTypes.empty()) {
    WI.PrivateVarsStructTy = llvm::StructType::create(
        LLVMCtx, PrivateStructMemberTypes, "omp.private.struct");
    PrivateVarsStructPtr = Bld.CreateBitCast(
        &PrivateVarsArg, WI.PrivateVarsStructTy->getPointerTo());
  }
  if (!SharedStructMemberTypes.empty()) {
    WI.SharedVarsStructTy = llvm::StructType::create(
        LLVMCtx, SharedStructMemberTypes, "omp.shared.struct");
    SharedVarsStructPtr = Bld.CreateBitCast(
        &SharedVarsArg, WI.SharedVarsStructTy->getPointerTo());
  }

  assert(WI.CaptureIsPrivate.size() + /* global_tid & bound_tid */ 2 ==
             OutlinedParallelFn->arg_size() &&
         "Not all arguments have been processed!");

  unsigned PrivateIdx = 0, SharedIdx = 0;
  for (int i = 0, e = WI.CaptureIsPrivate.size(); i < e; i++) {
    bool IsPrivate = WI.CaptureIsPrivate[i];

    llvm::Value *StructPtr =
        IsPrivate ? PrivateVarsStructPtr : SharedVarsStructPtr;
    unsigned &Idx = IsPrivate ? PrivateIdx : SharedIdx;

    // TODO: Figure out the real alignment
    if (IsPrivate) {
      Args.emplace_back(
          Bld.CreateAlignedLoad(Bld.CreateStructGEP(StructPtr, Idx++), 1));
    } else {
      llvm::Value *GEP = Bld.CreateStructGEP(StructPtr, Idx++);
      Args.emplace_back(GEP);
    }
  }

  assert(Args.size() == OutlinedParallelFn->arg_size());
  emitOutlinedFunctionCall(CGF, D.getBeginLoc(), OutlinedParallelFn, Args);

  CGF.FinishFunction();

  clearLocThreadIdInsertPt(CGF);
}

void CGOpenMPRuntimeTRegion::emitParallelCall(
    CodeGenFunction &CGF, SourceLocation Loc, llvm::Function *Fn,
    ArrayRef<llvm::Value *> CapturedVars, const Expr *IfCond) {
  if (!CGF.HaveInsertPoint())
    return;

  const WrapperInfo &WI = WrapperInfoMap[Fn];

  auto &&ParGen = [this, CapturedVars, WI](CodeGenFunction &CGF,
                                           PrePostActionTy &) {
    CGBuilderTy &Bld = CGF.Builder;
    assert(WI.WrapperFn && "Wrapper function does not exist!");

    llvm::Value *SharedVarsSize = llvm::Constant::getNullValue(CGM.Int16Ty);
    llvm::Value *PrivateVarsSize = SharedVarsSize;
    llvm::Value *SharedStructAlloca = llvm::UndefValue::get(CGM.VoidPtrTy);
    llvm::Value *PrivateStructAlloca = SharedStructAlloca;

    if (WI.SharedVarsStructTy) {
      SharedStructAlloca = CGF.CreateDefaultAlignTempAlloca(
                                  WI.SharedVarsStructTy, ".shared.vars")
                               .getPointer();
      const llvm::DataLayout &DL = WI.WrapperFn->getParent()->getDataLayout();
      SharedVarsSize = Bld.getInt16(DL.getTypeAllocSize(WI.SharedVarsStructTy));
    }
    if (WI.PrivateVarsStructTy) {
      PrivateStructAlloca = CGF.CreateDefaultAlignTempAlloca(
                                   WI.PrivateVarsStructTy, ".private.vars")
                                .getPointer();
      const llvm::DataLayout &DL = WI.WrapperFn->getParent()->getDataLayout();
      PrivateVarsSize =
          Bld.getInt16(DL.getTypeAllocSize(WI.PrivateVarsStructTy));
    }

    llvm::SmallVector<llvm::Value *, 4> Args;
    Args.push_back(
        /* Ident */ llvm::Constant::getNullValue(getIdentTyPointerTy()));
    Args.push_back(
        /* UseSPMDMode */ Bld.getInt16(getExecutionMode()));
    Args.push_back(
        /* RequiredOMPRuntime */ Bld.getInt1(mayNeedRuntimeSupport()));
    Args.push_back(WI.WrapperFn);
    Args.push_back(CGF.EmitCastToVoidPtr(SharedStructAlloca));
    Args.push_back(SharedVarsSize);
    Args.push_back(CGF.EmitCastToVoidPtr(PrivateStructAlloca));
    Args.push_back(PrivateVarsSize);
    Args.push_back(
        /* SharedPointers */ Bld.getInt1(0));

    assert((CapturedVars.empty() ||
            (WI.SharedVarsStructTy || WI.PrivateVarsStructTy)) &&
           "Expected the shared or private struct type to be set if variables "
           "are captured!");
    assert((CapturedVars.empty() ||
            CapturedVars.size() ==
                (WI.SharedVarsStructTy ? WI.SharedVarsStructTy->getNumElements()
                                       : 0) +
                    (WI.PrivateVarsStructTy
                         ? WI.PrivateVarsStructTy->getNumElements()
                         : 0)) &&
           "# elements in shared struct types should be number of captured "
           "variables!");

    // Store all captured variables into a single local structure that is then
    // passed to the runtime library.
    unsigned PrivateIdx = 0, SharedIdx = 0;
    for (int i = 0, e = WI.CaptureIsPrivate.size(); i < e; i++) {
      bool IsPrivate = WI.CaptureIsPrivate[i];

      llvm::Value *StructPtr =
          IsPrivate ? PrivateStructAlloca : SharedStructAlloca;
      unsigned &Idx = IsPrivate ? PrivateIdx : SharedIdx;
      llvm::Value *GEP = Bld.CreateStructGEP(StructPtr, Idx++);
      llvm::Value *Var = IsPrivate ? CapturedVars[i]
                                   : Bld.CreateAlignedLoad(CapturedVars[i], 1);
      Bld.CreateDefaultAlignedStore(Var, GEP);
    }

    CGF.EmitRuntimeCall(
        createTargetRuntimeFunction(OMPRTL__kmpc_target_region_kernel_parallel),
        Args);

    SharedIdx = 0;
    for (int i = 0, e = WI.CaptureIsPrivate.size(); i < e; i++) {
      bool IsPrivate = WI.CaptureIsPrivate[i];
      if (IsPrivate)
        continue;

      llvm::Value *GEP = Bld.CreateStructGEP(SharedStructAlloca, SharedIdx++);
      llvm::Value *Var = Bld.CreateAlignedLoad(GEP, 1);
      Bld.CreateDefaultAlignedStore(Var, CapturedVars[i]);
    }
  };

  auto &&SeqGen = [this, &ParGen, Loc](CodeGenFunction &CGF,
                                       PrePostActionTy &Action) {
    // Use an artifical "num_threads(1)" clause to force sequential execution if
    // the expression in the 'if clause' evaluated to false. We expect the
    // middle-end to clean this up.
    emitNumThreadsClause(CGF, CGF.Builder.getInt32(/* C */ 1), Loc);
    ParGen(CGF, Action);
  };

  if (IfCond) {
    emitOMPIfClause(CGF, IfCond, ParGen, SeqGen);
  } else {
    CodeGenFunction::RunCleanupsScope Scope(CGF);
    RegionCodeGenTy ThenRCG(ParGen);
    ThenRCG(CGF);
  }
}

void CGOpenMPRuntimeTRegion::emitCriticalRegion(
    CodeGenFunction &CGF, StringRef CriticalName,
    const RegionCodeGenTy &CriticalOpGen, SourceLocation Loc,
    const Expr *Hint) {
  llvm_unreachable(
      "TODO: TRegion code generation does not support critical regions yet!");
}

void CGOpenMPRuntimeTRegion::emitReduction(
    CodeGenFunction &CGF, SourceLocation Loc, ArrayRef<const Expr *> Privates,
    ArrayRef<const Expr *> LHSExprs, ArrayRef<const Expr *> RHSExprs,
    ArrayRef<const Expr *> ReductionOps, ReductionOptionsTy Options) {
  llvm_unreachable(
      "TODO: TRegion code generation does not support reductions yet!");
}
