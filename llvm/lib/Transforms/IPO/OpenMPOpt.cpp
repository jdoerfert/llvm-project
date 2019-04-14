//===-- IPO/OpenMPOpt.cpp - Collection of OpenMP specific optimizations ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// OpenMP specific optimizations
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;

#define DEBUG_TYPE "openmp-opt"

static cl::opt<bool> BuildCustomStateMachines(
    "openmp-opt-kernel-state-machines", cl::ZeroOrMore,
    cl::desc("Build custom state machines for non-SPMD kernels."), cl::Hidden,
    cl::init(true));

static cl::opt<bool> PerformOpenMPSIMDIZATION(
    "openmp-opt-kernel-simdization", cl::ZeroOrMore,
    cl::desc("Convert non-SPMD kernels to SPMD mode if possible."), cl::Hidden,
    cl::init(true));

static cl::opt<bool> ForceOpenMPSIMDIZATION(
    "openmp-opt-kernel-force-simdization", cl::ZeroOrMore,
    cl::desc("Force execution of non-SPMD kernels in SPMD mode."), cl::Hidden,
    cl::init(false));

STATISTIC(NumKernelsConvertedToSPMD,
          "Number of GPU kernels converted to SPMD mode");
STATISTIC(NumParallelCallsConvertedToSPMD,
          "Number of parallel GPU kernel regions converted to SPMD mode");
STATISTIC(NumParallelCallsModeSpecialized,
          "Number of parallel GPU kernel regions notified of their mode");
STATISTIC(NumKernelsNonSPMDNoParallelism,
          "Number of GPU kernel in non-SPMD mode without parallelism");
STATISTIC(NumCustomStateMachinesCreated,
          "Number of custom GPU kernel non-SPMD mode state machines created");
STATISTIC(NumCustomStateMachinesNoFallback,
          "Number of custom GPU kernel non-SPMD mode state machines without "
          "fallback");

namespace {

/// Set of constants that describe the positions of arguments (ARG_FN_NAME) and
/// the meaning of return values (RET_FN_MEANING) for the target region kernel
/// interface. Has to be kept in sync with
///   openmp/libomptarget/deviceRTLs/common/target_region.h
/// and the respective implementations.
enum {
  ARG_INIT_IDENT = 0,
  ARG_INIT_USE_SPMD_MODE = 1,
  ARG_INIT_REQUIRES_OMP_RUNTIME = 2,
  ARG_INIT_USE_STATE_MACHINE = 3,
  ARG_INIT_REQUIRES_DATA_SHARING = 4,

  RET_INIT_IS_WORKER = -1,
  RET_INIT_IS_SURPLUS = 0,
  RET_INIT_IS_MASTER = 1,

  ARG_DEINIT_IDENT = 0,
  ARG_DEINIT_USE_SPMD_MODE = 1,
  ARG_DEINIT_REQUIRES_OMP_RUNTIME = 2,

  ARG_PARALLEL_IDENT = 0,
  ARG_PARALLEL_USE_SPMD_MODE = 1,
  ARG_PARALLEL_REQUIRES_OMP_RUNTIME = 2,
  ARG_PARALLEL_WORK_FUNCTION = 3,
  ARG_PARALLEL_SHARED_VARS = 4,
  ARG_PARALLEL_SHARED_VARS_BYTES = 5,
  ARG_PARALLEL_PRIVATE_VARS = 6,
  ARG_PARALLEL_PRIVATE_VARS_BYTES = 7,
  ARG_PARALLEL_SHARED_MEM_POINTERS = 8,

  ARG_REDUCTION_FINALIZE_IDENT = 0,
  ARG_REDUCTION_FINALIZE_USE_SPMD_MODE = 1,
  ARG_REDUCTION_FINALIZE_REQUIRES_OMP_RUNTIME = 2,
  ARG_REDUCTION_FINALIZE_GLOBAL_TID = 3,
  ARG_REDUCTION_FINALIZE_IS_PARALLEL_REDUCTION = 4,
  ARG_REDUCTION_FINALIZE_IS_TEAM_REDUCTION = 5,
  ARG_REDUCTION_FINALIZE_ORIGINAL_LOCATION = 6,
  ARG_REDUCTION_FINALIZE_REDUCTION_LOCATION = 7,
  ARG_REDUCTION_FINALIZE_NUM_REDUCTION_LOCATIONS = 8,
  ARG_REDUCTION_FINALIZE_REDUCTION_OPERATOR_KIND = 9,
  ARG_REDUCTION_FINALIZE_REDUCTION_BASE_TYPE = 10,
};

/// A macro list to represent known functions from the omp, __kmpc, and target
/// region interfaces. The first value is an enum identifier, see FunctionID,
/// the second value is the function name, and the third the expected number of
/// arguments.
#define KNOWN_FUNCTIONS()                                                      \
  KF(FID_OMP_GET_TEAM_NUM, "omp_get_team_num", 0)                              \
  KF(FID_OMP_GET_NUM_TEAMS, "omp_get_num_teams", 0)                            \
  KF(FID_OMP_GET_THREAD_NUM, "omp_get_thread_num", 0)                          \
  KF(FID_OMP_GET_NUM_THREADS, "omp_get_num_threads", 0)                        \
  KF(FID_OMP_SET_NUM_THREADS, "omp_set_num_threads", 1)                        \
  KF(FID_KMPC_TREGION_KERNEL_INIT, "__kmpc_target_region_kernel_init", 5)      \
  KF(FID_KMPC_TREGION_KERNEL_DEINIT, "__kmpc_target_region_kernel_deinit", 3)  \
  KF(FID_KMPC_TREGION_KERNEL_PARALLEL, "__kmpc_target_region_kernel_parallel", \
     9)                                                                        \
  KF(FID_KMPC_TREGION_KERNEL_REDUCTION_FINALIZE,                               \
     "__kmpc_target_region_kernel_reduction_finalize", 11)                      \
  KF(FID_KMPC_FOR_STATIC_INIT_4, "__kmpc_for_static_init_4", 9)                \
  KF(FID_KMPC_FOR_STATIC_FINI, "__kmpc_for_static_fini", 2)                    \
  KF(FID_KMPC_GLOBAL_THREAD_NUM, "__kmpc_global_thread_num", 1)                \
  KF(FID_KMPC_DISPATCH_INIT_4, "__kmpc_dispatch_init_4", 7)                    \
  KF(FID_KMPC_DISPATCH_NEXT_4, "__kmpc_dispatch_next_4", 6)

/// An identifier enum for each known function as well as the different kinds
/// of unknown functions we distinguish.
enum FunctionID {
#define KF(NAME, STR, NARGS) NAME,
  KNOWN_FUNCTIONS()
#undef KF
  // Unknown functions
  //{
  FID_KMPC_UNKNOWN, ///< unknown __kmpc_XXXX function
  FID_OMP_UNKOWN,   ///< unknown omp_XXX function
  FID_NVVM_UNKNOWN, ///< unknown llvm.nvvm.XXX function
  FID_LLVM_UNKNOWN, ///< unknown llvm.XXX function
  FID_UNKNOWN       ///< unknown function without known prefix.
  //}
};

static FunctionID getFunctionID(Function *F) {
  if (!F)
    return FID_UNKNOWN;
#define KF(NAME, STR, NARGS) .Case(STR, NAME)
  return StringSwitch<FunctionID>(F->getName()) KNOWN_FUNCTIONS()
      .StartsWith("__kmpc_", FID_KMPC_UNKNOWN)
      .StartsWith("omp_", FID_OMP_UNKOWN)
      .StartsWith("llvm.nvvm.", FID_NVVM_UNKNOWN)
      .StartsWith("llvm.", FID_LLVM_UNKNOWN)
      .Default(FID_UNKNOWN);
#undef KF
}

static Type *getOrCreateStructIdentTypePtr(Module &M) {
  // TODO create if not present!
  return M.getTypeByName("struct.ident_t")->getPointerTo();
}

// TODO: Simplify function declaration
static Function *getOrCreateFn(Type *RT, const char *Name, Module &M) {
  Function *Fn = M.getFunction(Name);
  if (!Fn) {
    FunctionType *FType = FunctionType::get(RT, {}, false);
    Fn =
        Function::Create(FType, llvm::GlobalVariable::ExternalLinkage, Name, M);
  }
  return Fn;
}
static Function *getOrCreateFn(Type *RT, Type *T0, Type *T1, const char *Name,
                               Module &M) {
  Function *Fn = M.getFunction(Name);
  if (!Fn) {
    FunctionType *FType = FunctionType::get(RT, {T0, T1}, false);
    Fn =
        Function::Create(FType, llvm::GlobalVariable::ExternalLinkage, Name, M);
  }
  return Fn;
}

static Function *getOrCreateSimpleSPMDBarrierFn(Module &M) {
  static const char *Name = "__kmpc_barrier_simple_spmd";
  Function *Fn = M.getFunction(Name);
  if (!Fn) {
    LLVMContext &Ctx = M.getContext();
    FunctionType *FType = FunctionType::get(
        Type::getVoidTy(Ctx),
        {getOrCreateStructIdentTypePtr(M), Type::getInt32Ty(Ctx)}, false);
    Fn =
        Function::Create(FType, llvm::GlobalVariable::ExternalLinkage, Name, M);
  }
  return Fn;
}

/// A helper class to introduce smart guarding code.
struct GuardGenerator {

  /// Inform the guard generator about the side-effect instructions collected in
  /// @p SideEffectInst.
  ///
  /// \Returns True if all registered side-effects can be (efficiently) guarded.
  bool registerSideEffects(SmallVectorImpl<Instruction *> &SideEffectInst) {
    bool Guarded = true;
    if (SideEffectInst.empty())
      return Guarded;

    const Module &M = *SideEffectInst.front()->getModule();
    const DataLayout &DL = M.getDataLayout();

    SmallVector<Instruction *, 16> UnguardedSideEffectInst;
    for (Instruction *I : SideEffectInst) {
      if (CallInst *CI = dyn_cast<CallInst>(I)) {
        if (getFunctionID(CI->getCalledFunction()) != FID_UNKNOWN)
          continue;
      } else if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
        if (isa<AllocaInst>(
                SI->getPointerOperand()->stripInBoundsConstantOffsets()))
          continue;
      } else if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
        if (isSafeToLoadUnconditionally(LI->getPointerOperand(),
                                        LI->getAlignment(), DL))
          continue;
      }
      LLVM_DEBUG(dbgs() << "Non-SPMD side effect found: " << *I << "\n");
      UnguardedSideEffectInst.push_back(I);
    }

    return UnguardedSideEffectInst.empty();
  }

  bool registerReadEffects(SmallVectorImpl<Instruction *> &ReadEffectInst) {
    return registerSideEffects(ReadEffectInst);
  }

  void introduceGuards() {
    // TODO: The guard generator cannot introduce guards yet but the registerXXX
    //       functions above are aware of that!
  }
};

/// Helper structure to represent and work with a target region kernel.
struct KernelTy {

  KernelTy(Function *KernelFn) : KernelFn(*KernelFn) {}

  /// Optimize this kernel, return true if something was done.
  bool optimize();

private:
  enum FunctionKind {
    FK_KERNEL, // The function is only called from the kernel function and never
               // from the outside world.
    FK_PARALLEL, // The function is only called from within parallel regions
                 // inside the kernel function and never from the outside world.
    FK_UNKNOWN, // The function might be called from the kernel function, from a
                // parallel region, or from the outside world.
  };

  /// Analyze this kernel, return true if successful.
  bool
  analyze(Function &F, SmallPtrSetImpl<Function *> &Visited,
          FunctionKind FnKind);

  /// Return true if the kernel is executed in SPMD mode.
  bool isExecutedInSPMDMode();

  /// Specialzie the "unknown mode" flag for parallel regions.
  bool specializeParallelRegionsModeFlag();

  /// Convert a non-SPMD mode kernel to SPMD mode, return true if successful.
  bool convertToSPMD();

  /// Create a custom state machine in the module, return true if successful.
  bool createCustomStateMachine();

  /// All side-effect instructions potentially executed in this kernel.
  SmallVector<Instruction *, 16> SideEffectInst;

  /// All read-only instructions potentially executed in this kernel.
  SmallVector<Instruction *, 16> ReadOnlyInst;

  /// All non-analyzed calls contained in this kernel. They are separated by
  /// their function ID which describes identifies known calls.
  SmallVector<CallInst *, 2> KernelCalls[FID_UNKNOWN + 1];

  /// All non-analyzed calls contained in parallel regions which are part of
  /// this kernel. They are separated by their function ID which describes
  /// identifies known calls.
  SmallVector<CallInst *, 2> ParallelRegionCalls[FID_UNKNOWN + 1];

  /// All non-analyzed calls contained in unknown regions which are part of
  /// this kernel. They are separated by their function ID which describes
  /// identifies known calls.
  SmallVector<CallInst *, 2> UnknownRegionCalls[FID_UNKNOWN + 1];

  /// The entry function of this kernel.
  Function &KernelFn;
};

bool KernelTy::analyze(Function &F, SmallPtrSetImpl<Function *> &Visited,
                       FunctionKind FnKind) {
  if (!Visited.insert(&F).second)
    return true;

  LLVM_DEBUG(dbgs() << "Analyze " << F.getName() << " as "
                    << (FnKind == FK_KERNEL
                            ? "internal kernel"
                            : (FnKind == FK_PARALLEL ? "internal parallel only"
                                                     : "general"))
                    << " function\n");

  // Determine the container to remember the call based on the function kind.
  auto &CallsArray =
      FnKind == FK_PARALLEL
          ? ParallelRegionCalls
          : (FnKind == FK_KERNEL ? KernelCalls : UnknownRegionCalls);

  for (Instruction &I : instructions(&F)) {

    // In parallel regions we only look for calls, outside, we look for all
    // side-effect and read-only instructions.
    if (FnKind != FK_PARALLEL) {
      // Handle non-side-effect instructions first. These will not write or
      // throw which makes reading the only interesting potential property.
      if (!I.mayHaveSideEffects()) {
        if (I.mayReadFromMemory()) {
          LLVM_DEBUG(dbgs() << "- read-only: " << I << "\n");
          ReadOnlyInst.push_back(&I);
        }
        continue;
      }

      // Now we handle all non-call instructions.
      if (!isa<CallInst>(I)) {
        LLVM_DEBUG(dbgs() << "- side-effect: " << I << "\n");
        SideEffectInst.push_back(&I);
        continue;
      }
    }

    if (!isa<CallInst>(I))
      continue;

    CallInst &CI = cast<CallInst>(I);
    Function *Callee = CI.getCalledFunction();
    FunctionID ID = getFunctionID(Callee);

    // For exact definitions of unknown functions we recurs.
    if (Callee && !Callee->isDeclaration() && Callee->isDefinitionExact() &&
        ID == FID_UNKNOWN) {
      // If the callee has external linkage we cannot keep the current function
      // kind but have to assume it is called from any possible context.
      FunctionKind CalleeFnKind =
          Callee->hasInternalLinkage() ? FnKind : FK_UNKNOWN;
      // If recursive analysis failed we bail, otherwise the
      // information was collected in the internal state.
      if (!analyze(*Callee, Visited, CalleeFnKind))
        return false;
      continue;
    }

    switch (ID) {
    // Check that know functions have the right number of arguments early on.
    // Additionally provide debug output based on the function ID.
#define KF(NAME, STR, NARGS)                                                   \
  case NAME:                                                                   \
    LLVM_DEBUG(                                                                \
        dbgs() << "- known call "                                              \
               << (CI.getNumArgOperands() != NARGS ? "[#arg missmatch!]" : "") \
               << ": " << I << "\n");                                          \
    if (CI.getNumArgOperands() != NARGS)                                       \
      ID = FID_UNKNOWN;                                                        \
    break;
      KNOWN_FUNCTIONS()
#undef KF
    case FID_KMPC_UNKNOWN:
      LLVM_DEBUG(dbgs() << "- unknown __kmpc_* call: " << I << "\n");
      break;
    case FID_OMP_UNKOWN:
      LLVM_DEBUG(dbgs() << "- unknown omp_* call: " << I << "\n");
      break;
    case FID_NVVM_UNKNOWN:
      LLVM_DEBUG(dbgs() << "- unknown llvm.nvvm.* call: " << I << "\n");
      break;
    case FID_LLVM_UNKNOWN:
      LLVM_DEBUG(dbgs() << "- unknown llvm.* call: " << I << "\n");
      break;
    case FID_UNKNOWN:
      LLVM_DEBUG(dbgs() << "- unknown call: " << I << "\n");
      break;
    }

    CallsArray[ID].push_back(&CI);
  }

  // If we did not analyze the kernel function but some other one down the call
  // chain we are done now.
  // TODO: Add more verification code here.
  if (&F != &KernelFn)
    return true;

  assert(&KernelCalls == &CallsArray);

  // If we are analyzing the kernel function we need to verify we have at least
  // the calls we expect to see in the right places.
  if (KernelCalls[FID_KMPC_TREGION_KERNEL_INIT].size() != 1 ||
      KernelCalls[FID_KMPC_TREGION_KERNEL_DEINIT].size() != 1 ||
      KernelCalls[FID_KMPC_TREGION_KERNEL_INIT].front()->getParent() !=
          &F.getEntryBlock()) {
    LLVM_DEBUG(dbgs() << "- malformed kernel: [#Init: "
                      << KernelCalls[FID_KMPC_TREGION_KERNEL_INIT].size()
                      << "][#DeInit: "
                      << KernelCalls[FID_KMPC_TREGION_KERNEL_DEINIT].size()
                      << "]\n");
    return false;
  }

  return true;
}

bool KernelTy::isExecutedInSPMDMode() {
  assert(KernelCalls[FID_KMPC_TREGION_KERNEL_INIT].size() == 1 &&
         "Non-canonical kernel form!");
  auto *SPMDFlag = cast<Constant>(
      KernelCalls[FID_KMPC_TREGION_KERNEL_INIT].front()->getArgOperand(
          ARG_INIT_USE_SPMD_MODE));
  assert(SPMDFlag->isZeroValue() || SPMDFlag->isOneValue());
  return SPMDFlag->isOneValue();
}

bool KernelTy::optimize() {
  bool Changed = false;

  // First analyze the code. If that fails for some reason we bail out early.
  SmallPtrSet<Function *, 8> Visited;
  if (!analyze(KernelFn, Visited, /* FnKind */ FK_KERNEL))
    return Changed;

  Visited.clear();
  for (CallInst *ParCI : KernelCalls[FID_KMPC_TREGION_KERNEL_PARALLEL]) {
    Value *ParCIParallelFnArg =
        ParCI->getArgOperand(ARG_PARALLEL_WORK_FUNCTION);
    Function *ParallelFn =
        dyn_cast<Function>(ParCIParallelFnArg->stripPointerCasts());

    // If the work function has external linkage we cannot assume it is only
    // used here but have to assume it is called from any possible context.
    FunctionKind FnKind =
        ParallelFn->hasInternalLinkage() ? FK_PARALLEL : FK_UNKNOWN;
    if (!ParallelFn ||
        !analyze(*ParallelFn, Visited, /* FnKind */ FnKind))
      return Changed;
  }

  Changed |= convertToSPMD();
  Changed |= createCustomStateMachine();
  Changed |= specializeParallelRegionsModeFlag();

  return Changed;
}

bool KernelTy::convertToSPMD() {
  if (isExecutedInSPMDMode())
    return false;

  bool Changed = false;

  // Use a generic guard generator to determine if suitable guards for all
  // side effect instructions can be placed.
  GuardGenerator GG;

  // Check if SIMDIZATION is possible, in case it is not forced.
  if (!ForceOpenMPSIMDIZATION) {
    // Unknown calls are not handled yet and will cause us to bail.
    if (!KernelCalls[FID_UNKNOWN].empty())
      return Changed;

    // If we cannot guard all side effect instructions bail out.
    if (!GG.registerSideEffects(SideEffectInst))
      return Changed;

    if (!GG.registerReadEffects(ReadOnlyInst))
      return Changed;

    // TODO: Emit a remark.
    LLVM_DEBUG(dbgs() << "Transformation to SPMD OK\n");

    // If we disabled SIMDIZATION we only emit the debug message and bail.
    if (!PerformOpenMPSIMDIZATION)
      return Changed;
  }

  // Actually emit the guard code after we decided to perform SIMDIZATION.
  GG.introduceGuards();

  // Create an "is-SPMD" flag.
  Type *InitFlagTy = KernelCalls[FID_KMPC_TREGION_KERNEL_INIT][0]
                         ->getArgOperand(ARG_INIT_USE_SPMD_MODE)
                         ->getType();
  Constant *InitSPMDFlag = ConstantInt::getTrue(InitFlagTy);

  // Update the init and deinit calls with the "is-SPMD" flag to indicate
  // SPMD mode.
  assert(KernelCalls[FID_KMPC_TREGION_KERNEL_INIT].size() == 1 &&
         "Non-canonical kernel form!");
  assert(KernelCalls[FID_KMPC_TREGION_KERNEL_DEINIT].size() == 1 &&
         "Non-canonical kernel form!");
  KernelCalls[FID_KMPC_TREGION_KERNEL_INIT][0]->setArgOperand(
      ARG_INIT_USE_SPMD_MODE, InitSPMDFlag);
  KernelCalls[FID_KMPC_TREGION_KERNEL_DEINIT][0]->setArgOperand(
      ARG_DEINIT_USE_SPMD_MODE, InitSPMDFlag);

  // Use the simple barrier to synchronize all threads in SPMD mode after each
  // parallel region.
  Function *SimpleBarrierFn =
      getOrCreateSimpleSPMDBarrierFn(*KernelFn.getParent());

  // For each parallel region, identified by the
  // __kmpc_target_region_kernel_parallel call, we set the "is-SPMD" flag and
  // introduce a succeeding barrier call.
  Type *ParallelFlagTy = KernelCalls[FID_KMPC_TREGION_KERNEL_PARALLEL][0]
                     ->getArgOperand(ARG_PARALLEL_USE_SPMD_MODE)
                     ->getType();
  Constant *ParallelSPMDFlag = ConstantInt::getSigned(ParallelFlagTy, 1);
  for (CallInst *ParCI : KernelCalls[FID_KMPC_TREGION_KERNEL_PARALLEL]) {
    ParCI->setArgOperand(ARG_PARALLEL_USE_SPMD_MODE, ParallelSPMDFlag);
    auto AI = SimpleBarrierFn->arg_begin();
    CallInst::Create(SimpleBarrierFn,
                     {Constant::getNullValue((AI++)->getType()),
                      Constant::getNullValue((AI)->getType())},
                     "", ParCI->getNextNode());
  }

  // TODO: serialize nested parallel regions

  // TODO: Adjust the schedule parameters

  // Finally, we change the global exec_mode variable to indicate SPMD mode.
  GlobalVariable *ExecMode = KernelFn.getParent()->getGlobalVariable(
      (KernelFn.getName() + "_exec_mode").str());
  assert(ExecMode &&
         "Assumed to find an execution mode hint among the globals");
  assert(ExecMode->getInitializer()->isOneValue() &&
         "Assumed target_region execution mode prior to 'SPMD'-zation");
  ExecMode->setInitializer(
      Constant::getNullValue(ExecMode->getInitializer()->getType()));

  // Bookkeeping
  NumKernelsConvertedToSPMD++;
  NumParallelCallsConvertedToSPMD +=
      KernelCalls[FID_KMPC_TREGION_KERNEL_PARALLEL].size();

  return Changed;
}

bool KernelTy::createCustomStateMachine() {
  if (isExecutedInSPMDMode())
    return false;

  // TODO: Warn or eliminate the offloading if no parallel regions are present.
  // TODO: Use reachability to eliminate the loop and if-cascade
  //
  // The user module code looks as follows if this function returns true.
  //
  //   ThreadKind = __kmpc_target_region_kernel_init(...)
  //   if (ThreadKind == -1) {               //  actual worker thread
  //     do {
  //       __kmpc_barrier_simple_spmd(...)
  //       void *WorkFn;
  //       bool IsActive = __kmpc_kernel_parallel(&WorkFn, ...);
  //       if (!WorkFn)
  //         goto exit;
  //       if (IsActive) {
  //         char *SharedVars = __kmpc_target_region_kernel_get_shared_memory();
  //         char *PrivateVars =
  //         __kmpc_target_region_kernel_get_private_memory();
  //
  //         ((ParallelWorkFnTy)WorkFn)(SharedVars, PrivateVars);
  //
  //         __kmpc_kernel_end_parallel();
  //       }
  //       __kmpc_barrier_simple_spmd(...)
  //     } while (true);
  //   } else if (ThreadKind == 0) {         // surplus worker thread
  //     goto exit;
  //   } else {                              //    team master thread
  //     goto user_code;
  //   }

  if (KernelCalls[FID_KMPC_TREGION_KERNEL_PARALLEL].size() == 0) {
    LLVM_DEBUG(dbgs() << "Will not build a custom state machine because there "
                         "are no known parallel regions in the kernel.\n");
    // TODO: If we also know there are no hidden parallel calls we can terminate
    // all but the
    //       master thread right away.
    NumKernelsNonSPMDNoParallelism++;
    return false;
  }

  assert(KernelCalls[FID_KMPC_TREGION_KERNEL_INIT].size() == 1 &&
         "Non-canonical kernel form!");
  CallInst *InitCI = KernelCalls[FID_KMPC_TREGION_KERNEL_INIT][0];

  // Check if a custom state machine was already implemented.
  auto *UseSM =
      dyn_cast<ConstantInt>(InitCI->getArgOperand(ARG_INIT_USE_STATE_MACHINE));
  if (!UseSM || !UseSM->isOne()) {
    LLVM_DEBUG(dbgs() << "Will not build a custom state machine because of "
                      << *KernelCalls[FID_KMPC_TREGION_KERNEL_INIT][0] << "\n");
    return false;
  }

  InitCI->setName("thread_kind");
  LLVMContext &Ctx = InitCI->getContext();

  // Create local storage for the work function pointer.
  Type *VoidPtrTy = Type::getInt8PtrTy(Ctx);
  AllocaInst *WorkFnAI = new AllocaInst(VoidPtrTy, 0, "work_fn.addr",
                                        &KernelFn.getEntryBlock().front());

  Instruction *IP = InitCI->getNextNode();

  Type *FlagTy = InitCI->getArgOperand(ARG_INIT_USE_STATE_MACHINE)->getType();
  Constant *SMFlag = ConstantInt::getFalse(FlagTy);
  InitCI->setArgOperand(ARG_INIT_USE_STATE_MACHINE, SMFlag);

  // Check the return value of __kmpc_target_region_kernel_init. First compare
  // it to RET_INIT_IS_WORKER.
  Instruction *WorkerCnd = new ICmpInst(
      IP, ICmpInst::ICMP_EQ, InitCI,
      ConstantInt::getSigned(InitCI->getType(), RET_INIT_IS_WORKER),
      "is_worker");

  // Create the conditional which is entered by worker threads.
  Instruction *WaitTI = SplitBlockAndInsertIfThen(WorkerCnd, IP, false);
  BasicBlock *WaitBB = WaitTI->getParent();
  WaitBB->setName("worker.wait");
  IP->getParent()->setName("master_check");

  Instruction *MasterCheckTI = IP->getParent()->getTerminator();
  assert(MasterCheckTI->getNumSuccessors() == 2);
  assert(WaitTI->getNumSuccessors() == 1);

  // Determine the final block, that is a trivial one where the kernel ends.
  BasicBlock *FinalBB = nullptr;
  if (MasterCheckTI->getSuccessor(0)->size() == 1 &&
      isa<ReturnInst>(MasterCheckTI->getSuccessor(0)->getTerminator()))
    FinalBB = MasterCheckTI->getSuccessor(0);
  else if (MasterCheckTI->getSuccessor(1)->size() == 1 &&
           isa<ReturnInst>(MasterCheckTI->getSuccessor(1)->getTerminator()))
    FinalBB = MasterCheckTI->getSuccessor(1);
  assert(FinalBB && "Could not determine the final kernal block.");

  // Use the simple barrier to synchronize all threads in SPMD mode after each
  // parallel region.
  Module &M = *KernelFn.getParent();
  Function *SimpleBarrierFn = getOrCreateSimpleSPMDBarrierFn(M);

  auto AI = SimpleBarrierFn->arg_begin();
  Instruction *BarrierCall =
      CallInst::Create(SimpleBarrierFn,
                       {Constant::getNullValue((AI++)->getType()),
                        Constant::getNullValue((AI)->getType())},
                       "", WaitTI);

  Function *KernelParallelFn =
      getOrCreateFn(Type::getInt1Ty(Ctx), VoidPtrTy->getPointerTo(),
                    Type::getInt16Ty(Ctx), "__kmpc_kernel_parallel", M);

  Value *RequiresOMPRuntime = CastInst::CreateZExtOrBitCast(
      InitCI->getArgOperand(ARG_INIT_REQUIRES_OMP_RUNTIME),
      Type::getInt16Ty(Ctx), "", WaitTI);
  Instruction *ActiveCnd = CallInst::Create(
      KernelParallelFn, {WorkFnAI, RequiresOMPRuntime}, "is_active", WaitTI);

  Type *WorkFnPrototype =
      FunctionType::get(Type::getVoidTy(Ctx), {VoidPtrTy, VoidPtrTy}, false)
          ->getPointerTo();
  Value *WorkFnAICast = BitCastInst::CreatePointerBitCastOrAddrSpaceCast(
      WorkFnAI, WorkFnPrototype->getPointerTo(), "Work_fn.addr_cast", WaitTI);
  Value *WorkFn = new LoadInst(WorkFnAICast, "work_fn", WaitTI);

  Instruction *WorkFnCnd =
      new ICmpInst(WaitTI, ICmpInst::ICMP_EQ, WorkFn,
                   Constant::getNullValue(WorkFn->getType()), "no_work");

  Instruction *FinishedTI = SplitBlockAndInsertIfThen(WorkFnCnd, WaitTI, false);
  FinishedTI->getParent()->setName("worker.finished");
  WaitTI->getParent()->setName("worker.active_check");

  Instruction *ActiveTI = SplitBlockAndInsertIfThen(ActiveCnd, WaitTI, false);
  ActiveTI->getParent()->setName("worker.active");
  WaitTI->getParent()->setName("worker.inactive");

  Function *KernelGetSharedVars = getOrCreateFn(
      VoidPtrTy, "__kmpc_target_region_kernel_get_shared_memory", M);
  Value *SharedVars = CallInst::Create(KernelGetSharedVars, "", ActiveTI);
  Function *KernelGetPrivateVars = getOrCreateFn(
      VoidPtrTy, "__kmpc_target_region_kernel_get_private_memory", M);
  Value *PrivateVars = CallInst::Create(KernelGetPrivateVars, "", ActiveTI);

  BasicBlock *ExecuteBB = ActiveTI->getParent();
  BasicBlock *ParallelEndBB = SplitBlock(ExecuteBB, ActiveTI);
  ParallelEndBB->setName("worker.parallel_end");

  Function *KernelEndParallelFn =
      getOrCreateFn(Type::getVoidTy(Ctx), "__kmpc_kernel_end_parallel", M);
  CallInst::Create(KernelEndParallelFn, "", ActiveTI);

  // A fallback is required if we might not see all parallel regions
  // (__kmpc_target_region_kernel_parallel calls). This could be the case if
  // there is an unknown function call with side effects in the target region
  // or inside one of the parallel regions.
  bool RequiresFallback = !KernelCalls[FID_UNKNOWN].empty() ||
                          !ParallelRegionCalls[FID_UNKNOWN].empty() ||
                          !UnknownRegionCalls[FID_UNKNOWN].empty();

  // Collect all target region parallel calls
  // (__kmpc_target_region_kernel_parallel).
  SmallVector<CallInst *, 16> KernelParallelCalls;
  KernelParallelCalls.append(
      KernelCalls[FID_KMPC_TREGION_KERNEL_PARALLEL].begin(),
      KernelCalls[FID_KMPC_TREGION_KERNEL_PARALLEL].end());
  KernelParallelCalls.append(
      ParallelRegionCalls[FID_KMPC_TREGION_KERNEL_PARALLEL].begin(),
      ParallelRegionCalls[FID_KMPC_TREGION_KERNEL_PARALLEL].end());

  IP = ExecuteBB->getTerminator();

  // For each parallel call create a conditional that compares the work function
  // against the parallel work function of this parallel call, if available. If
  // the function pointers are equal we call the known parallel call work
  // function directly and continue to the end of the if-cascade.
  for (CallInst *ParCI : KernelParallelCalls) {
    Function *ParFn = dyn_cast<Function>(
        ParCI->getArgOperand(ARG_PARALLEL_WORK_FUNCTION)->stripPointerCasts());
    if (!ParFn) {
      LLVM_DEBUG(
          dbgs() << "Require fallback due to unknown parallel function\n");
      RequiresFallback = true;
      continue;
    }

    Value *ParFnCnd =
        new ICmpInst(IP, ICmpInst::ICMP_EQ, WorkFn, ParFn, "par_fn_check");
    Instruction *ParFnTI = SplitBlockAndInsertIfThen(ParFnCnd, IP, false);
    IP->getParent()->setName("worker.check.next");
    ParFnTI->getParent()->setName("worker.execute." + ParFn->getName());
    CallInst::Create(ParFn, {SharedVars, PrivateVars}, "", ParFnTI);
    ParFnTI->setSuccessor(0, ParallelEndBB);
  }

  // If a fallback is required we emit a indirect call before we jump to the
  // point where all cases converge.
  if (RequiresFallback)
    CallInst::Create(WorkFn, {SharedVars, PrivateVars}, "", IP);

  // Insert a barrier call at the convergence point, right before the back edge.
  BarrierCall->clone()->insertBefore(WaitTI);

  // Rewire the CFG edges to introduce the back and exit edge of the new loop.
  // TODO: Add the new loop to LI!
  FinishedTI->setSuccessor(0, FinalBB);
  WaitTI->setSuccessor(0, WaitBB);

  // Bookkeeping.
  NumCustomStateMachinesCreated++;
  NumCustomStateMachinesNoFallback += !RequiresFallback;

  return true;
}

bool KernelTy::specializeParallelRegionsModeFlag() {
  bool IsSPMDMode = isExecutedInSPMDMode();

  // Create an "is-SPMD" flag.
  Type *FlagTy = nullptr;

  if (!KernelCalls[FID_KMPC_TREGION_KERNEL_PARALLEL].empty()) {
    FlagTy = KernelCalls[FID_KMPC_TREGION_KERNEL_PARALLEL][0]
                 ->getArgOperand(ARG_PARALLEL_USE_SPMD_MODE)
                 ->getType();
  } else if (!KernelCalls[FID_KMPC_TREGION_KERNEL_REDUCTION_FINALIZE].empty()) {
    FlagTy = KernelCalls[FID_KMPC_TREGION_KERNEL_REDUCTION_FINALIZE][0]
                 ->getArgOperand(ARG_REDUCTION_FINALIZE_USE_SPMD_MODE)
                 ->getType();
  } else {
    // No calls found for which we could specialize the UseSPMD flag.
    return false;
  }

  Constant *SPMDFlag = ConstantInt::getSigned(FlagTy, IsSPMDMode);
  for (CallInst *ParCI : KernelCalls[FID_KMPC_TREGION_KERNEL_PARALLEL])
    ParCI->setArgOperand(ARG_PARALLEL_USE_SPMD_MODE, SPMDFlag);
  for (CallInst *ParCI : ParallelRegionCalls[FID_KMPC_TREGION_KERNEL_PARALLEL])
    ParCI->setArgOperand(ARG_PARALLEL_USE_SPMD_MODE, SPMDFlag);

  for (CallInst *RedCI : KernelCalls[FID_KMPC_TREGION_KERNEL_REDUCTION_FINALIZE])
    RedCI->setArgOperand(ARG_REDUCTION_FINALIZE_USE_SPMD_MODE, SPMDFlag);
  for (CallInst *RedCI : ParallelRegionCalls[FID_KMPC_TREGION_KERNEL_REDUCTION_FINALIZE])
    RedCI->setArgOperand(ARG_REDUCTION_FINALIZE_USE_SPMD_MODE, SPMDFlag);

  unsigned NumChangedCalls =
      KernelCalls[FID_KMPC_TREGION_KERNEL_PARALLEL].size() +
      ParallelRegionCalls[FID_KMPC_TREGION_KERNEL_PARALLEL].size();

  NumParallelCallsModeSpecialized += NumChangedCalls;
  return NumChangedCalls;
}

template <class T>
static void collectCallersOf(Module &M, StringRef Name,
                             SmallVectorImpl<T> &Callers) {
  Function *Callee = M.getFunction(Name);

  // If the callee function is not present, we are done.
  if (!Callee)
    return;

  // If it exists we check all users.
  for (const Use &U : Callee->uses()) {
    CallSite CS(U.getUser());

    // Filter out non-callee uses and non-call uses.
    if (!CS || !CS.isCallee(&U) || !isa<CallInst>(CS.getInstruction()))
      continue;

    // Found a caller, use it to create a T type object and put the result
    // in the Callers vector.
    Callers.emplace_back(T(CS.getCaller()));
  }
}

/// OpenMPOpt - The interprocedural OpenMP optimization pass implementation.
struct OpenMPOpt {

  bool runOnModule(Module &M) {
    bool Changed = false;

    // Collect target regions kernels identified by a call to
    // __kmpc_target_region_kernel_init.
    collectCallersOf(M, "__kmpc_target_region_kernel_init", TRKernels);

    for (KernelTy &K : TRKernels)
      Changed |= K.optimize();

    return Changed;
  }

private:
  /// A collection of all target regions kernels we found.
  SmallVector<KernelTy, 4> TRKernels;
};

// TODO: This could be a CGSCC pass as well.
struct OpenMPOptLegacy : public ModulePass {
  static char ID; // Pass identification, replacement for typeid
  OpenMPOpt OMPOpt;

  OpenMPOptLegacy() : ModulePass(ID) {
    initializeOpenMPOptLegacyPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {}

  bool runOnModule(Module &M) override { return OMPOpt.runOnModule(M); }
};

// TODO: Add a new PM entry point.

} // namespace

char OpenMPOptLegacy::ID = 0;

INITIALIZE_PASS_BEGIN(OpenMPOptLegacy, "openmp-opt",
                      "OpenMP specific optimizations", false, false)
INITIALIZE_PASS_END(OpenMPOptLegacy, "openmp-opt",
                    "OpenMP specific optimizations", false, false)

Pass *llvm::createOpenMPOptLegacyPass() { return new OpenMPOptLegacy(); }
